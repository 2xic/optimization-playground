import os
import sys
import time
import json
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore", message=".*TripleDES.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def worker():
    import torch.distributed as dist
    from datetime import timedelta
    from torch.amp import autocast
    from training.model import (
        Config,
        Model,
        TransformerLayerType,
        PositionalEmbeddingType,
    )
    from training.objectives import NextTokenPrediction
    from training.trainer import (
        apply_runtime_optimizations,
        best_autocast_dtype,
        GradScalerTrainer,
        TrainingOptions,
        DistributedStrategy,
    )
    from training.optimizer import (
        AdamConfig,
        AdamWConfig,
        MuonConfig,
        RMSpropConfig,
    )
    from optimization_playground_shared.nlp.utils.sampling import temperature_sampling

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    device = torch.device(f"cuda:{local}")
    apply_runtime_optimizations()

    strategy = os.environ["STRATEGY"]
    strat_map = {
        "single": DistributedStrategy.NONE,
        "ddp": DistributedStrategy.DDP,
        "fsdp": DistributedStrategy.FSDP,
    }
    dist_strategy = strat_map[strategy]
    compile_on = os.environ["COMPILE"] == "1"
    dim = int(os.environ["DIM"])
    layers = int(os.environ["LAYERS"])
    vocab = int(os.environ["VOCAB"])
    seq = int(os.environ["SEQ"])
    batch = int(os.environ["BATCH"])
    steps = int(os.environ["STEPS"])
    warmup = int(os.environ["WARMUP"])
    result = os.environ["RESULT_FILE"]

    heads = max(1, dim // 64)
    config = Config(
        dropout=0,
        dim_embeddings=dim,
        num_attention_heads=heads,
        num_transformer_layers=layers,
        vocab_size=vocab,
        sequence_length=seq,
        padding_index=0,
        transformer_layer=TransformerLayerType.GPT2,
        positional_embedding=PositionalEmbeddingType.NN_EMBEDDING,
        gradient_checkpointing=False,
    )
    model = Model(config)
    params = sum(p.numel() for p in model.parameters())
    mp_dtype = best_autocast_dtype()

    criterion = NextTokenPrediction(
        padding_index=0, vocab_size=vocab, sampler=temperature_sampling
    )
    opt_configs = {
        "adam": AdamConfig,
        "adamw": AdamWConfig,
        "muon": MuonConfig,
        "rmsprop": RMSpropConfig,
    }
    opt_config = opt_configs[os.environ.get("OPTIMIZER", "adam")]()
    optimizer = opt_config.create_optimizer(model.parameters())
    trainer = GradScalerTrainer(model, criterion, optimizer)
    training_options = TrainingOptions(
        batch_size=batch, device=device,
        distributed_strategy=dist_strategy, optimizer=opt_config,
    )
    trainer._training_options = training_options

    is_fsdp = dist_strategy == DistributedStrategy.FSDP and dist.is_initialized()
    if is_fsdp:
        trainer._original_model.to(device)
    else:
        trainer._original_model.to(device).to(trainer.type)

    err = None
    if compile_on:
        try:
            trainer.maybe_compile(training_options)
        except Exception as e:
            err = f"compile-init: {e}"
    trainer.apply_distributed_strategy(training_options, is_fsdp)
    trainer.model.train()
    torch.cuda.reset_peak_memory_stats(device)
    model = trainer.model
    optimizer = trainer.optimizer

    def run_step():
        X = torch.randint(0, vocab, (batch, seq), device=device)
        y = torch.randint(0, vocab, (batch, seq), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=mp_dtype):
            loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    try:
        for _ in range(warmup):
            run_step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            run_step()
        torch.cuda.synchronize()
        dt = time.time() - t0
    except Exception as e:
        if rank == 0:
            with open(result, "w") as f:
                json.dump({"error": f"run: {e}"}, f)
        dist.destroy_process_group()
        return

    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device) / 1e9, device=device
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    world = dist.get_world_size()
    step_s = dt / steps
    global_tokens = batch * seq * world
    tokens_per_s = global_tokens / step_s
    tflops = 6 * params * global_tokens / step_s / 1e12
    if rank == 0:
        with open(result, "w") as f:
            json.dump(
                {
                    "params_m": round(params / 1e6, 1),
                    "step_ms": round(step_s * 1000, 1),
                    "tokens_per_s": round(tokens_per_s),
                    "peak_gb": round(float(peak.item()), 2),
                    "tflops": round(tflops, 2),
                    "dtype": str(mp_dtype).replace("torch.", ""),
                    "warn": err,
                },
                f,
            )
    dist.barrier()
    dist.destroy_process_group()


def driver():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--optimizer", type=str, default="adam")
    args = parser.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {name} | dim={args.dim} layers={args.layers} seq={args.seq} "
          f"batch={args.batch} steps={args.steps} nproc(fsdp)={args.nproc}")
    print("-" * 78)
    print(f"{'strategy':>12s} {'compile':>8s} {'ms/step':>9s} {'tok/s':>10s} "
          f"{'TFLOP/s':>8s} {'peak/card':>10s} {'note':>6s}")

    cells = [
        ("single", 0, 1), ("single", 1, 1),
        ("ddp", 0, args.nproc), ("ddp", 1, args.nproc),
        ("fsdp", 0, args.nproc), ("fsdp", 1, args.nproc),
    ]
    runs = {}
    for label, comp, nproc in cells:
        fd, result = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(result)
        env = dict(os.environ)
        env.update(
            WORKER="1", STRATEGY=label, COMPILE=str(comp),
            DIM=str(args.dim), LAYERS=str(args.layers), VOCAB=str(args.vocab),
            SEQ=str(args.seq), BATCH=str(args.batch), STEPS=str(args.steps),
            WARMUP=str(args.warmup), RESULT_FILE=result, OPTIMIZER=args.optimizer,
        )
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={nproc}",
            os.path.abspath(__file__),
        ]
        try:
            proc = subprocess.run(cmd, env=env, timeout=args.timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"{label:>12s} {comp:>8d} {'timeout':>9s}")
            continue
        if not os.path.exists(result):
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
            print(f"{label:>12s} {comp:>8d} {'FAIL':>9s}")
            print(tail)
            continue
        with open(result) as f:
            r = json.load(f)
        os.remove(result)
        runs[(label, comp)] = r
        if "error" in r:
            print(f"{label:>12s} {comp:>8d} {'ERROR':>9s}  {r['error'][:50]}")
            continue
        note = "!" if r.get("warn") else ""
        print(f"{label:>12s} {comp:>8d} {r['step_ms']:>8.1f}  {r['tokens_per_s']:>10d} "
              f"{r['tflops']:>8.2f} {r['peak_gb']:>9.2f}G {note:>6s}")

    print("-" * 78)
    for label in ("single", "ddp", "fsdp"):
        e, c = runs.get((label, 0)), runs.get((label, 1))
        if e and c and "tokens_per_s" in e and "tokens_per_s" in c:
            s = c["tokens_per_s"] / max(e["tokens_per_s"], 1)
            print(f"{label}: compile is {s:.3f}x ({(s - 1) * 100:+.1f}% tok/s)")
    for w in [r for r in runs.values() if r.get("warn")]:
        print(f"warn: {w['warn']}")


if __name__ == "__main__":
    if os.environ.get("WORKER") == "1":
        worker()
    else:
        driver()
