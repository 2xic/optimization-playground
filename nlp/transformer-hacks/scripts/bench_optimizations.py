import os
import sys
import time
import json
import subprocess
import warnings

warnings.filterwarnings("ignore", message=".*TripleDES.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from training.model import (
    Config,
    Model,
    TransformerLayerType,
    PositionalEmbeddingType,
)
from training.objectives import NextTokenPrediction
from training.trainer import apply_runtime_optimizations, best_autocast_dtype
from training.device_caps import supports_torch_compile
from training.optimizer import AdamConfig, AdamWConfig, MuonConfig, RMSpropConfig
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling


def worker():
    ckpt = os.environ["CKPT"] == "1"
    dim = int(os.environ.get("DIM", 768))
    layers = int(os.environ.get("LAYERS", 12))
    vocab = int(os.environ.get("VOCAB", 50304))
    seq = int(os.environ.get("SEQ", 256))
    batch = int(os.environ.get("BATCH", 8))
    steps = int(os.environ.get("STEPS", 50))
    warmup = int(os.environ.get("WARMUP", 5))
    result = os.environ["RESULT_FILE"]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    apply_runtime_optimizations()

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
        gradient_checkpointing=ckpt,
    )
    model = Model(config).to(device).train()
    params = sum(p.numel() for p in model.parameters())
    if os.environ.get("COMPILE") == "1" and supports_torch_compile():
        model = torch.compile(model)
    torch.cuda.reset_peak_memory_stats(device)
    criterion = NextTokenPrediction(
        padding_index=0, vocab_size=vocab, sampler=temperature_sampling
    )
    opt_configs = {
        "adam": AdamConfig,
        "adamw": AdamWConfig,
        "muon": MuonConfig,
        "rmsprop": RMSpropConfig,
    }
    optimizer = opt_configs[os.environ.get("OPTIMIZER", "adam")]().create_optimizer(
        model.parameters()
    )
    mp_dtype = best_autocast_dtype()

    def run_step():
        X = torch.randint(0, vocab, (batch, seq), device=device)
        y = torch.randint(0, vocab, (batch, seq), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=mp_dtype):
            loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        run_step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        run_step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    step_s = dt / steps
    tokens_per_s = batch * seq / step_s
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    with open(result, "w") as f:
        json.dump(
            {
                "params_m": round(params / 1e6, 1),
                "step_ms": round(step_s * 1000, 1),
                "tokens_per_s": round(tokens_per_s),
                "peak_gb": round(peak_gb, 2),
            },
            f,
        )


def driver():
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="adam")
    args = parser.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {name} | dim={args.dim} layers={args.layers} "
          f"seq={args.seq} batch={args.batch} steps={args.steps}")
    print("-" * 60)

    runs = {}
    for comp in (0, 1):
        fd, result = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        env = dict(os.environ)
        env.update(
            WORKER="1", CKPT="0", COMPILE=str(comp), DIM=str(args.dim),
            LAYERS=str(args.layers), SEQ=str(args.seq), BATCH=str(args.batch),
            STEPS=str(args.steps), WARMUP=str(args.warmup), RESULT_FILE=result,
            OPTIMIZER=args.optimizer,
        )
        subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)
        with open(result) as f:
            runs[comp] = json.load(f)
        os.remove(result)

    eager, comp = runs[0], runs[1]
    print(f"{'mode':>12s} {'ms/step':>9s} {'tok/s':>10s} {'peak/card':>10s}")
    print(f"{'eager':>12s} {eager['step_ms']:>8.1f}  {eager['tokens_per_s']:>10d} {eager['peak_gb']:>9.2f}G")
    print(f"{'compiled':>12s} {comp['step_ms']:>8.1f}  {comp['tokens_per_s']:>10d} {comp['peak_gb']:>9.2f}G")
    print("-" * 60)
    speedup = comp["tokens_per_s"] / max(eager["tokens_per_s"], 1)
    print(f"compiled is {speedup:.3f}x faster ({(speedup - 1) * 100:+.1f}% tok/s)")


if __name__ == "__main__":
    if os.environ.get("WORKER") == "1":
        worker()
    else:
        driver()
