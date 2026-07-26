import os
import gc
import time
import json
import argparse
import torch
from training.model import (
    Config,
    Model,
    TransformerLayerType,
    PositionalEmbeddingType,
)
from training.objectives import NextTokenPrediction
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling


def gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def free_total_gb():
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


LADDER = [
    ("d256_l4", 256, 4),
    ("d384_l6", 384, 6),
    ("d512_l8", 512, 8),
    ("d768_l12", 768, 12),
    ("d1024_l16", 1024, 16),
    ("d1280_l20", 1280, 20),
    ("d1536_l24", 1536, 24),
    ("d2048_l24", 2048, 24),
]


def build_config(dim, layers, vocab, seq, pad, checkpointing):
    head_dim = 64
    heads = max(1, dim // head_dim)
    return Config(
        dropout=0,
        dim_embeddings=dim,
        num_attention_heads=heads,
        num_transformer_layers=layers,
        vocab_size=vocab,
        sequence_length=seq,
        padding_index=pad,
        transformer_layer=TransformerLayerType.GPT2,
        positional_embedding=PositionalEmbeddingType.NN_EMBEDDING,
        gradient_checkpointing=checkpointing,
    )


def run_config(dim, layers, vocab, seq, pad, batch, steps, checkpointing, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    config = build_config(dim, layers, vocab, seq, pad, checkpointing)
    model = Model(config).to(device)
    model.train()
    params = sum(p.numel() for p in model.parameters())
    criterion = NextTokenPrediction(
        padding_index=pad, vocab_size=vocab, sampler=temperature_sampling
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    try:
        t0 = time.time()
        for _ in range(steps):
            X = torch.randint(0, vocab, (batch, seq), device=device)
            y = torch.randint(0, vocab, (batch, seq), device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 1e9
    finally:
        del model, optimizer, criterion
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "params": params,
        "params_m": round(params / 1e6, 1),
        "peak_gb": round(peak, 2),
        "step_ms": round(dt / steps * 1000, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("PROBE_STEPS", 128)))
    parser.add_argument("--seq", type=int, default=int(os.environ.get("SEQ_LEN", 256)))
    parser.add_argument("--vocab", type=int, default=int(os.environ.get("VOCAB_SIZE", 50304)))
    parser.add_argument("--pad", type=int, default=int(os.environ.get("PAD_INDEX", 0)))
    parser.add_argument("--out", default=os.environ.get("PROBE_OUT", "memory_probe.json"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "no cuda"
    device = torch.device("cuda:0")
    name = gpu_name()
    _, total = free_total_gb()
    print(f"GPU: {name} | {total:.1f} GB total | batch={args.batch} seq={args.seq} steps={args.steps}")
    print("-" * 90)

    results = []
    best = {False: None, True: None}
    for ckpt in (False, True):
        stop = False
        for label, dim, layers in LADDER:
            if stop:
                break
            entry = {
                "label": label, "dim": dim, "layers": layers,
                "checkpointing": ckpt, "batch": args.batch, "seq": args.seq,
            }
            try:
                r = run_config(dim, layers, args.vocab, args.seq, args.pad,
                               args.batch, args.steps, ckpt, device)
                entry.update(r)
                entry["status"] = "fit"
                best[ckpt] = entry
                print(f"[ckpt={int(ckpt)}] {label:12s} {r['params_m']:7.1f}M  "
                      f"peak={r['peak_gb']:6.2f}GB  {r['step_ms']:6.1f}ms/step  FIT")
            except torch.cuda.OutOfMemoryError:
                entry["status"] = "oom"
                print(f"[ckpt={int(ckpt)}] {label:12s} OOM -> stop this mode")
                torch.cuda.empty_cache()
                stop = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    entry["status"] = "oom"
                    print(f"[ckpt={int(ckpt)}] {label:12s} OOM -> stop this mode")
                    torch.cuda.empty_cache()
                    stop = True
                else:
                    raise
            results.append(entry)

    print("-" * 62)
    print(f"{'config':12s} {'params':>9s} {'ckpt':>5s} {'status':>7s} {'peak_gb':>9s} {'ms/step':>9s}")
    print("-" * 62)
    for e in results:
        pm = f"{e.get('params_m', 0):.1f}M" if e.get("params_m") else "-"
        peak = f"{e.get('peak_gb'):.2f}" if e.get("peak_gb") else "-"
        ms = f"{e.get('step_ms'):.1f}" if e.get("step_ms") else "-"
        print(f"{e['label']:12s} {pm:>9s} {int(e['checkpointing']):>5d} "
              f"{e['status']:>7s} {peak:>9s} {ms:>9s}")
    print("-" * 62)
    for ckpt in (False, True):
        b = best[ckpt]
        tag = "with checkpointing" if ckpt else "no checkpointing  "
        if b:
            print(f"MAX {tag}: {b['label']} ({b['params_m']}M) peak={b['peak_gb']}GB")
        else:
            print(f"MAX {tag}: none fit")

    report = {"gpu": name, "total_gb": round(total, 1), "batch": args.batch,
              "seq": args.seq, "steps": args.steps, "results": results,
              "best_no_ckpt": best[False], "best_ckpt": best[True]}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
