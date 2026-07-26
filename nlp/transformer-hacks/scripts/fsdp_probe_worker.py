import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.amp import autocast
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from training.model import (
    Config,
    Model,
    TransformerLayerType,
    PositionalEmbeddingType,
)
from training.objectives import NextTokenPrediction
from training.trainer import best_autocast_dtype
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    device = torch.device(f"cuda:{local}")

    dim = int(os.environ["DIM"])
    layers = int(os.environ["LAYERS"])
    vocab = int(os.environ["VOCAB"])
    seq = int(os.environ["SEQ"])
    pad = int(os.environ["PAD"])
    batch = int(os.environ["BATCH"])
    steps = int(os.environ["STEPS"])
    ckpt = os.environ["CKPT"] == "1"
    result = os.environ["RESULT_FILE"]

    heads = max(1, dim // 64)
    config = Config(
        dropout=0,
        dim_embeddings=dim,
        num_attention_heads=heads,
        num_transformer_layers=layers,
        vocab_size=vocab,
        sequence_length=seq,
        padding_index=pad,
        transformer_layer=TransformerLayerType.GPT2,
        positional_embedding=PositionalEmbeddingType.NN_EMBEDDING,
        gradient_checkpointing=ckpt,
    )

    torch.cuda.reset_peak_memory_stats(device)
    model = Model(config)
    params = sum(p.numel() for p in model.parameters())

    mp_dtype = best_autocast_dtype()
    bf16 = mp_dtype == torch.bfloat16
    mp_policy = MixedPrecisionPolicy(
        param_dtype=mp_dtype if bf16 else None, reduce_dtype=mp_dtype
    )
    blocks = [
        layer
        for mod in model.modules()
        if isinstance(mod, torch.nn.ModuleList)
        for layer in mod
    ]
    for layer in blocks:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    model.to(device)
    model.train()

    criterion = NextTokenPrediction(
        padding_index=pad, vocab_size=vocab, sampler=temperature_sampling
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    t0 = time.time()
    for _ in range(steps):
        X = torch.randint(0, vocab, (batch, seq), device=device)
        y = torch.randint(0, vocab, (batch, seq), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=mp_dtype):
            output = model(X)
            loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dt = time.time() - t0

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
                    "params": params,
                    "params_m": round(params / 1e6, 1),
                    "peak_gb": round(float(peak.item()), 2),
                    "step_ms": round(step_s * 1000, 1),
                    "tokens_per_s": round(tokens_per_s),
                    "tflops": round(tflops, 2),
                    "dtype": str(mp_dtype).replace("torch.", ""),
                },
                f,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
