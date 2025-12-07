from utils.web_dataloader import WebDataloader
import os
import torch
import time
import torch.optim as optim
from dotenv import load_dotenv
from training.objectives import NextTokenPrediction
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
)
from training.model import Model
from experiments import create_default_config
from torch.cuda.amp import autocast, GradScaler

load_dotenv()

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import statistics
import os

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12355"
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"


def train():
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{dist.get_rank()}")

    dataset = WebDataloader(os.environ["WEB_DATALOADER"], "medium-web", batch_size=128)
    # device = torch.device("cuda:1")
    config = create_default_config(
        dataset,
    )
    # config.num_transformer_layers = 12
    # config.num_attention_heads = 12
    # config.dim_embeddings = 768
    # config.dropout = 0.0
    # config.transformer_layer = TransformerLayerType.GPT2
    # config.masked_order = MaskOrder.TRIL

    model = Model(config).to(device)
    model = FSDP(model)

    evaluator = NextTokenPrediction(
        padding_index=dataset.padding_index,
        vocab_size=dataset.vocab_size,
        sampler=temperature_sampling,
    )
    optimizer = optim.Adam(model.parameters())
    dataloader = dataset.iter()

    loss_total = 0
    accuracy_total = 0
    accuracy_rows = 0
    for _ in range(100):
        for index, (X, y) in enumerate(dataloader):
            print(f"index {index}")
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = evaluator.forward(y_pred, y)
            loss.backward()
            optimizer.step()

            acc, _rows = evaluator.evaluator(y_pred, y)
            accuracy_total += acc
            accuracy_rows += _rows
            loss_total += loss
            if index % 100 == 0 and index > 1:
                print((loss_total / index), (accuracy_total / accuracy_rows) * 100)


def profile():
    effective_batch_size = 2048 * 1
    micro_batch_size = 2048
    accumulation_steps = effective_batch_size // micro_batch_size

    device = torch.device("cuda:0")
    dataset = WebDataloader(
        os.environ["WEB_DATALOADER"], "medium-web", batch_size=micro_batch_size
    )
    config = create_default_config(dataset)
    model = Model(config).to(device)
    model = torch.compile(model)

    evaluator = NextTokenPrediction(
        padding_index=dataset.padding_index,
        vocab_size=dataset.vocab_size,
        sampler=temperature_sampling,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    scaler = GradScaler()
    dataloader = dataset.iter(batch_size=micro_batch_size, workers=8)

    # Warmup
    optimizer.zero_grad(set_to_none=True)
    for i, (X, y) in enumerate(dataloader):
        if i >= 5:
            break
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            loss = evaluator.forward(model(X), y) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # Profile 50 effective batches (50 * accumulation_steps micro batches)
    times = {"data": [], "to_device": [], "forward": [], "backward": [], "optim": []}
    queue_sizes = []
    dataloader_iter = iter(dataloader)

    for i in range(50):
        t_data, t_device, t_fwd, t_bwd = 0, 0, 0, 0

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(accumulation_steps):
            t0 = time.perf_counter()
            X, y = next(dataloader_iter)
            t1 = time.perf_counter()

            if hasattr(dataloader.iter, "batch_queue"):
                queue_sizes.append(dataloader.iter.batch_queue.qsize())

            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            with autocast():
                loss = evaluator.forward(model(X), y) / accumulation_steps
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            t4 = time.perf_counter()

            t_data += t1 - t0
            t_device += t2 - t1
            t_fwd += t3 - t2
            t_bwd += t4 - t3

        t5 = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t6 = time.perf_counter()

        times["data"].append(t_data)
        times["to_device"].append(t_device)
        times["forward"].append(t_fwd)
        times["backward"].append(t_bwd)
        times["optim"].append(t6 - t5)

    # Results
    print(
        f"\nEffective batch size: {effective_batch_size} ({accumulation_steps}x{micro_batch_size})"
    )
    print("\nTiming (ms)        Mean      Std      %")
    print("-" * 45)
    total_mean = sum(statistics.mean(v) for v in times.values())
    for stage, vals in times.items():
        mean = statistics.mean(vals) * 1000
        std = statistics.stdev(vals) * 1000
        pct = (statistics.mean(vals) / total_mean) * 100
        print(f"{stage:<12} {mean:>10.2f} {std:>8.2f} {pct:>6.1f}%")

    print(f"\nTotal: {total_mean * 1000:.1f}ms/batch, {1 / total_mean:.1f} batch/s")
    print(f"Tokens/s: {effective_batch_size * 32 / total_mean:.0f}")

    if queue_sizes:
        print(
            f"Queue: mean={statistics.mean(queue_sizes):.1f}, empty={queue_sizes.count(0)}/{len(queue_sizes)}"
        )


if __name__ == "__main__":
    #    profile()
    profile()
