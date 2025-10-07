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
from training.model import Model, TransformerLayerType, MaskOrder
from experiments import create_default_config
from torch.cuda.amp import autocast, GradScaler

load_dotenv()

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import os

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12355"
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"


def train():
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{dist.get_rank()}")

    dataset = WebDataloader(
        os.environ["WEB_DATALOADER"], "medium-512-web", batch_size=128
    )
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
    for i in range(100):
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
    device = torch.device("cuda:0")
    dataset = WebDataloader(os.environ["WEB_DATALOADER"], "small-web", batch_size=256)
    config = create_default_config(dataset)
    model = Model(config).to(device)

    evaluator = NextTokenPrediction(
        padding_index=dataset.padding_index,
        vocab_size=dataset.vocab_size,
        sampler=temperature_sampling,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    # scaler = GradScaler()
    scaler = GradScaler()

    dataloader = dataset.iter(batch_size=256, workers=8)

    # Warmup
    for i, (X, y) in enumerate(dataloader):
        if i >= 5:
            break
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            y_pred = model(X)
            loss = evaluator.forward(y_pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Detailed timing
    times = {"to_device": [], "forward": [], "backward": [], "optimizer": []}

    for i, (X, y) in enumerate(dataloader):
        if i >= 50:
            break

        t0 = time.time()
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            y_pred = model(X)
            loss = evaluator.forward(y_pred, y)
        torch.cuda.synchronize()
        t2 = time.time()

        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        t3 = time.time()

        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t4 = time.time()

        times["to_device"].append(t1 - t0)
        times["forward"].append(t2 - t1)
        times["backward"].append(t3 - t2)
        times["optimizer"].append(t4 - t3)

    print("\nPer-batch timing breakdown:")
    print(f"  To device:  {sum(times['to_device']) / 50 * 1000:.1f}ms")
    print(f"  Forward:    {sum(times['forward']) / 50 * 1000:.1f}ms")
    print(f"  Backward:   {sum(times['backward']) / 50 * 1000:.1f}ms")
    print(f"  Optimizer:  {sum(times['optimizer']) / 50 * 1000:.1f}ms")
    print(f"  TOTAL:      {sum(sum(v) for v in times.values()) / 50 * 1000:.1f}ms")


if __name__ == "__main__":
#    profile()
#    profile()
