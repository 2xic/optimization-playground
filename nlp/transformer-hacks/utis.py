import pynvml
from training.objectives import NextTokenPrediction
import torch
import time

from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling,
)


_last_gpu_log_time = 0

def get_best_gpu(model_size_gb, margins_gb=4):
    global _last_gpu_log_time
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        best_gpu = None
        max_free_memory = 0
        minimum_free_memory = model_size_gb + margins_gb

        now = time.time()
        should_log = (now - _last_gpu_log_time) >= 300

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / (1024**3)

            if should_log:
                print(f"GPU {i}: {free_gb:.2f} GB free")

            if free_gb >= minimum_free_memory and info.free > max_free_memory:
                max_free_memory = info.free
                best_gpu = i

        if should_log:
            _last_gpu_log_time = now

        pynvml.nvmlShutdown()
        return best_gpu

    except Exception as e:
        print(f"Error checking GPU memory: {e}")
        return None


# Estimate the model size
def estimate_model_size_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Assume float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_gb = size_bytes / (1024**3)
    return size_gb


def estimate_cuda_size(model):
    model_size = estimate_model_size_gb(model)
    training_multiplier = 3
    activation_multiplier = 1.5

    total_size = model_size * training_multiplier * activation_multiplier
    return total_size


def benchmark_training(
    model,
    dataset,
    optimizer,
    batch_size,
    gradient_accumulation_steps=1,
    num_iterations=100,
    warmup_iterations=10,
):
    criterion = NextTokenPrediction(
        padding_index=dataset.padding_index,
        vocab_size=dataset.vocab_size,
        sampler=temperature_sampling,
    )
    dataset.set_batch_size(batch_size)
    dataloader = iter(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = optimizer.create_optimizer(
        model.parameters()
    )  # torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    print("Warming up...")
    for idx in range(warmup_iterations):
        X, y = next(dataloader)
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        if idx % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Synchronize before timing (important for GPU)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Actual benchmark
    print(f"Running benchmark for {num_iterations} iterations...")
    start_time = time.time()

    for _ in range(num_iterations):
        X, y = next(dataloader)
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        if idx % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Synchronize after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    time_per_iteration = total_time / num_iterations
    samples_per_second = (batch_size * num_iterations) / total_time

    print(f"\n{'=' * 50}")
    print("Benchmark Results:")
    print(f"{'=' * 50}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Time per iteration: {time_per_iteration * 1000:.2f} ms")
    print(f"Throughput: {samples_per_second:.2f} samples/second")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Gradient Accumulation steps: {gradient_accumulation_steps}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    gpu = get_best_gpu()
    if gpu is not None:
        print(f"Best GPU: {gpu}")
    else:
        print("No suitable GPU found.")
