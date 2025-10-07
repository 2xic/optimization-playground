import torch
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling_old,
    temperature_sampling,
)
import torch.utils.benchmark as benchmark
import time
import numpy as np
import torch
from utils.web_dataloader import WebDataloader
import os
from dotenv import load_dotenv
import time

load_dotenv()


def test_sampling():
    a = torch.randn(1000, 1000)

    results = []
    for name, func in [
        ("New", temperature_sampling),
        ("Old", temperature_sampling_old),
    ]:
        # Timing
        timer = benchmark.Timer(
            stmt="func(a)",
            globals={"func": func, "a": a},
            label="temperature_sampling",
            sub_label=name,
            description="Speed",
        )
        results.append(timer.blocked_autorange(min_run_time=1))

    compare = benchmark.Compare(results)
    compare.print()


def test_memory_creation():
    batch = [[1, 2, 3, 4, 5] * 100 for _ in range(256)]
    t0 = time.time()
    for _ in range(100):
        tensors = [torch.tensor(item, dtype=torch.long) for item in batch]
        _ = torch.stack(tensors)
    t1 = time.time()

    t2 = time.time()
    for _ in range(100):
        arr = np.array(batch, dtype=np.int64)
        _ = torch.from_numpy(arr)
    t3 = time.time()

    print(f"torch.stack: {(t1 - t0) * 1000:.1f}ms")
    print(f"numpy→torch: {(t3 - t2) * 1000:.1f}ms")
    print(f"Speedup: {(t1 - t0) / (t3 - t2):.1f}x")


def dataloader():
    dataloader = WebDataloader(
        os.environ["WEB_DATALOADER"], "small-web", batch_size=1024
    )

    start = time.time()
    for index, (X, y) in enumerate(dataloader.iter()):
        if index > 100:
            break
    print(time.time() - start)


if __name__ == "__main__":
    #    test_memory_creation()
    test_sampling()
