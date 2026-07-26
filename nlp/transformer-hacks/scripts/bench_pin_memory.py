import argparse
import time
import torch


def bench(pin, batch, seq, iters, warmup, compute_dim, device):
    host = []
    for _ in range(iters + warmup):
        x = torch.randint(0, 50304, (batch, seq), dtype=torch.int64)
        y = torch.randint(0, 50304, (batch, seq), dtype=torch.int64)
        if pin:
            x, y = x.pin_memory(), y.pin_memory()
        host.append((x, y))

    w = torch.randn(compute_dim, compute_dim, device=device)

    def step(i):
        x, y = host[i]
        gx = x.to(device, non_blocking=True)
        gy = y.to(device, non_blocking=True)
        a = torch.randn(compute_dim, compute_dim, device=device)
        for _ in range(4):
            a = a @ w
        return gx, gy, a

    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()

    t0 = time.time()
    for i in range(warmup, warmup + iters):
        step(i)
    torch.cuda.synchronize()
    dt = time.time() - t0

    bytes_moved = (batch * seq * 8) * 2 * iters
    return dt / iters * 1000, bytes_moved / dt / 1e9


def pin_cost(pin, batch, seq, iters):
    t0 = time.time()
    for _ in range(iters):
        x = torch.randint(0, 50304, (batch, seq), dtype=torch.int64)
        if pin:
            x = x.pin_memory()
    return (time.time() - t0) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--compute-dim", type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"batch={args.batch} seq={args.seq} "
          f"payload={args.batch * args.seq * 8 * 2 / 1e6:.2f} MB/step "
          f"compute_dim={args.compute_dim}")
    print(f"{'mode':>10s} {'ms/step':>9s} {'H2D GB/s':>9s} {'pin_ms':>8s}")

    rows = {}
    for pin in (False, True):
        ms, gbs = bench(pin, args.batch, args.seq, args.iters,
                        args.warmup, args.compute_dim, device)
        pc = pin_cost(pin, args.batch, args.seq, args.iters)
        rows[pin] = ms
        print(f"{'pinned' if pin else 'pageable':>10s} "
              f"{ms:9.3f} {gbs:9.2f} {pc:8.3f}")

    speedup = rows[False] / rows[True]
    print(f"pinned is {speedup:.3f}x {'faster' if speedup > 1 else 'slower'} "
          f"({(speedup - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
