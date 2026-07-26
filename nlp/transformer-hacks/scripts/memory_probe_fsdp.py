import os
import sys
import json
import argparse
import subprocess
import tempfile
import torch


LADDER = [
    ("d256_l4", 256, 4),
    ("d384_l6", 384, 6),
    ("d512_l8", 512, 8),
    ("d768_l12", 768, 12),
    ("d1024_l16", 1024, 16),
    ("d1280_l20", 1280, 20),
    ("d1536_l24", 1536, 24),
    ("d2048_l24", 2048, 24),
    ("d2560_l32", 2560, 32),
    ("d3072_l36", 3072, 36),
]

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fsdp_probe_worker.py")


def run_config(dim, layers, args, ckpt, timeout, batch):
    fd, result = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(result)
    env = dict(os.environ)
    env.update(
        DIM=str(dim), LAYERS=str(layers), VOCAB=str(args.vocab), SEQ=str(args.seq),
        PAD=str(args.pad), BATCH=str(batch), STEPS=str(args.steps),
        CKPT="1" if ckpt else "0", RESULT_FILE=result,
    )
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={args.nproc}", WORKER,
    ]
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if proc.returncode != 0 or not os.path.exists(result):
        blob = (proc.stdout + proc.stderr).lower()
        status = "oom" if "out of memory" in blob or "outofmemory" in blob else "fail"
        if status == "fail":
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
            print(tail)
        return None, status
    with open(result) as f:
        r = json.load(f)
    os.remove(result)
    return r, "fit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("PROBE_STEPS", 128)))
    parser.add_argument("--seq", type=int, default=int(os.environ.get("SEQ_LEN", 256)))
    parser.add_argument("--vocab", type=int, default=int(os.environ.get("VOCAB_SIZE", 50304)))
    parser.add_argument("--pad", type=int, default=int(os.environ.get("PAD_INDEX", 0)))
    parser.add_argument("--nproc", type=int, default=int(os.environ.get("NUM_PROCESS", 2)))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("PROBE_TIMEOUT", 600)))
    parser.add_argument("--out", default=os.environ.get("PROBE_OUT", "memory_probe_fsdp.json"))
    args = parser.parse_args()

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    total = torch.cuda.mem_get_info()[1] / 1e9 if torch.cuda.is_available() else 0
    print(f"GPU: {name} x{args.nproc} FSDP | {total:.1f} GB/card | "
          f"batch={args.batch} seq={args.seq} steps={args.steps}")
    header = (f"{'ckpt':>4s}  {'config':12s} {'params':>9s} {'peak/card':>11s} "
              f"{'ms/step':>9s} {'tok/s':>10s} {'TFLOP/s':>8s} {'dtype':>8s}  status")
    print("-" * 90)
    print(header)
    print("-" * 90)

    results = []
    best = {False: None, True: None}
    for ckpt in (False, True):
        for label, dim, layers in LADDER:
            entry = {"label": label, "dim": dim, "layers": layers,
                     "checkpointing": ckpt, "batch": args.batch, "seq": args.seq}
            r, status = run_config(dim, layers, args, ckpt, args.timeout, args.batch)
            entry["status"] = status
            if status == "fit":
                entry.update(r)
                best[ckpt] = entry
                print(f"{int(ckpt):>4d}  {label:12s} {r['params_m']:8.1f}M "
                      f"{r['peak_gb']:8.2f}GB/c {r['step_ms']:7.1f}ms "
                      f"{r['tokens_per_s']:>10d} {r['tflops']:>8.2f} "
                      f"{r['dtype']:>8s}  FIT")
            else:
                print(f"{int(ckpt):>4d}  {label:12s} {'':>9s} {'':>11s} "
                      f"{'':>9s} {'':>10s} {'':>8s} {'':>8s}  {status.upper()} -> stop")
                results.append(entry)
                break
            results.append(entry)

    print("-" * 84)
    print(f"{'config':12s} {'params':>10s} {'ckpt':>5s} {'status':>8s} "
          f"{'peak/card':>10s} {'ms/step':>9s} {'tok/s':>10s} {'TFLOP/s':>8s}")
    print("-" * 84)
    for e in results:
        pm = f"{e.get('params_m', 0):.1f}M" if e.get("params_m") else "-"
        peak = f"{e.get('peak_gb'):.2f}" if e.get("peak_gb") else "-"
        ms = f"{e.get('step_ms'):.1f}" if e.get("step_ms") else "-"
        tps = f"{e.get('tokens_per_s'):d}" if e.get("tokens_per_s") else "-"
        tf = f"{e.get('tflops'):.2f}" if e.get("tflops") else "-"
        print(f"{e['label']:12s} {pm:>10s} {int(e['checkpointing']):>5d} "
              f"{e['status']:>8s} {peak:>10s} {ms:>9s} {tps:>10s} {tf:>8s}")
    print("-" * 84)
    for ckpt in (False, True):
        b = best[ckpt]
        tag = "with checkpointing" if ckpt else "no checkpointing  "
        if b:
            print(f"MAX {tag}: {b['label']} ({b['params_m']}M) "
                  f"peak={b['peak_gb']}GB/card")
        else:
            print(f"MAX {tag}: none fit")

    report = {"gpu": name, "nproc": args.nproc, "total_gb": round(total, 1),
              "strategy": "fsdp", "batch": args.batch, "seq": args.seq,
              "steps": args.steps, "results": results,
              "best_no_ckpt": best[False], "best_ckpt": best[True]}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
