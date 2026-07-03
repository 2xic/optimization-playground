"""
Shared scaffolding for autoparam-style executors.

Handles argparse, dist init, try/except/finally, and result writing.
Per-task executors supply a build_and_run callback that returns (score, status).
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from typing import Callable, Tuple

import torch
import torch.distributed as dist


def setup_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _parse_and_init_dist(result_path_holder: list) -> Tuple[argparse.Namespace, dict, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path_holder.append(args.result)

    with open(args.config) as f:
        cfg = json.load(f)

    rank = int(os.environ.get("RANK", "0"))
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    except Exception as e:
        if rank == 0:
            with open(args.result, "w") as f:
                json.dump(
                    {"score": {}, "status": "failed", "error_message": f"dist init failed: {e}"},
                    f,
                )
        os._exit(1)
    return args, cfg, rank


def run(
    build_and_run: Callable[[dict, int, Callable[[str], None]], Tuple[dict, str]],
    log_tag: str = "exp",
):
    """
    build_and_run(cfg, rank, log) -> (score_dict, status_str)
    On exception, status is set to "failed" and error_message captured.
    """
    result_holder = []
    args, cfg, rank = _parse_and_init_dist(result_holder)

    def _log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[rank{rank}][{ts}][{log_tag}] {msg}", flush=True)

    score, status, error_message = {}, "failed", None
    try:
        result = build_and_run(cfg, rank, _log)
        if len(result) == 3:
            score, status, error_message = result
        else:
            score, status = result
    except torch.cuda.OutOfMemoryError as e:
        error_message = f"CUDA out of memory: {e}"
        _log(f"exception caught:\n{traceback.format_exc()}")
    except Exception as e:
        error_message = str(e)
        _log(f"exception caught:\n{traceback.format_exc()}")
    finally:
        _log(f"finally | status={status} | error={error_message}")
        if error_message is not None:
            if rank == 0:
                try:
                    with open(args.result, "w") as f:
                        json.dump(
                            {"score": score, "status": status, "error_message": error_message}, f
                        )
                except Exception:
                    pass
            os._exit(1)
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as barrier_exc:
                _log(f"dist.barrier() FAILED: {barrier_exc}")
            torch.cuda.empty_cache()
            try:
                dist.destroy_process_group()
            except Exception as destroy_exc:
                _log(f"dist.destroy_process_group() FAILED: {destroy_exc}")

    if rank == 0:
        with open(args.result, "w") as f:
            json.dump(
                {"score": score, "status": status, "error_message": error_message}, f
            )
    sys.exit(0)


def tail_mean(arr, frac=0.25):
    if not arr:
        return 0.0
    n = max(1, int(len(arr) * frac))
    tail = arr[-n:]
    return float(sum(tail) / len(tail))


def slope(arr):
    if len(arr) < 2:
        return 0.0
    n = len(arr)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(arr) / n
    num = sum((xs[i] - mean_x) * (arr[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return float(num / den)


def common_score_extras(results, config, training_options, world_size, model_dict, training_dict, distributed_strategy):
    """Compute peak_memory_gb, steps_run, tokens_seen, and knobs."""
    extras = {}
    try:
        extras["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    try:
        steps_run = len(results.step_loss.min_max_avg) or len(results.loss.min_max_avg)
        extras["steps_run"] = steps_run
        seq_len = getattr(config, "sequence_length", 0) or getattr(config, "context_length", 0) or 0
        bs = getattr(training_options, "batch_size", 0) or 0
        accum = getattr(training_options, "accumulation_steps", 1) or 1
        extras["tokens_seen"] = int(steps_run * bs * accum * seq_len * world_size)
    except Exception:
        pass
    extras["knobs"] = {
        "model_config": model_dict,
        "training_config": training_dict,
        "distributed_strategy": distributed_strategy,
        "world_size": world_size,
    }
    return extras
