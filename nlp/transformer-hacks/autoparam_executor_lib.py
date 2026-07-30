"""
Shared scaffolding for autoparam-style executors.

Handles argparse, dist init, try/except/finally, and result writing.
Per-task executors supply a build_and_run callback that returns (score, status).
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import timedelta
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
import contextlib


def setup_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def bootstrap():
    setup_env()
    from dotenv import load_dotenv
    load_dotenv()
    from scheduler.cooperative import install_shutdown_handler
    install_shutdown_handler()


def _parse_and_init_dist(result_path_holder: list) -> tuple[argparse.Namespace, dict, int]:
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


BuildResult = Union[tuple[dict, str], tuple[dict, str, Optional[str]]]


def run(
    build_and_run: Callable[[dict, int, Callable[[str], None]], BuildResult],
    log_tag: str = "exp",
):
    """
    build_and_run(cfg, rank, log) -> (score, status) or (score, status, error_message)
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
        _log(f"finally | status={status} | batches={score.get('batches_seen')} rows={score.get('rows_seen')} | error={error_message}")
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


def val_series(results):
    val_accs = [c.mean for c in results.step_val_accuracy.min_max_avg]
    val_losses = [c.mean for c in results.step_val_loss.min_max_avg]
    return val_accs, val_losses


VAL_WINDOW_FRAC = 0.5


def apply_val_metrics(score, val_accs, val_losses):
    if val_accs:
        score["val_accuracy"] = tail_mean(val_accs, VAL_WINDOW_FRAC)
        score["val_accuracy_last"] = float(val_accs[-1])
    if val_losses:
        score["val_loss"] = tail_mean(val_losses, VAL_WINDOW_FRAC)
        score["val_loss_last"] = float(val_losses[-1])
    return score


OBJECTIVE_LOSS_WEIGHT = 1.0


def extract_objective(score: dict) -> float:
    acc = score.get("val_accuracy", score.get("final_accuracy"))
    val_loss = score.get("val_loss", score.get("final_loss"))
    has_acc = acc is not None and not math.isnan(acc)
    has_loss = val_loss is not None and not math.isnan(val_loss)
    if has_acc:
        return float(acc) - (OBJECTIVE_LOSS_WEIGHT * float(val_loss) if has_loss else 0.0)
    if has_loss:
        return -float(val_loss)
    return -1.0


def base_val_score(results, params_count) -> dict[str, Any]:
    val_accs, val_losses = val_series(results)
    final_acc = tail_mean(val_accs, VAL_WINDOW_FRAC) if val_accs else 0.0
    acc_slope = slope(val_accs) if val_accs else 0.0
    final_loss = tail_mean(val_losses, VAL_WINDOW_FRAC) if val_losses else 0.0
    try:
        perplexity = float(math.exp(final_loss)) if final_loss > 0 else 0.0
    except OverflowError:
        perplexity = float("inf")
    score: dict[str, Any] = {
        "final_accuracy": float(final_acc),
        "accuracy_slope": float(acc_slope),
        "final_loss": float(final_loss),
        "perplexity": float(perplexity),
        "steps_to_threshold": 0.0,
        "params_count": params_count,
    }
    apply_val_metrics(score, val_accs, val_losses)
    return score


def prepare_common(cfg, rank, dataset):
    from training.model import SamplingMethod
    from training.trainer import DistributedStrategy
    from autoparam import ConfigSerializer
    from utils.checkpoints import apply_checkpoint_tag

    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy_name = cfg.get("distributed_strategy", "FSDP")
    strategy = DistributedStrategy[strategy_name]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}")
    config = ConfigSerializer.dict_to_config(model_dict, dataset)
    config.sampling_method = SamplingMethod.ARGMAX
    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, cfg["timeout_minutes"], strategy, device
    )
    apply_checkpoint_tag(training_options, cfg)
    return config, training_options, world_size, device, strategy_name


def _load_matching(target, source_sd):
    tsd = target.state_dict()
    loaded, skipped = [], []
    for k, v in source_sd.items():
        key = k.replace("module.", "")
        if key in tsd and tsd[key].shape == v.shape:
            tsd[key] = v
            loaded.append(key)
        else:
            skipped.append(key)
    target.load_state_dict(tsd, strict=False)
    return loaded, skipped


def warm_start(cfg, proposed_config, log=None):
    """Build inner base model, optionally partial-loading pretrain weights (init_from).
    Returns (inner, warning). Fresh optimizer — this is warm-start, not resume."""
    from training.model import Model
    from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path
    inner = Model(proposed_config)
    tag, path = cfg.get("init_from_tag"), cfg.get("init_from_path")
    if not (tag or path):
        return inner, None
    ckpt = load_modeL_tag(tag) if tag else path
    if log:
        log(f"warm-starting from: {ckpt}")
    base, _ = load_model_from_path(ckpt)
    loaded, skipped = _load_matching(inner, base.state_dict())
    warning = None
    if skipped:
        warning = f"arch mismatch: loaded {len(loaded)} tensors, skipped {len(skipped)}."
        if log:
            log(warning)
    return inner, warning


def apply_resume(cfg, model, training_options, log=None):
    """Continue this task's own run: load full wrapped model + optimizer + step.
    Returns resume dict for execute(), or None if no resume checkpoint."""
    path = cfg.get("resume_from_path")
    if not path:
        return None
    from utils.load_mode_from_checkpoint import load_raw_from_path
    model_state, optimizer_state, stats = load_raw_from_path(path)
    model.load_state_dict(model_state)
    step = int(stats.get("steps", 0))
    sched = getattr(training_options, "lr_scheduler", None)
    if sched is not None and hasattr(sched, "last_epoch"):
        sched.last_epoch = step
    if log:
        log(f"resuming model + optimizer + step from {step} ({path})")
    return {"optimizer_state": optimizer_state, "step": step}


def no_data_result(score, results, message="No training data collected"):
    no_data = (
        len(results.step_val_accuracy.min_max_avg) == 0
        and len(results.step_accuracy.min_max_avg) == 0
    )
    if no_data:
        return score, "failed", message
    return score, "success", None


def common_score_extras(results, config, training_options, world_size, model_dict, training_dict, distributed_strategy):
    """Compute peak_memory_gb, steps_run, tokens_seen, and knobs."""
    extras = {}
    with contextlib.suppress(Exception):
        extras["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
    try:
        record_interval = getattr(training_options, "record_interval_steps", 1) or 1
        records = len(results.step_loss.min_max_avg) or len(results.loss.min_max_avg)
        steps_run = records * record_interval
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
