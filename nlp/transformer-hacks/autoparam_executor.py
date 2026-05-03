"""
Executor for autoparam - runs a single training experiment under torchrun.
Launched by autoparam.py (coordinator) via subprocess.

Usage (not called directly):
    torchrun --nproc_per_node=N autoparam_executor.py --config <path> --result <path>
"""

import json
import os
import sys
import argparse
import traceback
import time
from datetime import timedelta

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as dist

from dotenv import load_dotenv

load_dotenv()

from experiments import execute, NAMED_DATASETS
from training.model import Model, SamplingMethod
from training.trainer import DistributedStrategy
from autoparam import ConfigSerializer, StabilityMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

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
                json.dump({"score": {}, "status": "failed", "error_message": f"dist init failed: {e}"}, f)
        os._exit(1)

    dataset_name = cfg["dataset_name"]
    exp_name = cfg["exp_name"]
    timeout_minutes = cfg["timeout_minutes"]
    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy = DistributedStrategy[cfg.get("distributed_strategy", "FSDP")]

    dataset = NAMED_DATASETS[dataset_name]
    config = ConfigSerializer.dict_to_config(model_dict, dataset)
    config.sampling_method = SamplingMethod.ARGMAX
    device = torch.device(f"cuda:{rank}")
    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, timeout_minutes, strategy, device
    )

    def _dist_log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[rank{rank}][{ts}][dist] {msg}", flush=True)

    score, status, error_message = {}, "failed", None
    try:
        model = Model(config)
        params_count = sum(p.numel() for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        _, results = execute(dataset, exp_name, model, training_options)
        score = StabilityMetric.compute(results)
        score["params_count"] = params_count
        try:
            score["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            pass
        try:
            steps_run = len(results.step_loss.min_max_avg) or len(results.loss.min_max_avg)
            score["steps_run"] = steps_run
            seq_len = getattr(config, "sequence_length", 0) or getattr(config, "context_length", 0) or 0
            bs = getattr(training_options, "batch_size", 0) or 0
            accum = getattr(training_options, "accumulation_steps", 1) or 1
            world = int(os.environ.get("WORLD_SIZE", "1"))
            score["tokens_seen"] = int(steps_run * bs * accum * seq_len * world)
        except Exception:
            pass
        try:
            val_losses = [c.mean for c in results.step_val_loss.min_max_avg]
            val_accs = [c.mean for c in results.step_val_accuracy.min_max_avg]
            if val_losses:
                score["val_loss"] = float(val_losses[-1])
            if val_accs:
                score["val_accuracy"] = float(val_accs[-1])
                score["overfit_gap"] = float(score.get("final_accuracy", 0.0) - val_accs[-1])
        except Exception:
            pass
        score["knobs"] = {
            "model_config": model_dict,
            "training_config": training_dict,
            "distributed_strategy": cfg.get("distributed_strategy", "FSDP"),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }
        no_data = (
            len(results.accuracy.min_max_avg) == 0
            and len(results.step_accuracy.min_max_avg) == 0
        )
        if no_data:
            status = "failed"
            error_message = "No training data collected (dataloader may be failing)"
        else:
            status = "success"
    except torch.cuda.OutOfMemoryError as e:
        error_message = f"CUDA out of memory: {e}"
        _dist_log(f"exception caught:\n{traceback.format_exc()}")
    except Exception as e:
        error_message = str(e)
        _dist_log(f"exception caught:\n{traceback.format_exc()}")
    finally:
        _dist_log(f"finally block entered | status={status} | error={error_message}")
        if dist.is_initialized():
            if error_message is None:
                _dist_log("calling dist.barrier()")
                try:
                    dist.barrier()
                    _dist_log("dist.barrier() succeeded")
                except Exception as barrier_exc:
                    _dist_log(f"dist.barrier() FAILED: {barrier_exc}")
            torch.cuda.empty_cache()
            _dist_log("calling dist.destroy_process_group()")
            try:
                dist.destroy_process_group()
                _dist_log("dist.destroy_process_group() returned")
            except Exception as destroy_exc:
                _dist_log(f"dist.destroy_process_group() FAILED: {destroy_exc}")
        if error_message is not None:
            if rank == 0:
                with open(args.result, "w") as f:
                    json.dump(
                        {"score": score, "status": status, "error_message": error_message}, f
                    )
            os._exit(1)
    if rank == 0:
        with open(args.result, "w") as f:
            json.dump(
                {"score": score, "status": status, "error_message": error_message}, f
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
