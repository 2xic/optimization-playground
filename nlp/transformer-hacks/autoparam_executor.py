"""
Executor for autoparam - runs a single training experiment under torchrun.
Launched by autoparam.py (coordinator) via subprocess.

Usage (not called directly):
    torchrun --nproc_per_node=N autoparam_executor.py --config <path> --result <path>
"""

import json
import os
import argparse
import traceback
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
from training.model import Model
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
        dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
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
    device = torch.device(f"cuda:{rank}")
    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, timeout_minutes, strategy, device
    )

    score, status, error_message = {}, "failed", None
    try:
        _, results = execute(dataset, exp_name, Model(config), training_options)
        score = StabilityMetric.compute(results)
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
        traceback.print_exc()
    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    if rank == 0:
        with open(args.result, "w") as f:
            json.dump(
                {"score": score, "status": status, "error_message": error_message}, f
            )

    os._exit(0)


if __name__ == "__main__":
    main()
