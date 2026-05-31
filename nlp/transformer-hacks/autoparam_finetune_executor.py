"""
Executor for autoparam_finetune - runs a single SFT experiment under torchrun.
Launched by autoparam_finetune.py via subprocess.
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
from autoparam import ConfigSerializer
from scheduler.cooperative import install_shutdown_handler
install_shutdown_handler()
from utils.web_dataloader import WebDataloader
from utils.mixture_dataloader import WebDataloaderMixture
from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path
from utils.checkpoints import apply_checkpoint_tag
from experiments import PretrainedModelConstruction

FINETUNE_DATASET_NAMES = [
    "smoltalk-256",
    "everyday-conversations-256",
    "self-oss-instruct-sc2-H4-256",
]


def _build_mixture(rank, world_size, batch_size, split):
    base_url = os.environ["WEB_DATALOADER"]
    loaders = [
        WebDataloader(
            base_url,
            name,
            split=split,
            rank=rank,
            world_size=world_size,
            batch_size=batch_size,
        )
        for name in FINETUNE_DATASET_NAMES
    ]
    return WebDataloaderMixture(loaders)


def _partial_load(target_model: Model, source_state_dict):
    target_sd = target_model.state_dict()
    loaded, skipped = [], []
    for k, v in source_state_dict.items():
        key = k.replace("module.", "")
        if key in target_sd and target_sd[key].shape == v.shape:
            target_sd[key] = v
            loaded.append(key)
        else:
            skipped.append(key)
    target_model.load_state_dict(target_sd)
    return loaded, skipped


def _tail_mean(arr, frac=0.25):
    if not arr:
        return 0.0
    n = max(1, int(len(arr) * frac))
    tail = arr[-n:]
    return float(sum(tail) / len(tail))


def _slope(arr):
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

    exp_name = cfg["exp_name"]
    timeout_minutes = cfg["timeout_minutes"]
    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy = DistributedStrategy[cfg.get("distributed_strategy", "FSDP")]
    init_from_tag = cfg.get("init_from_tag")
    init_from_path = cfg.get("init_from_path")

    def _dist_log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[rank{rank}][{ts}][ft] {msg}", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}")

    score, status, error_message = {}, "failed", None
    partial_load_warning = None
    try:
        if init_from_tag:
            checkpoint_path = load_modeL_tag(init_from_tag)
        elif init_from_path:
            checkpoint_path = init_from_path
        else:
            raise ValueError("config requires init_from_tag or init_from_path")
        _dist_log(f"loading pretrained from: {checkpoint_path}")
        base_model, pretrain_config = load_model_from_path(checkpoint_path)

        proposed_config = ConfigSerializer.dict_to_config(model_dict, NAMED_DATASETS[FINETUNE_DATASET_NAMES[0]])
        proposed_config.sampling_method = SamplingMethod.ARGMAX

        arch_keys = ("num_transformer_layers", "dim_embeddings", "num_attention_heads",
                     "vocab_size", "sequence_length", "transformer_layer", "positional_embedding")
        arch_match = all(
            getattr(proposed_config, k, None) == getattr(pretrain_config, k, None)
            for k in arch_keys
        )

        training_options = ConfigSerializer.dict_to_training_options(
            training_dict, timeout_minutes, strategy, device
        )
        apply_checkpoint_tag(training_options, cfg)
        bs = getattr(training_options, "batch_size", 32) or 32
        train_mixture = _build_mixture(rank, world_size, bs, split="train")
        try:
            val_mixture = _build_mixture(rank, world_size, bs, split="test")
            for vl in val_mixture.dataloaders:
                _ = vl.info
            training_options.val_loader = val_mixture
        except Exception as e:
            _dist_log(f"val mixture unavailable: {e}")
            training_options.val_loader = None

        if arch_match:
            _dist_log("arch matches pretrain - using PretrainedModelConstruction")
            model = PretrainedModelConstruction(pretrain_config, base_model)()
        else:
            _dist_log("arch MISMATCH - partial loading matching tensors")
            model = Model(proposed_config)
            loaded, skipped = _partial_load(model, base_model.state_dict())
            partial_load_warning = (
                f"arch mismatch: loaded {len(loaded)} tensors, "
                f"skipped {len(skipped)} (re-initialized). "
                f"Pretrain: layers={getattr(pretrain_config,'num_transformer_layers',None)}, "
                f"dim={getattr(pretrain_config,'dim_embeddings',None)}, "
                f"heads={getattr(pretrain_config,'num_attention_heads',None)}. "
                f"Proposed: layers={getattr(proposed_config,'num_transformer_layers',None)}, "
                f"dim={getattr(proposed_config,'dim_embeddings',None)}, "
                f"heads={getattr(proposed_config,'num_attention_heads',None)}."
            )
            _dist_log(partial_load_warning)

        del base_model

        params_count = sum(p.numel() for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        _, results = execute(train_mixture, exp_name, model, training_options)

        val_accs = [c.mean for c in results.step_val_accuracy.min_max_avg]
        val_losses = [c.mean for c in results.step_val_loss.min_max_avg]

        final_acc = _tail_mean(val_accs) if val_accs else 0.0
        acc_slope = _slope(val_accs) if val_accs else 0.0
        final_loss = _tail_mean(val_losses) if val_losses else 0.0
        import math
        try:
            perplexity = float(math.exp(final_loss)) if final_loss > 0 else 0.0
        except OverflowError:
            perplexity = float("inf")

        score = {
            "final_accuracy": float(final_acc),
            "accuracy_slope": float(acc_slope),
            "final_loss": float(final_loss),
            "perplexity": float(perplexity),
            "steps_to_threshold": 0.0,
            "params_count": params_count,
        }
        if val_accs:
            score["val_accuracy"] = float(val_accs[-1])
        if val_losses:
            score["val_loss"] = float(val_losses[-1])
        try:
            score["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        except Exception:
            pass
        try:
            steps_run = len(results.step_loss.min_max_avg) or len(results.loss.min_max_avg)
            score["steps_run"] = steps_run
            seq_len = getattr(proposed_config, "sequence_length", 0) or 0
            accum = getattr(training_options, "accumulation_steps", 1) or 1
            score["tokens_seen"] = int(steps_run * bs * accum * seq_len * world_size)
        except Exception:
            pass
        score["knobs"] = {
            "model_config": model_dict,
            "training_config": training_dict,
            "distributed_strategy": cfg.get("distributed_strategy", "FSDP"),
            "world_size": world_size,
            "init_from_tag": init_from_tag,
            "init_from_path": init_from_path,
        }
        if partial_load_warning:
            score["partial_load_warning"] = partial_load_warning

        no_data = len(results.step_val_accuracy.min_max_avg) == 0 and len(results.step_accuracy.min_max_avg) == 0
        if no_data:
            status = "failed"
            error_message = "No training data collected"
        else:
            status = "success"
    except torch.cuda.OutOfMemoryError as e:
        error_message = f"CUDA out of memory: {e}"
        _dist_log(f"exception caught:\n{traceback.format_exc()}")
    except Exception as e:
        error_message = str(e)
        _dist_log(f"exception caught:\n{traceback.format_exc()}")
    finally:
        _dist_log(f"finally | status={status} | error={error_message}")
        if dist.is_initialized():
            if error_message is None:
                try:
                    dist.barrier()
                except Exception as barrier_exc:
                    _dist_log(f"dist.barrier() FAILED: {barrier_exc}")
            torch.cuda.empty_cache()
            try:
                dist.destroy_process_group()
            except Exception as destroy_exc:
                _dist_log(f"dist.destroy_process_group() FAILED: {destroy_exc}")
        if error_message is not None:
            if rank == 0:
                with open(args.result, "w") as f:
                    json.dump({"score": score, "status": status, "error_message": error_message}, f)
            os._exit(1)

    if rank == 0:
        with open(args.result, "w") as f:
            json.dump({"score": score, "status": status, "error_message": error_message}, f)
    sys.exit(0)


if __name__ == "__main__":
    main()
