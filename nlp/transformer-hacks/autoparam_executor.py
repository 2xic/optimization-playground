"""
Executor for autoparam - runs a single training experiment under torchrun.
Launched by autoparam.py (coordinator) via subprocess.

Usage (not called directly):
    torchrun --nproc_per_node=N autoparam_executor.py --config <path> --result <path>
"""

import os

from autoparam_executor_lib import setup_env, run, common_score_extras

setup_env()

import torch
from dotenv import load_dotenv

load_dotenv()

from experiments import execute, NAMED_DATASETS, PretrainedModelConstruction
from training.model import Model, SamplingMethod
from training.trainer import DistributedStrategy
from autoparam import ConfigSerializer, StabilityMetric
from scheduler.cooperative import install_shutdown_handler
install_shutdown_handler()
from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path, load_raw_from_path
from utils.checkpoints import apply_checkpoint_tag


def build_and_run(cfg, rank, log):
    dataset_name = cfg["dataset_name"]
    exp_name = cfg["exp_name"]
    timeout_minutes = cfg["timeout_minutes"]
    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy_name = cfg.get("distributed_strategy", "FSDP")
    strategy = DistributedStrategy[strategy_name]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dataset = NAMED_DATASETS[dataset_name]
    config = ConfigSerializer.dict_to_config(model_dict, dataset)
    config.sampling_method = SamplingMethod.ARGMAX
    device = torch.device(f"cuda:{rank}")
    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, timeout_minutes, strategy, device
    )
    apply_checkpoint_tag(training_options, cfg)

    init_from_tag = cfg.get("init_from_tag")
    init_from_path = cfg.get("init_from_path")
    resume = None
    if init_from_tag or init_from_path:
        checkpoint_path = load_modeL_tag(init_from_tag) if init_from_tag else init_from_path
        log(f"resuming from: {checkpoint_path}")
        base_model, pretrain_config = load_model_from_path(checkpoint_path)
        arch_keys = ("num_transformer_layers", "dim_embeddings", "num_attention_heads",
                     "vocab_size", "sequence_length", "transformer_layer", "positional_embedding")
        arch_match = all(
            getattr(config, k, None) == getattr(pretrain_config, k, None)
            for k in arch_keys
        )
        if not arch_match:
            raise ValueError(
                f"arch mismatch: preset config does not match checkpoint {checkpoint_path}"
            )
        log("arch matches checkpoint - using PretrainedModelConstruction")
        model = PretrainedModelConstruction(pretrain_config, base_model)()
        if cfg.get("resume_optimizer", True):
            _, optimizer_state, stats = load_raw_from_path(checkpoint_path)
            resume_step = int(stats.get("steps", 0))
            resume = {"optimizer_state": optimizer_state, "step": resume_step}
            sched = getattr(training_options, "lr_scheduler", None)
            if sched is not None and hasattr(sched, "last_epoch"):
                sched.last_epoch = resume_step
            log(f"resuming optimizer + step from {resume_step}")
    else:
        model = Model(config)
    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(dataset, exp_name, model, training_options, resume=resume)
    score = StabilityMetric.compute(results)
    score["params_count"] = params_count
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
    score.update(common_score_extras(
        results, config, training_options, world_size, model_dict, training_dict, strategy_name
    ))
    no_data = (
        len(results.accuracy.min_max_avg) == 0
        and len(results.step_accuracy.min_max_avg) == 0
    )
    if no_data:
        return score, "failed", "No training data collected (dataloader may be failing)"
    return score, "success", None


if __name__ == "__main__":
    run(build_and_run, log_tag="dist")
