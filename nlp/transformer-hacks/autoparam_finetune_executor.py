"""
Executor for autoparam_finetune - runs a single SFT experiment under torchrun.
Launched by autoparam_finetune.py via subprocess.
"""

import math
import os

from autoparam_executor_lib import setup_env, run, tail_mean, slope, common_score_extras

setup_env()

import torch
from dotenv import load_dotenv

load_dotenv()

from experiments import execute, NAMED_DATASETS, PretrainedModelConstruction
from training.model import Model, SamplingMethod
from training.trainer import DistributedStrategy
from autoparam import ConfigSerializer
from scheduler.cooperative import install_shutdown_handler
install_shutdown_handler()
from utils.web_dataloader import WebDataloader
from utils.mixture_dataloader import WebDataloaderMixture
from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path
from utils.checkpoints import apply_checkpoint_tag

WARMUP_CLAMP_ENABLED = False

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


def build_and_run(cfg, rank, log):
    exp_name = cfg["exp_name"]
    timeout_minutes = cfg["timeout_minutes"]
    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy_name = cfg.get("distributed_strategy", "FSDP")
    strategy = DistributedStrategy[strategy_name]
    init_from_tag = cfg.get("init_from_tag")
    init_from_path = cfg.get("init_from_path")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}")

    if init_from_tag:
        checkpoint_path = load_modeL_tag(init_from_tag)
    elif init_from_path:
        checkpoint_path = init_from_path
    else:
        raise ValueError("config requires init_from_tag or init_from_path")
    log(f"loading pretrained from: {checkpoint_path}")
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
    sched = getattr(training_options, "lr_scheduler", None)
    if WARMUP_CLAMP_ENABLED and sched is not None and getattr(sched, "warmup_steps", 0):
        max_warmup = 200
        if sched.warmup_steps > max_warmup:
            log(f"clamping warmup_steps {sched.warmup_steps} -> {max_warmup} (SFT runs are short)")
            sched.warmup_steps = max_warmup
    bs = getattr(training_options, "batch_size", 32) or 32
    train_mixture = _build_mixture(rank, world_size, bs, split="train")
    try:
        val_mixture = _build_mixture(rank, world_size, bs, split="test")
        for vl in val_mixture.dataloaders:
            _ = vl.info
        training_options.val_loader = val_mixture
    except Exception as e:
        log(f"val mixture unavailable: {e}")
        training_options.val_loader = None

    partial_load_warning = None
    if arch_match:
        log("arch matches pretrain - using PretrainedModelConstruction")
        model = PretrainedModelConstruction(pretrain_config, base_model)()
    else:
        log("arch MISMATCH - partial loading matching tensors")
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
        log(partial_load_warning)

    del base_model

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(train_mixture, exp_name, model, training_options)

    val_accs = [c.mean for c in results.step_val_accuracy.min_max_avg]
    val_losses = [c.mean for c in results.step_val_loss.min_max_avg]

    final_acc = tail_mean(val_accs) if val_accs else 0.0
    acc_slope = slope(val_accs) if val_accs else 0.0
    final_loss = tail_mean(val_losses) if val_losses else 0.0
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
    score.update(common_score_extras(
        results, proposed_config, training_options, world_size, model_dict, training_dict, strategy_name
    ))
    score["knobs"]["init_from_tag"] = init_from_tag
    score["knobs"]["init_from_path"] = init_from_path
    if partial_load_warning:
        score["partial_load_warning"] = partial_load_warning

    no_data = len(results.step_val_accuracy.min_max_avg) == 0 and len(results.step_accuracy.min_max_avg) == 0
    if no_data:
        return score, "failed", "No training data collected"
    return score, "success", None


if __name__ == "__main__":
    run(build_and_run, log_tag="ft")
