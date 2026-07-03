"""
Executor for autoparam_feedback - runs a single binary-classification SFT
experiment over the feedback-256 dataset.
"""

import math
import os

from autoparam_executor_lib import setup_env, run, tail_mean, slope, common_score_extras

setup_env()

import torch
from dotenv import load_dotenv

load_dotenv()

from experiments import (
    execute,
    NAMED_DATASETS,
    PretrainedModelConstruction,
    create_feedback_classification_objective,
)
from training.model import Model, SamplingMethod
from training.heads import ClassificationHeadModel
from training.trainer import DistributedStrategy
from autoparam import ConfigSerializer
from scheduler.cooperative import install_shutdown_handler
install_shutdown_handler()
from utils.load_mode_from_checkpoint import load_modeL_tag, load_model_from_path
from utils.checkpoints import apply_checkpoint_tag


DATASET_NAME = "feedback_256"


def _batch_adapter(batch):
    return batch["input_ids"], batch["feedback"]


def _partial_load(target_model, source_state_dict):
    target_sd = target_model.state_dict()
    loaded, skipped = [], []
    for k, v in source_state_dict.items():
        key = k.replace("module.", "")
        if key in target_sd and target_sd[key].shape == v.shape:
            target_sd[key] = v
            loaded.append(key)
        else:
            skipped.append(key)
    target_model.load_state_dict(target_sd, strict=False)
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

    dataset = NAMED_DATASETS[cfg.get("dataset_name", DATASET_NAME)]
    proposed_config = ConfigSerializer.dict_to_config(model_dict, dataset)
    proposed_config.sampling_method = SamplingMethod.ARGMAX

    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, timeout_minutes, strategy, device
    )
    apply_checkpoint_tag(training_options, cfg)

    partial_load_warning = None
    if init_from_tag or init_from_path:
        checkpoint_path = load_modeL_tag(init_from_tag) if init_from_tag else init_from_path
        log(f"loading pretrained base from: {checkpoint_path}")
        base_model, pretrain_config = load_model_from_path(checkpoint_path)
        arch_keys = ("num_transformer_layers", "dim_embeddings", "num_attention_heads",
                     "vocab_size", "sequence_length", "transformer_layer", "positional_embedding")
        arch_match = all(
            getattr(proposed_config, k, None) == getattr(pretrain_config, k, None)
            for k in arch_keys
        )
        if arch_match:
            log("arch matches pretrain - using base directly")
            inner = PretrainedModelConstruction(pretrain_config, base_model)()
        else:
            log("arch MISMATCH - partial loading matching tensors")
            inner = Model(proposed_config)
            loaded, skipped = _partial_load(inner, base_model.state_dict())
            partial_load_warning = (
                f"arch mismatch: loaded {len(loaded)} tensors, skipped {len(skipped)}."
            )
            log(partial_load_warning)
        del base_model
    else:
        log("training feedback head from scratch (no pretrain)")
        inner = Model(proposed_config)

    model = ClassificationHeadModel(inner)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(
        dataset, exp_name, model, training_options,
        objective_factory=create_feedback_classification_objective,
        batch_adapter=_batch_adapter,
    )

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
    run(build_and_run, log_tag="feedback")
