"""
Executor for autoparam_evm_triplet - runs a single triplet contrastive experiment
over the evm-triplet-256 dataset. Trains from scratch.
"""

import math
import os

import torch
from autoparam_executor_lib import setup_env, run, tail_mean, slope, common_score_extras

setup_env()

from dotenv import load_dotenv

load_dotenv()

from experiments import (
    execute,
    NAMED_DATASETS,
    create_triplet_contrastive_objective,
)
from training.model import Model, SamplingMethod
from training.heads import TripletEmbeddingModel
from training.trainer import DistributedStrategy
from autoparam import ConfigSerializer
from scheduler.cooperative import install_shutdown_handler
install_shutdown_handler()
from utils.checkpoints import apply_checkpoint_tag


DATASET_NAME = "evm-cluster_triplet-256"


def _batch_adapter(batch):
    anchor = batch["anchor_tokens"]
    positive = batch["positive_tokens"]
    negative = batch["negative_tokens"]
    x = torch.cat([anchor, positive, negative], dim=0)
    y = torch.zeros(anchor.shape[0], device=anchor.device)
    return x, y


def build_and_run(cfg, rank, log):
    exp_name = cfg["exp_name"]
    timeout_minutes = cfg["timeout_minutes"]
    model_dict = cfg["model_config"]
    training_dict = cfg["training_config"]
    strategy_name = cfg.get("distributed_strategy", "FSDP")
    strategy = DistributedStrategy[strategy_name]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{rank}")

    dataset = NAMED_DATASETS[DATASET_NAME]
    proposed_config = ConfigSerializer.dict_to_config(model_dict, dataset)
    proposed_config.sampling_method = SamplingMethod.ARGMAX

    training_options = ConfigSerializer.dict_to_training_options(
        training_dict, timeout_minutes, strategy, device
    )
    apply_checkpoint_tag(training_options, cfg)

    log("training triplet embedding from scratch")
    inner = Model(proposed_config)
    model = TripletEmbeddingModel(inner)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(
        dataset, exp_name, model, training_options,
        objective_factory=create_triplet_contrastive_objective,
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

    no_data = len(results.step_val_accuracy.min_max_avg) == 0 and len(results.step_accuracy.min_max_avg) == 0
    if no_data:
        return score, "failed", "No training data collected"
    return score, "success", None


if __name__ == "__main__":
    run(build_and_run, log_tag="evm-triplet")
