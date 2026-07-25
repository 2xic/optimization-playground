"""
Executor for autoparam_evm_triplet - runs a single triplet contrastive experiment
over the evm-triplet-256 dataset. Trains from scratch.
"""

import random

import torch
from autoparam_executor_lib import bootstrap, run, base_val_score, common_score_extras, prepare_common, no_data_result, apply_resume

bootstrap()

from experiments import (
    execute,
    NAMED_DATASETS,
    make_triplet_objective_factory,
)
from training.objectives import TripletLoss
from training.model import Model
from training.heads import TripletEmbeddingModel


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
    training_dict = cfg["training_config"]

    dataset = NAMED_DATASETS[cfg.get("dataset_name", DATASET_NAME)]
    proposed_config, training_options, world_size, _device, strategy_name = prepare_common(cfg, rank, dataset)

    loss_name = training_dict.get("triplet_loss")
    loss = random.Random(exp_name).choice(list(TripletLoss)) if loss_name is None else TripletLoss[loss_name.upper()]
    training_dict["triplet_loss"] = loss.name
    log(f"training triplet embedding from scratch (loss={loss.name})")
    inner = Model(proposed_config)
    model = TripletEmbeddingModel(inner)
    resume = apply_resume(cfg, model, training_options, log)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(
        dataset, exp_name, model, training_options,
        objective_factory=make_triplet_objective_factory(loss),
        batch_adapter=_batch_adapter,
        resume=resume,
    )

    score = base_val_score(results, params_count)
    score["triplet_loss"] = loss.name
    score.update(common_score_extras(
        results, proposed_config, training_options, world_size, cfg["model_config"], training_dict, strategy_name
    ))
    return no_data_result(score, results)


if __name__ == "__main__":
    run(build_and_run, log_tag="evm-triplet")
