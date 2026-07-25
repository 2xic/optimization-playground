"""
Executor for autoparam_evm_selector - runs a single 3-class selector
classification experiment over an evm-selector-bytecode-* dataset. Trains
from scratch. Dataset chosen via AUTOPARAM_SELECTOR_DATASET env var.
"""

import os

import torch
from autoparam_executor_lib import bootstrap, run, base_val_score, common_score_extras, prepare_common, no_data_result, apply_resume

bootstrap()

from experiments import (
    execute,
    NAMED_DATASETS,
    create_selector_classification_objective,
)
from training.model import Model
from training.heads import ClassificationHeadModel


DATASET_NAME = os.environ.get("AUTOPARAM_SELECTOR_DATASET", "evm-selector-bytecode-17")
NUM_CLASSES = 3


def _batch_adapter(batch):
    return batch["window_tokens"], batch["label"]


def build_and_run(cfg, rank, log):
    dataset = NAMED_DATASETS[cfg.get("dataset_name", DATASET_NAME)]
    proposed_config, training_options, world_size, _device, strategy_name = prepare_common(cfg, rank, dataset)

    log(f"training selector classifier from scratch on {dataset.name}")
    inner = Model(proposed_config)
    model = ClassificationHeadModel(inner, num_classes=NUM_CLASSES)
    resume = apply_resume(cfg, model, training_options, log)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(
        dataset, cfg["exp_name"], model, training_options,
        objective_factory=create_selector_classification_objective,
        batch_adapter=_batch_adapter,
        resume=resume,
    )

    score = base_val_score(results, params_count)
    score.update(common_score_extras(
        results, proposed_config, training_options, world_size, cfg["model_config"], cfg["training_config"], strategy_name
    ))
    return no_data_result(score, results)


if __name__ == "__main__":
    run(build_and_run, log_tag="evm-selector")
