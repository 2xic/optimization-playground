"""
Executor for autoparam_feedback - runs a single binary-classification SFT
experiment over the feedback-256 dataset.
"""

from autoparam_executor_lib import bootstrap, run, base_val_score, common_score_extras, prepare_common, no_data_result, warm_start, apply_resume

bootstrap()

import torch
from experiments import (
    execute,
    NAMED_DATASETS,
    create_feedback_classification_objective,
)
from training.heads import ClassificationHeadModel


DATASET_NAME = "feedback_256"


def _batch_adapter(batch):
    return batch["input_ids"], batch["feedback"]


def build_and_run(cfg, rank, log):
    init_from_tag = cfg.get("init_from_tag")
    init_from_path = cfg.get("init_from_path")

    dataset = NAMED_DATASETS[cfg.get("dataset_name", DATASET_NAME)]
    proposed_config, training_options, world_size, _device, strategy_name = prepare_common(cfg, rank, dataset)

    inner, partial_load_warning = warm_start(cfg, proposed_config, log)
    model = ClassificationHeadModel(inner)
    resume = apply_resume(cfg, model, training_options, log)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(
        dataset, cfg["exp_name"], model, training_options,
        objective_factory=create_feedback_classification_objective,
        batch_adapter=_batch_adapter,
        resume=resume,
    )

    score = base_val_score(results, params_count)
    score.update(common_score_extras(
        results, proposed_config, training_options, world_size, cfg["model_config"], cfg["training_config"], strategy_name
    ))
    score["knobs"]["init_from_tag"] = init_from_tag
    score["knobs"]["init_from_path"] = init_from_path
    if partial_load_warning:
        score["partial_load_warning"] = partial_load_warning
    return no_data_result(score, results)


if __name__ == "__main__":
    run(build_and_run, log_tag="feedback")
