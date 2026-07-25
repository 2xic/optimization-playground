"""
Executor for autoparam - runs a single training experiment under torchrun.
Launched by autoparam.py (coordinator) via subprocess.

Usage (not called directly):
    torchrun --nproc_per_node=N autoparam_executor.py --config <path> --result <path>
"""

from autoparam_executor_lib import bootstrap, run, common_score_extras, val_series, apply_val_metrics, tail_mean, prepare_common, warm_start, apply_resume

bootstrap()

import torch
from experiments import execute, NAMED_DATASETS
from autoparam import StabilityMetric


def build_and_run(cfg, rank, log):
    exp_name = cfg["exp_name"]
    dataset = NAMED_DATASETS[cfg["dataset_name"]]
    config, training_options, world_size, _device, strategy_name = prepare_common(cfg, rank, dataset)

    model, _ = warm_start(cfg, config, log)
    resume = apply_resume(cfg, model, training_options, log)
    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(dataset, exp_name, model, training_options, resume=resume)
    score = StabilityMetric.compute(results)
    score["params_count"] = params_count
    try:
        val_accs, val_losses = val_series(results)
        apply_val_metrics(score, val_accs, val_losses)
        if val_accs:
            score["overfit_gap"] = float(score.get("final_accuracy", 0.0) - tail_mean(val_accs))
    except Exception:
        pass
    score.update(common_score_extras(
        results, config, training_options, world_size, cfg["model_config"], cfg["training_config"], strategy_name
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
