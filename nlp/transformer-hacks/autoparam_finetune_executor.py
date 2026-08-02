"""
Executor for autoparam_finetune - runs a single SFT experiment under torchrun.
Launched by autoparam_finetune.py via subprocess.
"""

import os

from autoparam_executor_lib import bootstrap, run, base_val_score, common_score_extras, prepare_common, no_data_result, warm_start, apply_resume

bootstrap()

import torch
from experiments import execute, NAMED_DATASETS
from utils.web_dataloader import WebDataloader
from utils.mixture_dataloader import WebDataloaderMixture

WARMUP_CLAMP_ENABLED = False

FINETUNE_DATASET_NAMES = [
    "smoltalk-256",
    "smoltalk2-sft-256",
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


def build_and_run(cfg, rank, log):
    init_from_tag = cfg.get("init_from_tag")
    init_from_path = cfg.get("init_from_path")
    if not (init_from_tag or init_from_path):
        raise ValueError("config requires init_from_tag or init_from_path")

    proposed_config, training_options, world_size, _device, strategy_name = prepare_common(
        cfg, rank, NAMED_DATASETS[FINETUNE_DATASET_NAMES[0]]
    )

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

    model, partial_load_warning = warm_start(cfg, proposed_config, log)
    resume = apply_resume(cfg, model, training_options, log)

    params_count = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    _, results = execute(train_mixture, cfg["exp_name"], model, training_options, resume=resume)

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
    run(build_and_run, log_tag="ft")
