"""
Autonomous LLM-guided hyperparameter optimization for evm-triplet-256.

Trains a transformer embedding model with triplet contrastive loss over
(anchor, positive, negative) bytecode token sequences. Trains from scratch.
Scoring uses held-out triplet ranking accuracy (sim(a,p) > sim(a,n)).
"""

import argparse
import os
import sys

import torch

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments import NAMED_DATASETS, TRAINING_TIME_MINUTES
from training.trainer import DistributedStrategy
from autoparam import (
    AutoparamLoopBase,
    LLMProposer,
    SEARCH_SPACE_DESCRIPTION,
    fetch_openrouter_daily_usage,
)
from scheduler.cooperative import install_shutdown_handler

install_shutdown_handler()


TRIPLET_TAG = "autoparam-evm-triplet"

_TRIPLET_SYSTEM_PROMPT = f"""You are an expert ML researcher running autonomous hyperparameter optimization \
for TRIPLET CONTRASTIVE LEARNING on the evm-triplet-256 dataset (EVM bytecode similarity). \
Each batch row carries (anchor_tokens, positive_tokens, negative_tokens). A transformer encoder produces \
an L2-normalized pooled embedding; the loss is triplet_margin_loss(anchor, positive, negative). \
Eval accuracy = fraction of triplets where sim(anchor, positive) > sim(anchor, negative).

Goal: maximize held-out **val_accuracy** (ranking accuracy) on the triplet val split.

{SEARCH_SPACE_DESCRIPTION}

Triplet-contrastive guidance:
- Training is FROM SCRATCH. Use lr in [0.0001, 0.003]; AdamW or Muon_hybrid are both fine.
- Prefer training_minutes in [30, 60, 120].
- Warmup 200-1000 steps with cosine or warmup_exp_decay; min_lr_ratio 0.05-0.1.
- batch_size should be MODEST (8-32 anchors per rank) since each effective rows becomes 3x after \
concatenation of (anchor, positive, negative) along the batch dimension.
- Dropout 0.0-0.1.
- Architecture is free to vary; smaller-but-deeper models often help contrastive embeddings.

Hard constraints:
- dim_embeddings MUST be divisible by num_attention_heads
- All enum values must match exactly (case-sensitive)
- Do not repeat a configuration nearly identical to one that already failed

You will receive the experiment history and must respond with a single valid JSON object.
No markdown, no prose outside the JSON.
"""


class TripletLLMProposer(LLMProposer):
    def propose(self, state, baseline_dict):
        import autoparam as _ap
        original = _ap._SYSTEM_PROMPT
        _ap._SYSTEM_PROMPT = _TRIPLET_SYSTEM_PROMPT
        try:
            return super().propose(state, baseline_dict)
        finally:
            _ap._SYSTEM_PROMPT = original


class TripletAutoparamLoop(AutoparamLoopBase):
    LOG_PREFIX = "[autoparam-evm-triplet]"
    EXP_NAME_PREFIX = "autoparam_evt"
    EXECUTOR_SCRIPT = "autoparam_evm_triplet_executor.py"
    PROMOTE_NAMESPACE = TRIPLET_TAG
    INCLUDES_DATASET_NAME_IN_CONFIG = False
    BASELINE_BATCH_SIZE = 32

    def __init__(self, state_path: str = "autoparam_evm_triplet_state.json", **kwargs):
        super().__init__(
            dataset=NAMED_DATASETS["evm-cluster_triplet-256"],
            state_path=state_path,
            plot_subdir=TRIPLET_TAG,
            **kwargs,
        )

    def _make_proposer(self, model: str):
        return TripletLLMProposer(model=model)

    def _run_tag(self) -> str:
        return "autoparam-evm-triplet"

    def _extract_accuracy(self, score: dict) -> float:
        return float(score.get("val_accuracy", score.get("final_accuracy", -1.0)))

    def _format_success_log(self, score: dict) -> str:
        params_m = score.get("params_count", 0) / 1e6
        pmem = score.get("peak_memory_gb", 0)
        tokens = score.get("tokens_seen", 0)
        return (
            f"Result: val_acc={score.get('val_accuracy', float('nan')):.2f}%  "
            f"val_loss={score.get('val_loss', float('nan')):.3f}  "
            f"slope={score.get('accuracy_slope', 0):.4f}  "
            f"params={params_m:.1f}M  peak_mem={pmem:.2f}GB  tokens={tokens:,}"
        )

    def _format_best_log(self, best) -> str:
        return (
            f"Best so far: #{best.experiment_id}  "
            f"val_acc={best.score.get('val_accuracy', best.score.get('final_accuracy', 0)):.2f}%  "
            f"val_loss={best.score.get('val_loss', 0):.3f}"
        )

    def _summary_sort_key(self, e):
        return e.score.get("val_accuracy", e.score.get("final_accuracy", 0))

    def _format_summary_line(self, rank: int, e) -> str:
        s = e.score
        return (
            f"  #{rank}  exp={e.experiment_id:03d}  "
            f"val_acc={s.get('val_accuracy', s.get('final_accuracy', 0)):.2f}%  "
            f"val_loss={s.get('val_loss', 0):.3f}  "
            f"slope={s.get('accuracy_slope', 0):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous evm-triplet-256 hyperparameter optimization")
    parser.add_argument("--max-experiments", type=lambda s: None if int(s) < 0 else int(s), default=50)
    parser.add_argument("--budget", type=float, default=5.00, metavar="USD")
    parser.add_argument("--state-file", default="autoparam_evm_triplet_state.json")
    parser.add_argument("--distributed-strategy", default="fsdp", choices=["none", "ddp", "fsdp"])
    parser.add_argument(
        "--nproc-per-node", type=int,
        default=max(1, torch.cuda.device_count()),
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--check-spend", action="store_true")
    parser.add_argument(
        "--timeout-minutes", type=int,
        default=int(os.environ.get("TRAINING_TIME_MINUTES", TRAINING_TIME_MINUTES)),
    )
    args = parser.parse_args()

    if args.check_spend:
        daily = fetch_openrouter_daily_usage()
        if daily < 0:
            print("Failed to fetch OpenRouter usage (check OPENROUTER_API_KEY).")
        else:
            print(f"OpenRouter spend today: ${daily:.4f}")
        sys.exit(0)

    strategy = DistributedStrategy[args.distributed_strategy.upper()]

    TripletAutoparamLoop(
        max_experiments=args.max_experiments,
        experiment_timeout_minutes=args.timeout_minutes,
        state_path=args.state_file,
        budget_usd=args.budget,
        distributed_strategy=strategy,
        nproc_per_node=args.nproc_per_node,
        max_consecutive_failures=args.max_consecutive_failures,
        random_only=args.random_only,
    ).run()
