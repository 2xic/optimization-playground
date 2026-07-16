"""
Autonomous LLM-guided hyperparameter optimization for evm-selector-bytecode-*.

Trains a small transformer with a 3-class classification head (none/function/
event) over windowed EVM bytecode token sequences. Trains from scratch.
Masked mean-pooling + linear head, cross-entropy loss. Scoring uses held-out
val_accuracy. Dataset selected via --dataset (default evm-selector-bytecode-17).
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


SELECTOR_TAG = "autoparam-evm-selector"

_SELECTOR_SYSTEM_PROMPT = f"""You are an expert ML researcher running autonomous hyperparameter optimization \
for 3-CLASS CLASSIFICATION (cross-entropy) on the evm-selector-bytecode dataset (EVM bytecode windows). \
Each row carries window_tokens (a short opcode-id sequence, right-padded with 0) and an integer label in \
{{0=none, 1=function, 2=event}}. A transformer encoder produces a masked mean-pooled representation; a \
linear head outputs 3 logits; loss is cross_entropy (optionally label-smoothed).

Goal: maximize held-out **val_accuracy** (and minimize val_loss / cross-entropy).

{SEARCH_SPACE_DESCRIPTION}

Selector-classification-specific guidance:
- Sequences are VERY SHORT (17 or 33 tokens). Small models work well: dim_embeddings 64-256, \
num_transformer_layers 2-6, num_attention_heads 2-8. Avoid oversized models — they overfit.
- The classes are IMBALANCED (event ~4%, function ~34%, none ~62%). Use label_smoothing 0.0-0.1 to help \
generalization; macro performance matters more than raw accuracy.
- Training is FROM SCRATCH. lr in [0.0003, 0.003]; AdamW or Muon_hybrid are fine.
- Prefer training_minutes in [15, 30, 60]; classification on a tiny label set converges fast.
- Short warmup (50-500 steps), cosine or warmup_exp_decay, min_lr_ratio 0.05-0.1.
- batch_size can be LARGE (128-512) given short sequences. weight_decay 0.0-0.1.
- Dropout 0.0-0.2. The 33-token split is SMALL (~21k rows) so favor more regularization there.

Hard constraints:
- dim_embeddings MUST be divisible by num_attention_heads
- All enum values must match exactly (case-sensitive)
- Do not repeat a configuration nearly identical to one that already failed

You will receive the experiment history and must respond with a single valid JSON object.
No markdown, no prose outside the JSON.
"""


class SelectorLLMProposer(LLMProposer):
    def propose(self, state, baseline_dict):
        import autoparam as _ap
        original = _ap._SYSTEM_PROMPT
        _ap._SYSTEM_PROMPT = _SELECTOR_SYSTEM_PROMPT
        try:
            return super().propose(state, baseline_dict)
        finally:
            _ap._SYSTEM_PROMPT = original


class SelectorAutoparamLoop(AutoparamLoopBase):
    LOG_PREFIX = "[autoparam-evm-selector]"
    EXP_NAME_PREFIX = "autoparam_evs"
    EXECUTOR_SCRIPT = "autoparam_evm_selector_executor.py"
    PROMOTE_NAMESPACE = SELECTOR_TAG
    INCLUDES_DATASET_NAME_IN_CONFIG = True
    BASELINE_BATCH_SIZE = 256

    def __init__(self, dataset_name: str = "evm-selector-bytecode-17",
                 version: str = "v1",
                 state_path: str = None, **kwargs):
        self.version = version
        self.PROMOTE_NAMESPACE = SELECTOR_TAG if version == "v1" else f"autoparam-{dataset_name}"
        if state_path is None:
            state_path = f"autoparam_{dataset_name}_state.json"
        super().__init__(
            dataset=NAMED_DATASETS[dataset_name],
            state_path=state_path,
            **kwargs,
        )

    def _make_proposer(self, model: str):
        return SelectorLLMProposer(model=model)

    def _run_tag(self) -> str:
        if self.version == "v1":
            return f"autoparam-evm-selector-{self.dataset.name}"
        return f"autoparam-{self.dataset.name}"

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
    parser = argparse.ArgumentParser(description="Autonomous evm-selector-bytecode hyperparameter optimization")
    parser.add_argument("--dataset", default="evm-selector-bytecode-17",
                        choices=["evm-selector-bytecode-17", "evm-selector-bytecode-33",
                                 "evm-selector-bytecode-17-v2", "evm-selector-bytecode-33-v2"])
    parser.add_argument("--version", default="v1")
    parser.add_argument("--max-experiments", type=lambda s: None if int(s) < 0 else int(s), default=50)
    parser.add_argument("--budget", type=float, default=5.00, metavar="USD")
    parser.add_argument("--state-file", default=None)
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

    SelectorAutoparamLoop(
        dataset_name=args.dataset,
        version=args.version,
        max_experiments=args.max_experiments,
        experiment_timeout_minutes=args.timeout_minutes,
        state_path=args.state_file,
        budget_usd=args.budget,
        distributed_strategy=strategy,
        nproc_per_node=args.nproc_per_node,
        max_consecutive_failures=args.max_consecutive_failures,
        random_only=args.random_only,
    ).run()
