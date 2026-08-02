"""
Autonomous LLM-guided hyperparameter optimization for stage-2 SFT.

Takes a pretrained base LM checkpoint and searches over SFT hyperparameters,
training on a chat/instruction mixture (smoltalk + everyday-conversations +
self-oss-instruct). Scoring uses held-out chat val accuracy.

Usage:
    OPENROUTER_API_KEY=... python autoparam_finetune.py \
        --pretrain-tag autoparam-fineweb-256 --max-experiments 50

    OPENROUTER_API_KEY=... python autoparam_finetune.py \
        --use-best-of fineweb-256 --max-experiments 50
"""

import argparse
import json
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
from utils.load_mode_from_checkpoint import load_modeL_tag
from utils.checkpoints import StorageBox
from scheduler.cooperative import install_shutdown_handler

_LOCKED_ARCH_FIELDS = (
    "vocab_size", "dim_embeddings", "num_attention_heads", "num_transformer_layers",
    "feed_forward_layer", "bias", "hc_n", "transformer_layer", "positional_embedding",
    "normalization_layer", "attention_type", "qk_norm", "ffn_activation",
    "norm_placement", "tie_embeddings",
)


def _load_pretrain_arch(init_tag: str) -> dict:
    from enum import Enum
    from training.model import Config
    path = load_modeL_tag(init_tag)
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    cfg_dict = json.loads(storage.load_bytes(os.path.join(path, "config.json")))
    cfg_obj = Config.from_json(cfg_dict)
    out = {}
    for k in _LOCKED_ARCH_FIELDS:
        v = getattr(cfg_obj, k, None)
        if isinstance(v, Enum):
            out[k] = v.name
        else:
            out[k] = v
    return out

install_shutdown_handler()


CHAT_TAG = "autoparam-finetune"

_SFT_SYSTEM_PROMPT = f"""You are an expert ML researcher running autonomous hyperparameter optimization \
for stage-2 SUPERVISED FINETUNING (SFT) of a PRETRAINED transformer LM on a chat/instruction mixture \
(smoltalk + everyday-conversations + self-oss-instruct). The base model has already learned language; \
your job is to turn it into a chatbot WITHOUT destroying that knowledge.

Goal: maximize held-out chat **val_accuracy** (and minimize val_loss) on the chat mixture.

{SEARCH_SPACE_DESCRIPTION}

SFT-specific guidance (READ CAREFULLY):
- Learning rate should start ~10x LOWER than typical pretraining LR. Try lr in [0.00001, 0.0005]. \
Large LR will catastrophically forget the base model.
- Use FEWER total steps than pretraining. Prefer training_minutes in [15, 60, 120]. SFT converges fast.
- Prefer SHORT warmup (100-500 steps) and cosine or warmup_exp_decay schedulers with small min_lr_ratio.
- AdamW or Muon_hybrid with weight_decay 0.0-0.1 are good defaults.
- Architecture changes (num_transformer_layers, dim_embeddings, num_attention_heads, \
transformer_layer, positional_embedding) are STRONGLY DISCOURAGED — changing them forces \
re-initialization of mismatched weights, throwing away the pretrained representations. \
Only propose an arch change if you have a clear hypothesis that justifies the cost; \
otherwise keep the architecture identical to the baseline (which mirrors the pretrain config).
- Smaller batch_size (16-64) with accumulation_steps 1-4 is usually fine for SFT.
- Dropout 0.0-0.05 is typical for SFT.

Hard constraints:
- dim_embeddings MUST be divisible by num_attention_heads
- All enum values must match exactly (case-sensitive)
- lr must be between 0.0001 and 0.01 (the schema floor; for SFT prefer the low end)
- Do not repeat a configuration nearly identical to one that already failed

You will receive the experiment history and must respond with a single valid JSON object.
No markdown, no prose outside the JSON.
"""


class SFTLLMProposer(LLMProposer):
    def propose(self, state, baseline_dict):
        import autoparam as _ap
        original = _ap._SYSTEM_PROMPT
        _ap._SYSTEM_PROMPT = _SFT_SYSTEM_PROMPT
        try:
            return super().propose(state, baseline_dict)
        finally:
            _ap._SYSTEM_PROMPT = original


def _resolve_init_tag(args) -> str:
    if args.pretrain_tag:
        return args.pretrain_tag
    return f"best-{args.use_best_of}"


class FinetuneAutoparamLoop(AutoparamLoopBase):
    LOG_PREFIX = "[autoparam-ft]"
    EXP_NAME_PREFIX = "autoparam_ft"
    EXECUTOR_SCRIPT = "autoparam_finetune_executor.py"
    PROMOTE_NAMESPACE = CHAT_TAG
    INCLUDES_DATASET_NAME_IN_CONFIG = False
    BASELINE_BATCH_SIZE = 32

    def __init__(self, init_tag: str, state_path: str = "autoparam_finetune_state.json", **kwargs):
        self.init_tag = init_tag
        self.locked_arch = _load_pretrain_arch(init_tag)
        print(f"{self.LOG_PREFIX} Locked arch from pretrain: {self.locked_arch}", flush=True)
        super().__init__(
            dataset=NAMED_DATASETS["smoltalk-256"],
            state_path=state_path,
            **kwargs,
        )

    def _make_proposer(self, model: str):
        return SFTLLMProposer(model=model)

    def _post_init(self):
        for k, v in self.locked_arch.items():
            if k in self.baseline_dict:
                self.baseline_dict[k] = v

    def _apply_locks_to_dict(self, cand: dict):
        for k, v in self.locked_arch.items():
            if k == "vocab_size":
                continue
            cand[k] = v

    def _apply_locks_to_config(self, cfg_obj):
        from training.model import (
            TransformerLayerType, PositionalEmbeddingType, NormalizationLayerType,
            AttentionType, FFNActivation, NormPlacement,
        )
        enum_map = {
            "transformer_layer": TransformerLayerType,
            "positional_embedding": PositionalEmbeddingType,
            "normalization_layer": NormalizationLayerType,
            "attention_type": AttentionType,
            "ffn_activation": FFNActivation,
            "norm_placement": NormPlacement,
        }
        for k, v in self.locked_arch.items():
            if v is None:
                setattr(cfg_obj, k, None)
            elif k in enum_map:
                setattr(cfg_obj, k, enum_map[k][v] if isinstance(v, str) else v)
            else:
                setattr(cfg_obj, k, v)

    def _run_tag(self) -> str:
        return "autoparam-finetune"

    def _extra_config_data(self) -> dict:
        return {"init_from_tag": self.init_tag}

    def _extract_objective(self, score: dict) -> float:
        return float(score.get("val_accuracy", score.get("final_accuracy", -1.0)))

    def _format_success_log(self, score: dict) -> str:
        params_m = score.get("params_count", 0) / 1e6
        pmem = score.get("peak_memory_gb", 0)
        tokens = score.get("tokens_seen", 0)
        warn = score.get("partial_load_warning")
        warn_str = f"  WARNING: {warn}" if warn else ""
        return (
            f"Result: val_acc={score.get('val_accuracy', float('nan')):.2f}%  "
            f"val_loss={score.get('val_loss', float('nan')):.3f}  "
            f"slope={score.get('accuracy_slope', 0):.4f}  "
            f"params={params_m:.1f}M  peak_mem={pmem:.2f}GB  tokens={tokens:,}{warn_str}"
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
    parser = argparse.ArgumentParser(description="Autonomous SFT hyperparameter optimization")
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--pretrain-tag", help="Explicit pretrained checkpoint tag")
    init_group.add_argument("--use-best-of", metavar="DATASET",
                            help="Resolve to best-<DATASET> at startup (e.g. fineweb-256)")
    parser.add_argument("--max-experiments", type=lambda s: None if int(s) < 0 else int(s), default=50)
    parser.add_argument("--budget", type=float, default=5.00, metavar="USD")
    parser.add_argument("--state-file", default="autoparam_finetune_state.json")
    parser.add_argument("--distributed-strategy", default="fsdp", choices=["none", "ddp", "fsdp"])
    parser.add_argument(
        "--nproc-per-node", type=int,
        default=max(1, torch.cuda.device_count()),
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--use-best", action="store_true",
                        help="Resolve pretrain tag via best.json instead of latest.json")
    parser.add_argument("--check-spend", action="store_true")
    parser.add_argument(
        "--timeout-minutes", type=int,
        default=int(os.environ.get("TRAINING_TIME_MINUTES", TRAINING_TIME_MINUTES)),
        help="Default per-experiment timeout (LLM may propose different training_minutes)",
    )
    args = parser.parse_args()

    if args.check_spend:
        daily = fetch_openrouter_daily_usage()
        if daily < 0:
            print("Failed to fetch OpenRouter usage (check OPENROUTER_API_KEY).")
        else:
            print(f"OpenRouter spend today: ${daily:.4f}")
        sys.exit(0)

    if args.use_best:
        os.environ["PRETRAIN_TAG_FILE"] = "best.json"

    init_tag = _resolve_init_tag(args)
    try:
        resolved_path = load_modeL_tag(init_tag)
        print(f"[autoparam-ft] Resolved init tag '{init_tag}' -> {resolved_path}", flush=True)
    except Exception as e:
        print(f"[autoparam-ft] ERROR: failed to resolve init tag '{init_tag}': {e}", flush=True)
        sys.exit(1)

    strategy = DistributedStrategy[args.distributed_strategy.upper()]

    FinetuneAutoparamLoop(
        init_tag=init_tag,
        max_experiments=args.max_experiments,
        experiment_timeout_minutes=args.timeout_minutes,
        state_path=args.state_file,
        budget_usd=args.budget,
        distributed_strategy=strategy,
        nproc_per_node=args.nproc_per_node,
        max_consecutive_failures=args.max_consecutive_failures,
        random_only=args.random_only,
    ).run()
