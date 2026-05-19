"""
Autonomous LLM-guided hyperparameter optimization, inspired by karpathy/autoresearch.

Instead of modifying training code directly, this system searches over the structured
hyperparameter space (model architecture, optimizer, scheduler) using an LLM to propose
configurations based on the full history of results.

Runs a loop of short training experiments, using an LLM (via OpenRouter) to
propose the next configuration based on the full history of results.

Usage:
    OPENROUTER_API_KEY=... TRAINING_TIME_MINUTES=20 python autoparam.py \
        --dataset fineweb-256 --max-experiments 40

Resume after crash (state file is loaded automatically):
    OPENROUTER_API_KEY=... python autoparam.py --dataset fineweb-256 --max-experiments 40
"""

import json
import math
import os
import re
import argparse
import hashlib
import signal
import statistics
import time
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List

import torch
import pynvml

from dotenv import load_dotenv

load_dotenv()

# Reduce CUDA memory fragmentation so OOM on one experiment doesn't cascade into the next.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments import (
    create_default_config,
    NAMED_DATASETS,
    TRAINING_TIME_MINUTES,
)
from training.model import (
    TransformerLayerType,
    PositionalEmbeddingType,
    NormalizationLayerType,
    AttentionType,
    FFNActivation,
    NormPlacement,
    Model,
)
from training.trainer import TrainingOptions, DistributedStrategy
from training.optimizer import (
    AdamConfig,
    AdamWConfig,
    RMSpropConfig,
    MuonConfig,
    NoamScheduler,
    WarmupExpDecay,
    StepExponentialLR,
    CosineWithWarmup,
    TrapezoidalLR,
)
from utils.plot import Results
import matplotlib.pyplot as plt

STABILITY_TAIL_FRACTION = 0.25
STEPS_TO_ACCURACY_THRESHOLD = 50.0  # percent — convergence speed marker


_GPU_MEMORY_HEADROOM = 0.6


def _total_gpu_memory_gb() -> float:
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        per_gpu = min(
            pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total
            for i in range(count)
        )
        pynvml.nvmlShutdown()
        return (per_gpu / (1024 ** 3)) * _GPU_MEMORY_HEADROOM
    except Exception:
        return 16.0 * _GPU_MEMORY_HEADROOM


def _estimate_model_gb(config, num_gpus: int = 1) -> float:
    padded_vocab = math.ceil(config.vocab_size / 128) * 128
    num_params = (
        padded_vocab * config.dim_embeddings
        + config.num_transformer_layers * (
            4 * config.dim_embeddings ** 2
            + 2 * config.dim_embeddings * config.feed_forward_layer
        )
    )
    bytes_per_param = 4
    optimizer_multiplier = 3
    param_gb = num_params * bytes_per_param * optimizer_multiplier / (1024 ** 3) / max(1, num_gpus)
    max_batch = 128
    seq_len = getattr(config, "sequence_length", 256)
    activation_gb = max_batch * seq_len * config.dim_embeddings * config.num_transformer_layers * 4 * 4 / (1024 ** 3)
    return param_gb + activation_gb


def _estimate_footprint_gb_meta(config, batch_size: int, num_gpus: int = 1) -> tuple[float, int]:
    """Build Model on meta device, count params, return (estimated_gb, num_params)."""
    with torch.device("meta"):
        model = Model(config)
    num_params = sum(p.numel() for p in model.parameters())
    seq_len = getattr(config, "sequence_length", 256) or getattr(config, "context_length", 256) or 256
    D = config.dim_embeddings
    L = config.num_transformer_layers
    H = config.num_attention_heads
    F = config.feed_forward_layer
    B = batch_size
    S = seq_len
    GB = 1024 ** 3

    param_gb = num_params * 4 / GB / max(1, num_gpus)
    optimizer_gb = 2 * num_params * 4 / GB / max(1, num_gpus)
    grad_gb = num_params * 4 / GB / max(1, num_gpus)

    bytes_per_act = 4
    attn_scores = B * H * S * S * bytes_per_act
    attn_qkv = 4 * B * S * D * bytes_per_act
    ffn_hidden = 3 * B * S * F * bytes_per_act
    residual = 4 * B * S * D * bytes_per_act
    per_layer = attn_scores + attn_qkv + ffn_hidden + residual
    activation_gb = (L * per_layer + 2 * B * S * config.vocab_size * bytes_per_act) / GB
    return param_gb + optimizer_gb + grad_gb + activation_gb, num_params

LLM_MODEL = "anthropic/claude-opus-4-5"
LLM_MAX_TOKENS = 1024
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HISTORY_WINDOW = 15
MAX_TRAINING_MINUTES = 480

SEARCH_SPACE_DESCRIPTION = """
Searchable hyperparameter space (use ONLY the values listed):

Model config:
  dim_embeddings:          [128, 256, 384, 512, 768, 1024]
  num_attention_heads:     [4, 6, 8, 12, 16, 24, 32]   ← dim_embeddings MUST be divisible by this
  num_transformer_layers:  [2, 4, 6, 8, 12, 16, 24]
  dropout:                 [0.0, 0.05, 0.1, 0.2]
  feed_forward_layer:      [512, 1024, 2048, 4096, 8192]
  bias:                    [true, false]
  hc_n:                    [2, 4, 8]   ← only relevant for OLMO_HYPER_CONNECTIONS variants

Architecture (exact enum names):
  transformer_layer:       SIMPLE | GPT2 | LLAMA2 | LLAMA3 | DEEPSEEK | OLMO |
                           OLMO_HYPER_CONNECTIONS | OLMO_CONSTRAINED_HYPER_CONNECTIONS |
                           OLMO_IDENTITY_HYPER_CONNECTIONS | SIMPLE_NO_ATTENTION |
                           SIMPLE_ATTENTION_AT_HOME
  positional_embedding:    NN_EMBEDDING | SINUSOIDAL | ROTARY_POSITION_ENCODING | NONE
                           (LLAMA2/LLAMA3/OLMO* use ROTARY_POSITION_ENCODING for in-attention RoPE; set NONE to disable; other values add positional encoding at input instead)
  normalization_layer:     LAYER_NORM | DyT | RMS_NORM
  attention_type:          DEFAULT | MHA | GQA | MLA
                           (DEFAULT = architecture's native attention: DEEPSEEK→MLA, LLAMA3→GQA, others→MHA)
  qk_norm:                 [true, false]   ← apply RMSNorm to Q and K after projection (improves stability)
  ffn_activation:          SWIGLU | GEGLU | REGLU | SILU   ← FFN activation; gated variants use a gate proj, SILU is non-gated
  norm_placement:          PRE | POST | SANDWICH | PERI   ← residual norm placement (PERI = Gemma3 peri-norm)
  label_smoothing:         [0.0, 0.05, 0.1]   ← cross-entropy label smoothing
  tie_embeddings:          [true, false]   ← share input embedding ↔ lm_head weights
  init_std:                [0.006, 0.01, 0.02, 0.04]   ← std of normal init for linears/embeddings
  ffn_ratio:               [2.0, 2.67, 4.0, 8.0]   ← optional alias: feed_forward_layer = round(dim_embeddings * ffn_ratio); set EITHER ffn_ratio OR feed_forward_layer
  head_dim:                [32, 64, 128]   ← optional alias: num_attention_heads = dim_embeddings // head_dim; set EITHER head_dim OR num_attention_heads

Optimizer:
  optimizer_type:          adam | adamw | rmsprop | muon | muon_hybrid
  lr:                      float in [0.0001, 0.01]
  weight_decay:            [0, 0.01, 0.05, 0.1, 0.2]
  eps:                     [1e-8, 1e-10, 1e-12]   (adam/adamw only)
  max_grad_norm:           [0, 0.5, 1.0, 5.0]   ← 0 = disabled; controls gradient clipping
  beta1:                   float in [0.85, 0.95]   (adam/adamw only)
  beta2:                   float in [0.90, 0.999]  (adam/adamw only)
  alpha:                   [0.9, 0.95, 0.99]       (rmsprop only)
  momentum:                [0, 0.1, 0.9]           (rmsprop only)
  Note: muon/muon_hybrid ignore beta1/beta2/alpha/momentum/eps — only lr/weight_decay apply
  Note: muon_hybrid uses Muon for hidden Linear weights, AdamW for embeddings/lm_head (best of both)

Scheduler:
  scheduler_type:          none | noam | warmup_exp_decay | step_exp | cosine | trapezoidal
  warmup_steps:            [100, 250, 500, 1000, 2000, 4000]
  flat_steps:              [1000, 5000, 10000, 20000]   (trapezoidal only)
  decay_steps:             [10000, 50000, 100000, 200000]
  min_lr_ratio:            [0.01, 0.05, 0.1]   ← floor as fraction of initial lr

Training:
  batch_size:              [16, 32, 64, 128]
  accumulation_steps:      [1, 2, 4, 8, 16]
  training_minutes:        [5, 15, 30, 60, 120, 240, 480]   ← wall-clock budget per experiment; --timeout-minutes is the default short budget, top-K successes auto-extend up to 480 (8h)

RHO-Loss (optional, omit unless explicitly enabled):
  rho_loss: nested dict {{"mode": "ema"|"snapshot"|"tag", "ratio": <float>, ...}}
    mode="ema" (default): optional "decay" (default 0.999), "warmup_steps" (default 100)
    mode="snapshot":      optional "snapshot_steps" (default 500), "warmup_steps" (default 100)
    mode="tag":           requires "tag": <checkpoint tag of pre-trained IL model on holdout split>
  ratio in [0.1, 0.2, 0.4]; <1 enables RHO-Loss.
  Note: when enabled, batch_size is the *mega-batch* B; effective gradient batch is B * ratio.
"""

_SYSTEM_PROMPT = f"""You are an expert ML researcher running autonomous hyperparameter optimization \
of a PyTorch transformer language model trained on web text (next-token prediction / causal LM).

Your goal: find configurations that maximize accuracy AND training stability.

{SEARCH_SPACE_DESCRIPTION}

Hard constraints:
- dim_embeddings MUST be divisible by num_attention_heads
- All enum values must match exactly (case-sensitive)
- lr must be between 0.0001 and 0.01
- Do not repeat a configuration nearly identical to one that already failed

Exploration strategy:
- If recent scores are clustered within ~2% of each other, break out — try a fundamentally \
different architecture, optimizer, or scale rather than incremental tweaks.
- Prefer bold jumps over small perturbations when the search appears stuck.

You will receive the experiment history and must respond with a single valid JSON object.
No markdown, no prose outside the JSON.
"""


class ConfigSerializer:
    @staticmethod
    def config_to_dict(config) -> dict:
        return {
            "dim_embeddings": config.dim_embeddings,
            "num_attention_heads": config.num_attention_heads,
            "num_transformer_layers": config.num_transformer_layers,
            "dropout": config.dropout,
            "feed_forward_layer": config.feed_forward_layer,
            "bias": config.bias,
            "hc_n": config.hc_n,
            "transformer_layer": config.transformer_layer.name,
            "positional_embedding": config.positional_embedding.name,
            "normalization_layer": config.normalization_layer.name,
            "attention_type": config.attention_type.name,
            "qk_norm": config.qk_norm,
            "ffn_activation": config.ffn_activation.name if config.ffn_activation is not None else None,
            "norm_placement": config.norm_placement.name if config.norm_placement is not None else None,
            "label_smoothing": config.label_smoothing,
            "tie_embeddings": config.tie_embeddings,
            "init_std": config.init_std,
        }

    @staticmethod
    def training_options_to_dict(opts: TrainingOptions) -> dict:
        opt = opts.optimizer
        opt_type = type(opt).__name__.lower().replace("config", "")
        if opt_type == "muon" and getattr(opt, "hybrid", False):
            opt_type = "muon_hybrid"
        d = {
            "optimizer_type": opt_type,
            "lr": getattr(opt, "lr", 3e-4),
            "weight_decay": getattr(opt, "weight_decay", 0),
            "max_grad_norm": getattr(opt, "max_grad_norm", 0),
            "eps": getattr(opt, "eps", 1e-8),
            "batch_size": opts.batch_size,
            "accumulation_steps": opts.accumulation_steps,
            "training_minutes": opts.training_timeout_minutes,
        }
        if hasattr(opt, "betas"):
            d["beta1"] = opt.betas[0]
            d["beta2"] = opt.betas[1]
        if hasattr(opt, "alpha"):
            d["alpha"] = opt.alpha
        if hasattr(opt, "momentum"):
            d["momentum"] = opt.momentum

        sched = opts.lr_scheduler
        if isinstance(sched, CosineWithWarmup):
            d["scheduler_type"] = "cosine"
            d["warmup_steps"] = sched.warmup_steps
            d["decay_steps"] = sched.decay_steps
            d["min_lr_ratio"] = sched.min_lr_ratio
        elif isinstance(sched, WarmupExpDecay):
            d["scheduler_type"] = "warmup_exp_decay"
            d["warmup_steps"] = sched.warmup_steps
            d["decay_steps"] = sched.decay_steps
            d["min_lr_ratio"] = sched.min_lr_ratio
        elif isinstance(sched, NoamScheduler):
            d["scheduler_type"] = "noam"
            d["warmup_steps"] = sched.warmup_steps
            d["d_model"] = sched.d_model
        elif isinstance(sched, StepExponentialLR):
            d["scheduler_type"] = "step_exp"
            d["decay_steps"] = sched.decay_steps
            d["min_lr_ratio"] = sched.min_lr_ratio
        elif isinstance(sched, TrapezoidalLR):
            d["scheduler_type"] = "trapezoidal"
            d["warmup_steps"] = sched.warmup_steps
            d["flat_steps"] = sched.flat_steps
            d["decay_steps"] = sched.decay_steps
            d["min_lr_ratio"] = sched.min_lr_ratio
        else:
            d["scheduler_type"] = "none"
        if opts.rho_loss is not None:
            from training.rho_loss import RhoLossTagConfig, RhoLossEmaConfig, RhoLossSnapshotConfig
            rl = opts.rho_loss
            if isinstance(rl, RhoLossTagConfig):
                d["rho_loss"] = {"mode": "tag", "ratio": rl.ratio, "tag": rl.tag}
            elif isinstance(rl, RhoLossEmaConfig):
                d["rho_loss"] = {"mode": "ema", "ratio": rl.ratio, "decay": rl.decay, "warmup_steps": rl.warmup_steps}
            elif isinstance(rl, RhoLossSnapshotConfig):
                d["rho_loss"] = {"mode": "snapshot", "ratio": rl.ratio, "snapshot_steps": rl.snapshot_steps, "warmup_steps": rl.warmup_steps}
        return d

    @classmethod
    def dict_to_config(cls, d: dict, dataset):
        config = create_default_config(dataset)
        config.dim_embeddings = int(d["dim_embeddings"])
        if "head_dim" in d and "num_attention_heads" not in d:
            config.num_attention_heads = max(1, config.dim_embeddings // int(d["head_dim"]))
        else:
            config.num_attention_heads = int(d["num_attention_heads"])
        config.num_transformer_layers = int(d["num_transformer_layers"])
        config.dropout = float(d["dropout"])
        if "ffn_ratio" in d and "feed_forward_layer" not in d:
            config.feed_forward_layer = int(round(config.dim_embeddings * float(d["ffn_ratio"])))
        else:
            config.feed_forward_layer = int(d["feed_forward_layer"])
        config.bias = bool(d.get("bias", False))
        config.hc_n = int(d.get("hc_n", 4))
        config.transformer_layer = TransformerLayerType[d["transformer_layer"]]
        config.positional_embedding = PositionalEmbeddingType[d["positional_embedding"]]
        config.normalization_layer = NormalizationLayerType[d["normalization_layer"]]
        config.attention_type = AttentionType[d.get("attention_type", "DEFAULT")]
        config.qk_norm = bool(d.get("qk_norm", False))
        ffn_act = d.get("ffn_activation")
        config.ffn_activation = FFNActivation[ffn_act] if ffn_act else None
        norm_pl = d.get("norm_placement")
        config.norm_placement = NormPlacement[norm_pl] if norm_pl else None
        config.label_smoothing = float(d.get("label_smoothing", 0.0))
        config.tie_embeddings = bool(d.get("tie_embeddings", True))
        config.init_std = float(d.get("init_std", 0.02))
        return cls._validate_and_repair(config)

    @classmethod
    def dict_to_training_options(cls, d: dict, timeout_minutes: int, distributed_strategy: DistributedStrategy = DistributedStrategy.FSDP, device: Optional[torch.device] = None) -> TrainingOptions:
        opt_type = d.get("optimizer_type", "adamw")
        lr = float(d.get("lr", 3e-4))
        wd = float(d.get("weight_decay", 0.01))

        max_grad_norm = float(d.get("max_grad_norm", 0))
        eps = float(d.get("eps", 1e-8))
        if opt_type in ("adam", "adamw"):
            opt_cls = AdamConfig if opt_type == "adam" else AdamWConfig
            optimizer = opt_cls(
                lr=lr,
                betas=(float(d.get("beta1", 0.90)), float(d.get("beta2", 0.95))),
                weight_decay=wd,
                max_grad_norm=max_grad_norm,
                eps=eps,
            )
        elif opt_type == "rmsprop":
            optimizer = RMSpropConfig(
                lr=lr,
                alpha=float(d.get("alpha", 0.99)),
                weight_decay=wd,
                momentum=float(d.get("momentum", 0)),
                max_grad_norm=max_grad_norm,
            )
        elif opt_type == "muon_hybrid":
            optimizer = MuonConfig(lr=lr, hybrid=True, weight_decay=wd, max_grad_norm=max_grad_norm)
        else:
            optimizer = MuonConfig(lr=lr, weight_decay=wd, max_grad_norm=max_grad_norm)

        sched_type = d.get("scheduler_type", "none")
        min_lr_ratio = float(d.get("min_lr_ratio", 0.1))
        scheduler = None
        if sched_type == "warmup_exp_decay":
            scheduler = WarmupExpDecay(
                warmup_steps=int(d.get("warmup_steps", 1000)),
                decay_steps=int(d.get("decay_steps", 50000)),
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "noam":
            scheduler = NoamScheduler(
                d_model=int(d.get("d_model", d.get("dim_embeddings", 256))),
                warmup_steps=int(d.get("warmup_steps", 1000)),
            )
        elif sched_type == "step_exp":
            scheduler = StepExponentialLR(
                decay_steps=int(d.get("decay_steps", 50000)),
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "cosine":
            scheduler = CosineWithWarmup(
                warmup_steps=int(d.get("warmup_steps", 1000)),
                decay_steps=int(d.get("decay_steps", 50000)),
                min_lr_ratio=min_lr_ratio,
            )
        elif sched_type == "trapezoidal":
            scheduler = TrapezoidalLR(
                warmup_steps=int(d.get("warmup_steps", 500)),
                flat_steps=int(d.get("flat_steps", 5000)),
                decay_steps=int(d.get("decay_steps", 2000)),
                min_lr_ratio=min_lr_ratio,
            )

        opts = TrainingOptions(
            batch_size=int(d.get("batch_size", 32)),
            accumulation_steps=int(d.get("accumulation_steps", 1)),
            training_timeout_minutes=min(int(d.get("training_minutes", timeout_minutes)), MAX_TRAINING_MINUTES),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            record_interval_steps=50,
            val_interval_steps=int(d.get("val_interval_steps", 250)),
            val_max_batches=int(d.get("val_max_batches", 50)),
            distributed_strategy=distributed_strategy,
        )
        if "rho_loss" in d and d["rho_loss"]:
            from training.rho_loss import RhoLossTagConfig, RhoLossEmaConfig, RhoLossSnapshotConfig
            rl = d["rho_loss"]
            mode = rl.get("mode", "ema")
            ratio = float(rl.get("ratio", 0.2))
            if mode == "tag":
                opts.rho_loss = RhoLossTagConfig(tag=str(rl["tag"]), ratio=ratio)
            elif mode == "ema":
                opts.rho_loss = RhoLossEmaConfig(ratio=ratio, decay=float(rl.get("decay", 0.999)), warmup_steps=int(rl.get("warmup_steps", 100)))
            elif mode == "snapshot":
                opts.rho_loss = RhoLossSnapshotConfig(ratio=ratio, snapshot_steps=int(rl.get("snapshot_steps", 500)), warmup_steps=int(rl.get("warmup_steps", 100)))
            else:
                raise ValueError(f"Unknown rho_loss mode: {mode}")
        if device is not None:
            opts.device = device
        return opts

    @staticmethod
    def _validate_and_repair(config):
        while config.dim_embeddings % config.num_attention_heads != 0:
            config.num_attention_heads //= 2
            if config.num_attention_heads < 1:
                config.num_attention_heads = 1
                break
        config.dropout = max(0.0, min(float(config.dropout), 0.5))
        config.num_transformer_layers = max(1, min(config.num_transformer_layers, 24))
        config.dim_embeddings = max(64, config.dim_embeddings)

        available_gb = _total_gpu_memory_gb()
        num_gpus = max(1, torch.cuda.device_count())
        while _estimate_model_gb(config, num_gpus) > available_gb and config.num_transformer_layers > 2:
            config.num_transformer_layers -= 1

        rope_layers = {
            TransformerLayerType.LLAMA2, TransformerLayerType.LLAMA3,
            TransformerLayerType.DEEPSEEK, TransformerLayerType.OLMO,
            TransformerLayerType.OLMO_HYPER_CONNECTIONS,
            TransformerLayerType.OLMO_CONSTRAINED_HYPER_CONNECTIONS,
            TransformerLayerType.OLMO_IDENTITY_HYPER_CONNECTIONS,
        }
        if config.transformer_layer in rope_layers:
            if config.normalization_layer == NormalizationLayerType.LAYER_NORM:
                config.normalization_layer = NormalizationLayerType.RMS_NORM
        return config


class StabilityMetric:
    @staticmethod
    def compute(results: Results) -> dict:
        if len(results.step_accuracy.min_max_avg) > 4:
            _, _, avg = results.step_accuracy.get_arrays()
        elif len(results.accuracy.min_max_avg) > 0:
            _, _, avg = results.accuracy.get_arrays()
        else:
            return {
                "final_accuracy": 0.0,
                "final_loss": float("inf"),
                "perplexity": float("inf"),
                "stability_score": 0.0,
                "raw_variance": float("inf"),
                "accuracy_slope": 0.0,
                "steps_to_threshold": -1,
            }

        tail = avg[-max(1, int(len(avg) * STABILITY_TAIL_FRACTION)) :]
        final_accuracy = statistics.mean(tail)
        variance = statistics.variance(tail) if len(tail) > 1 else 0.0
        stability_score = 1.0 / (1.0 + variance**0.5)

        # Slope of accuracy over the tail (per step) — positive means still learning
        n = len(tail)
        if n > 1:
            x_mean = (n - 1) / 2.0
            y_mean = statistics.mean(tail)
            num = sum((i - x_mean) * (tail[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            accuracy_slope = num / den if den > 0 else 0.0
        else:
            accuracy_slope = 0.0

        # Steps to reach the accuracy threshold
        steps_to_threshold = -1
        for i, acc in enumerate(avg):
            if acc >= STEPS_TO_ACCURACY_THRESHOLD:
                steps_to_threshold = i
                break

        # Loss / perplexity
        if len(results.step_loss.min_max_avg) > 4:
            _, _, loss_avg = results.step_loss.get_arrays()
        elif len(results.loss.min_max_avg) > 0:
            _, _, loss_avg = results.loss.get_arrays()
        else:
            loss_avg = []

        if loss_avg:
            loss_tail = loss_avg[
                -max(1, int(len(loss_avg) * STABILITY_TAIL_FRACTION)) :
            ]
            final_loss = statistics.mean(loss_tail)
            perplexity = math.exp(min(final_loss, 20))  # cap to avoid overflow
        else:
            final_loss = float("inf")
            perplexity = float("inf")

        return {
            "final_accuracy": round(final_accuracy, 4),
            "final_loss": round(final_loss, 4)
            if math.isfinite(final_loss)
            else final_loss,
            "perplexity": round(perplexity, 2)
            if math.isfinite(perplexity)
            else perplexity,
            "stability_score": round(stability_score, 4),
            "raw_variance": round(variance, 6),
            "accuracy_slope": round(accuracy_slope, 6),
            "steps_to_threshold": steps_to_threshold,
        }


@dataclass
class ExperimentRecord:
    experiment_id: int
    name: str
    model_config: dict
    training_config: dict
    score: dict
    status: str
    error_message: Optional[str]
    timestamp_start: str
    timestamp_end: Optional[str]
    llm_reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        return cls(**d)


class AutoparamState:
    def __init__(self, state_path: str):
        self.state_path = state_path
        self.experiments: List[ExperimentRecord] = []
        self.best_experiment_id: Optional[int] = None
        self.best_score: float = -1.0
        self.session_start: str = datetime.now().isoformat()
        self._load_if_exists()

    def _load_if_exists(self):
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path) as f:
            data = json.load(f)
        self.experiments = [ExperimentRecord.from_dict(e) for e in data["experiments"]]
        self.best_experiment_id = data.get("best_experiment_id")
        self.best_score = data.get("best_score", -1.0)
        self.session_start = data.get("session_start", self.session_start)
        print(
            f"[autoparam] Resumed: {len(self.experiments)} previous experiments loaded."
        )

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "experiments": [e.to_dict() for e in self.experiments],
                    "best_experiment_id": self.best_experiment_id,
                    "best_score": self.best_score,
                    "session_start": self.session_start,
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        os.replace(tmp, self.state_path)

    def add_experiment(self, record: ExperimentRecord):
        self.experiments.append(record)
        if record.status == "success":
            acc = record.score.get("final_accuracy", -1.0)
            slope = record.score.get("accuracy_slope", 0.0)
            score = acc + 0.5 * max(0.0, slope * 500)
            if score > self.best_score:
                self.best_score = score
                self.best_experiment_id = record.experiment_id
        self.save()

    @property
    def best_record(self) -> Optional[ExperimentRecord]:
        if self.best_experiment_id is None:
            return None
        return next(
            (e for e in self.experiments if e.experiment_id == self.best_experiment_id),
            None,
        )

    def successful_experiments(self) -> List[ExperimentRecord]:
        return [e for e in self.experiments if e.status == "success"]

    def recent_experiments(self, n: int = HISTORY_WINDOW) -> List[ExperimentRecord]:
        return self.experiments[-n:]


RANDOM_EXPLORE_EVERY = 5  # force a random config every N experiments
MUTATE_PROB = 0.5  # within non-LLM route, probability to mutate a top run vs. fresh random
MUTATE_TOP_K = 5


def random_config_dict() -> dict:
    import random

    dim = random.choice([128, 256, 384, 512, 768, 1024])
    heads = random.choice([h for h in [4, 8, 12, 16, 32] if dim % h == 0])
    return {
        "reasoning": "forced random exploration",
        "dim_embeddings": dim,
        "num_attention_heads": heads,
        "num_transformer_layers": random.choice([2, 4, 6, 8, 12, 16, 24]),
        "dropout": random.choice([0.0, 0.05, 0.1, 0.2]),
        "feed_forward_layer": random.choice([512, 1024, 2048, 4096, 8192]),
        "bias": random.choice([True, False]),
        "hc_n": random.choice([2, 4, 8]),
        "transformer_layer": random.choice(
            [
                "SIMPLE",
                "GPT2",
                "LLAMA2",
                "LLAMA3",
                "DEEPSEEK",
                "OLMO",
                "OLMO_HYPER_CONNECTIONS",
                "OLMO_CONSTRAINED_HYPER_CONNECTIONS",
                "OLMO_IDENTITY_HYPER_CONNECTIONS",
                "SIMPLE_ATTENTION_AT_HOME",
            ]
        ),
        "positional_embedding": random.choice(
            [
                "NN_EMBEDDING",
                "SINUSOIDAL",
                "ROTARY_POSITION_ENCODING",
                "NONE",
            ]
        ),
        "normalization_layer": random.choice(["LAYER_NORM", "DyT", "RMS_NORM"]),
        "attention_type": random.choice(["DEFAULT", "MHA", "GQA", "MLA"]),
        "qk_norm": random.choice([True, False]),
        "ffn_activation": random.choice(["SWIGLU", "GEGLU", "REGLU", "SILU"]),
        "norm_placement": random.choice(["PRE", "POST", "SANDWICH", "PERI"]),
        "label_smoothing": random.choice([0.0, 0.05, 0.1]),
        "tie_embeddings": random.choice([True, False]),
        "init_std": random.choice([0.006, 0.01, 0.02, 0.04]),
        "optimizer_type": random.choice(["adam", "adamw", "rmsprop", "muon", "muon_hybrid"]),
        "lr": random.choice([0.0001, 0.0003, 0.001, 0.002]),
        "beta1": random.choice([0.85, 0.9, 0.95]),
        "beta2": random.choice([0.9, 0.95, 0.999]),
        "weight_decay": random.choice([0, 0.01, 0.1]),
        "max_grad_norm": random.choice([0, 0.5, 1.0, 5.0]),
        "alpha": random.choice([0.9, 0.95, 0.99]),
        "momentum": random.choice([0, 0.1, 0.9]),
        "scheduler_type": random.choice(
            ["none", "noam", "warmup_exp_decay", "step_exp", "cosine", "trapezoidal"]
        ),
        "warmup_steps": random.choice([500, 1000, 2000, 4000]),
        "flat_steps": random.choice([1000, 5000, 10000, 20000]),
        "decay_steps": random.choice([10000, 50000, 100000]),
        "min_lr_ratio": random.choice([0.01, 0.05, 0.1]),
        "batch_size": random.choice([16, 32, 64, 128]),
        "accumulation_steps": random.choice([1, 2, 4, 8]),
        "training_minutes": random.choice([5, 15, 30, 60, 120, 240, 480]),
    }


def mutate_config_dict(parent: dict) -> dict:
    """Mutate a known-good config: tweak 1-3 knobs to nearby values."""
    import random

    cfg = dict(parent)
    cfg.pop("reasoning", None)

    nearby = {
        "dim_embeddings": [128, 256, 384, 512, 768, 1024],
        "num_attention_heads": [4, 6, 8, 12, 16, 24, 32],
        "num_transformer_layers": [2, 4, 6, 8, 12, 16, 24],
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "feed_forward_layer": [512, 1024, 2048, 4096, 8192],
        "warmup_steps": [100, 250, 500, 1000, 2000, 4000],
        "decay_steps": [10000, 50000, 100000, 200000],
        "min_lr_ratio": [0.01, 0.05, 0.1],
        "batch_size": [16, 32, 64, 128],
        "accumulation_steps": [1, 2, 4, 8, 16],
        "training_minutes": [5, 15, 30, 60, 120, 240, 480],
        "weight_decay": [0, 0.01, 0.05, 0.1, 0.2],
        "max_grad_norm": [0, 0.5, 1.0, 5.0],
        "label_smoothing": [0.0, 0.05, 0.1],
        "init_std": [0.006, 0.01, 0.02, 0.04],
    }

    def step_along(key):
        if key not in cfg or key not in nearby:
            return
        vals = nearby[key]
        try:
            i = vals.index(cfg[key])
        except ValueError:
            cfg[key] = random.choice(vals)
            return
        delta = random.choice([-1, 1])
        cfg[key] = vals[max(0, min(len(vals) - 1, i + delta))]

    mutation_pool = [
        ("scale_lr", lambda: cfg.update(lr=max(1e-5, min(0.01, cfg.get("lr", 1e-3) * random.choice([0.5, 0.7, 1.5, 2.0]))))),
        ("longer_train", lambda: cfg.update(training_minutes=min(480, int(cfg.get("training_minutes", 15) * 2)))),
        ("step_dim", lambda: step_along("dim_embeddings")),
        ("step_layers", lambda: step_along("num_transformer_layers")),
        ("step_ffn", lambda: step_along("feed_forward_layer")),
        ("step_dropout", lambda: step_along("dropout")),
        ("step_warmup", lambda: step_along("warmup_steps")),
        ("step_decay", lambda: step_along("decay_steps")),
        ("step_batch", lambda: step_along("batch_size")),
        ("step_accum", lambda: step_along("accumulation_steps")),
        ("step_wd", lambda: step_along("weight_decay")),
        ("step_grad_clip", lambda: step_along("max_grad_norm")),
        ("step_label_smooth", lambda: step_along("label_smoothing")),
        ("swap_scheduler", lambda: cfg.update(scheduler_type=random.choice(
            ["noam", "warmup_exp_decay", "step_exp", "cosine", "trapezoidal"]))),
        ("toggle_qk_norm", lambda: cfg.update(qk_norm=not cfg.get("qk_norm", False))),
        ("toggle_tie_emb", lambda: cfg.update(tie_embeddings=not cfg.get("tie_embeddings", False))),
        ("swap_ffn_act", lambda: cfg.update(ffn_activation=random.choice(["SWIGLU", "GEGLU", "REGLU", "SILU"]))),
        ("swap_norm_placement", lambda: cfg.update(norm_placement=random.choice(["PRE", "POST", "SANDWICH", "PERI"]))),
    ]

    n_mutations = random.randint(1, 3)
    chosen = random.sample(mutation_pool, k=min(n_mutations, len(mutation_pool)))
    applied = []
    for name, fn in chosen:
        fn()
        applied.append(name)

    if cfg.get("dim_embeddings") and cfg.get("num_attention_heads"):
        if cfg["dim_embeddings"] % cfg["num_attention_heads"] != 0:
            valid = [h for h in [4, 6, 8, 12, 16, 24, 32] if cfg["dim_embeddings"] % h == 0]
            cfg["num_attention_heads"] = random.choice(valid) if valid else 4

    cfg["reasoning"] = f"mutated parent: {', '.join(applied)}"
    return cfg


def _pick_mutation_parent(state) -> Optional[dict]:
    import random

    successes = [e for e in state.experiments if e.status == "success"]
    if not successes:
        return None
    successes.sort(key=lambda e: e.score.get("final_accuracy", 0.0), reverse=True)
    pool = successes[:MUTATE_TOP_K]
    parent = random.choice(pool)
    return {**parent.model_config, **parent.training_config}


TRAINING_TIME_LADDER = [15, 60, 120, 240, 480]


def _config_signature_no_time(model_cfg: dict, training_cfg: dict) -> str:
    t = {k: v for k, v in training_cfg.items() if k != "training_minutes"}
    return json.dumps([model_cfg, t], sort_keys=True)


def _pick_extension_candidate(state) -> Optional[dict]:
    """Find a top-K success whose config hasn't been tried at the next rung of TRAINING_TIME_LADDER."""
    successes = [e for e in state.experiments if e.status == "success"]
    if not successes:
        return None
    successes.sort(key=lambda e: e.score.get("final_accuracy", 0.0), reverse=True)
    pool = successes[:MUTATE_TOP_K]

    max_minutes_by_sig: dict = {}
    for e in state.experiments:
        sig = _config_signature_no_time(e.model_config, e.training_config)
        m = e.training_config.get("training_minutes", 0) or 0
        if m > max_minutes_by_sig.get(sig, -1):
            max_minutes_by_sig[sig] = m

    for parent in pool:
        sig = _config_signature_no_time(parent.model_config, parent.training_config)
        cur_max = max_minutes_by_sig.get(sig, 0)
        next_rung = next((t for t in TRAINING_TIME_LADDER if t > cur_max), None)
        if next_rung is None:
            continue
        cfg = {**parent.model_config, **parent.training_config}
        cfg["training_minutes"] = next_rung
        cfg["reasoning"] = f"extend top run #{parent.experiment_id} ({cur_max}min -> {next_rung}min)"
        return cfg
    return None


def _promote_best_tag(dataset_name: str, source_tag: str) -> None:
    from utils.checkpoints import StorageBox
    storage = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )
    src = os.path.join("checkpoints", "tags", source_tag, "latest.json")
    dst = os.path.join("checkpoints", "tags", f"best-{dataset_name}", "latest.json")
    storage.save_bytes(storage.load_bytes(src, use_cache=False), dst)


def _gpus_are_stuck() -> bool:
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        stuck = False
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            if util.gpu < 50:
                continue
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if not procs:
                stuck = True
                break
        pynvml.nvmlShutdown()
        return stuck
    except Exception:
        return False


def fetch_openrouter_daily_usage() -> float:
    """Return today's USD spend for the current OpenRouter API key, or -1 on failure."""
    import urllib.request

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    req = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/auth/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
            return float(data.get("usage_daily", 0.0))
    except Exception:
        return -1.0


class LLMProposer:
    def __init__(self, model: str = LLM_MODEL):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"]
        )
        self.model = model

    def propose(self, state: AutoparamState, baseline_dict: dict) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_message(state, baseline_dict),
                },
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=1.2,
            response_format={"type": "json_object"},
        )
        return self._parse_json(response.choices[0].message.content)

    @staticmethod
    def _build_user_message(state: AutoparamState, baseline_dict: dict) -> str:
        recent = state.recent_experiments()
        history_lines = []
        for exp in recent:
            if exp.status == "success":
                s = exp.score
                steps = s.get("steps_to_threshold", -1)
                steps_str = f"{steps}" if steps >= 0 else "never"
                history_lines.append(
                    f"  #{exp.experiment_id} [ok] acc={s.get('final_accuracy', 0):.2f}% "
                    f"ppl={s.get('perplexity', 0):.1f} "
                    f"slope={s.get('accuracy_slope', 0):.4f} "
                    f"steps_to_{int(STEPS_TO_ACCURACY_THRESHOLD)}pct={steps_str} "
                    f"stability={s.get('stability_score', 0):.3f} | "
                    f"model={json.dumps(exp.model_config)} | "
                    f"training={json.dumps(exp.training_config)}"
                )
            else:
                history_lines.append(
                    f"  #{exp.experiment_id} [FAILED] {exp.error_message or 'unknown'} | "
                    f"model={json.dumps(exp.model_config)}"
                )

        best = state.best_record
        best_section = "No successful experiments yet — explore freely."
        best_acc = 0.0
        if best:
            bs = best.score
            best_acc = bs.get("final_accuracy", 0.0)
            best_section = (
                f"Best: #{best.experiment_id} acc={bs.get('final_accuracy', 0):.2f}%  "
                f"ppl={bs.get('perplexity', 0):.1f}  "
                f"slope={bs.get('accuracy_slope', 0):.4f}  "
                f"stability={bs.get('stability_score', 0):.3f}\n"
                f"  model={json.dumps(best.model_config)}\n"
                f"  training={json.dumps(best.training_config)}"
            )

        target_acc = max(30.0, (int(best_acc / 5) + 1) * 5)

        successful = state.successful_experiments()
        recent_successful = successful[-10:]
        stagnation_note = ""
        if len(recent_successful) >= 6:
            accs = [e.score.get("final_accuracy", 0.0) for e in recent_successful]
            top = max(accs)
            within = sum(1 for a in accs if top - a <= 1.0)
            if within >= 5:
                stagnation_note = (
                    f"\n\n## STAGNATION DETECTED\n"
                    f"{within} of last {len(accs)} successful runs are within 1% of {top:.2f}%. "
                    f"The search is stuck. Propose something FUNDAMENTALLY different: "
                    f"try a much larger or smaller scale, a different optimizer family, "
                    f"a different attention type, or a different schedule. "
                    f"Do NOT submit another minor variation of the current best."
                )

        if best_acc < 35.0:
            task_section = (
                f"Accuracy is currently near {best_acc:.1f}%. The goal is to break past {target_acc:.0f}%. "
                f"Prioritize configurations likely to exceed {target_acc:.0f}%: larger models (dim>=512, layers>=8), "
                f"muon_hybrid optimizer (Muon for hidden layers + AdamW for embeddings — strongest option), "
                f"LLAMA3/DEEPSEEK architectures with ROTARY_POSITION_ENCODING, and "
                f"trapezoidal scheduler (warmup→flat→decay). Avoid incremental tweaks to configs "
                f"already stuck near {best_acc:.0f}%. Reason about what fundamentally changes the learning dynamics."
            )
        else:
            task_section = (
                f"Current best is {best_acc:.2f}%. Goal: push past {target_acc:.0f}%. "
                f"You are no longer in the early-discovery phase — do NOT anchor to any single architecture, "
                f"optimizer, or schedule family. Consider: scaling up (more dims/layers/FFN, longer training), "
                f"scaling down with better hyperparameters, alternative attention types, alternative optimizers, "
                f"alternative position encodings, alternative normalizations. "
                f"Reason about what concrete bottleneck is preventing further gains and target it directly."
            )

        return f"""## Baseline configuration
{json.dumps(baseline_dict, indent=2)}

## Experiment history (last {len(recent)})
{chr(10).join(history_lines) if history_lines else "  (none yet)"}

## Current best
{best_section}{stagnation_note}

## Task
{task_section}

Respond with JSON matching this schema exactly:
{{
  "reasoning": "<your explanation>",
  "dim_embeddings": <int>, "num_attention_heads": <int>, "num_transformer_layers": <int>,
  "dropout": <float>, "feed_forward_layer": <int>, "bias": <bool>, "hc_n": <int>,
  "transformer_layer": "<string>", "positional_embedding": "<string>", "normalization_layer": "<string>", "attention_type": "<string>", "qk_norm": <bool>, "ffn_activation": "<string>", "norm_placement": "<string>", "label_smoothing": <float>, "tie_embeddings": <bool>, "init_std": <float>,
  "optimizer_type": "<string>", "lr": <float>, "beta1": <float>, "beta2": <float>,
  "weight_decay": <float>, "max_grad_norm": <float>, "alpha": <float>, "momentum": <float>,
  "scheduler_type": "<string>", "warmup_steps": <int>, "flat_steps": <int>, "decay_steps": <int>, "min_lr_ratio": <float>,
  "batch_size": <int>, "accumulation_steps": <int>, "training_minutes": <int>
}}"""

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse LLM response as JSON: {text[:300]}")


def plot_progress(state: AutoparamState, output_path: str):
    successes = [e for e in state.experiments if e.status == "success"]
    if not successes:
        return

    id_to_pos = {e.experiment_id: i for i, e in enumerate(successes)}
    ids = list(range(len(successes)))
    accuracy = [e.score["final_accuracy"] for e in successes]
    loss_vals = [
        e.score["final_loss"]
        for e in successes
        if math.isfinite(e.score.get("final_loss", float("inf")))
    ]
    loss_ids = [
        id_to_pos[e.experiment_id]
        for e in successes
        if math.isfinite(e.score.get("final_loss", float("inf")))
    ]

    def _elapsed_min(e):
        try:
            t0 = datetime.fromisoformat(e.timestamp_start)
            t1 = datetime.fromisoformat(e.timestamp_end) if e.timestamp_end else None
            if t1 is None:
                return None
            return (t1 - t0).total_seconds() / 60.0
        except Exception:
            return None

    best = state.best_record

    val_loss_pts = [
        (id_to_pos[e.experiment_id], e.score["val_loss"])
        for e in successes
        if math.isfinite(e.score.get("val_loss", float("inf")))
    ]
    val_acc_pts = [
        (id_to_pos[e.experiment_id], e.score["val_accuracy"])
        for e in successes
        if "val_accuracy" in e.score
    ]
    params_pts = [
        (e.score.get("params_count", 0) / 1e6, e.score["final_accuracy"], id_to_pos[e.experiment_id])
        for e in successes
        if e.score.get("params_count", 0) > 0
    ]
    time_pts = [
        (_elapsed_min(e), e.score["final_accuracy"], id_to_pos[e.experiment_id])
        for e in successes
    ]
    time_pts = [p for p in time_pts if p[0] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_acc, ax_val, ax_params, ax_time = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax_acc.plot(ids, accuracy, "b-o", label="Train Accuracy (%)", markersize=5)
    if loss_ids:
        ax_acc2 = ax_acc.twinx()
        ax_acc2.plot(loss_ids, loss_vals, "r--", marker="^", label="Train Loss",
                     markersize=4, alpha=0.6)
        ax_acc2.set_ylabel("Loss", color="red")
        ax_acc2.tick_params(axis="y", labelcolor="red")
        ax_acc2.legend(loc="upper right")
    if best and best.experiment_id in id_to_pos:
        ax_acc.axvline(x=id_to_pos[best.experiment_id], color="red", linestyle=":", alpha=0.6,
                       label=f"Best (#{best.experiment_id})")
    ax_acc.set_xlabel("Successful experiment #")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Train accuracy & loss over experiments")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    if val_acc_pts:
        xs, ys = zip(*val_acc_pts)
        ax_val.plot(xs, ys, "g-o", label="Val Accuracy (%)", markersize=5)
    if val_loss_pts:
        ax_val2 = ax_val.twinx()
        xs, ys = zip(*val_loss_pts)
        ax_val2.plot(xs, ys, "m--", marker="^", label="Val Loss",
                     markersize=4, alpha=0.6)
        ax_val2.set_ylabel("Val Loss", color="magenta")
        ax_val2.tick_params(axis="y", labelcolor="magenta")
        ax_val2.legend(loc="upper right")
    ax_val.set_xlabel("Successful experiment #")
    ax_val.set_ylabel("Val Accuracy (%)")
    ax_val.set_title("Validation curves over experiments")
    ax_val.legend(loc="lower right")
    ax_val.grid(True, alpha=0.3)

    if params_pts:
        xs = [p[0] for p in params_pts]
        ys = [p[1] for p in params_pts]
        cs = [p[2] for p in params_pts]
        sc_p = ax_params.scatter(xs, ys, c=cs, cmap="viridis", s=40, alpha=0.8)
        plt.colorbar(sc_p, ax=ax_params, label="Experiment #")
        top_idx = max(range(len(params_pts)), key=lambda i: params_pts[i][1])
        tx, ty, tid = params_pts[top_idx]
        ax_params.scatter([tx], [ty], s=160, facecolors="none", edgecolors="red",
                          linewidths=2, label=f"Top accuracy (#{tid})")
        ax_params.legend(loc="lower right")
        ax_params.set_xscale("log")
    ax_params.set_xlabel("Params (M, log)")
    ax_params.set_ylabel("Accuracy (%)")
    ax_params.set_title("Accuracy vs. parameter count")
    ax_params.grid(True, alpha=0.3, which="both")

    if time_pts:
        xs = [p[0] for p in time_pts]
        ys = [p[1] for p in time_pts]
        cs = [p[2] for p in time_pts]
        sc_t = ax_time.scatter(xs, ys, c=cs, cmap="viridis", s=40, alpha=0.8)
        plt.colorbar(sc_t, ax=ax_time, label="Experiment #")
        top_idx = max(range(len(time_pts)), key=lambda i: time_pts[i][1])
        tx, ty, tid = time_pts[top_idx]
        ax_time.scatter([tx], [ty], s=160, facecolors="none", edgecolors="red",
                        linewidths=2, label=f"Top accuracy (#{tid})")
        if best is not None and best.experiment_id in id_to_pos:
            be = _elapsed_min(best)
            if be is not None:
                ax_time.scatter([be], [best.score["final_accuracy"]],
                                s=160, facecolors="none", edgecolors="orange",
                                linewidths=2, linestyle="--",
                                label=f"Best composite (#{id_to_pos[best.experiment_id]})")
        ax_time.legend(loc="lower right")
    ax_time.set_xlabel("Training time (min)")
    ax_time.set_ylabel("Accuracy (%)")
    ax_time.set_title("Accuracy vs. training time")
    ax_time.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close("all")
    print(f"[autoparam] Progress plot saved: {output_path}")


class AutoparamLoop:
    def __init__(
        self,
        dataset_name: str,
        max_experiments: int = 40,
        experiment_timeout_minutes: int = 40,
        state_path: Optional[str] = None,
        llm_model: str = LLM_MODEL,
        budget_usd: Optional[float] = None,
        distributed_strategy: DistributedStrategy = DistributedStrategy.FSDP,
        nproc_per_node: int = 1,
        max_consecutive_failures: int = 5,
        random_only: bool = False,
    ):
        if state_path is None:
            state_path = f"autoparam_state_{dataset_name}.json"
        self.max_experiments = max_experiments
        self.budget_usd = budget_usd
        self.distributed_strategy = distributed_strategy
        self.nproc_per_node = nproc_per_node
        self.max_consecutive_failures = max_consecutive_failures
        self.log_path = state_path.replace(".json", ".log")
        self.timeout = experiment_timeout_minutes
        self.state = AutoparamState(state_path)
        self._llm_disabled = random_only
        self.proposer = None if random_only else LLMProposer(model=llm_model)
        self.dataset = NAMED_DATASETS[dataset_name]
        self.plot_path = os.path.join("plots", dataset_name, "autoparam_progress.png")
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        self._active_proc = None
        self._active_pgid = None
        import atexit
        atexit.register(self._kill_active_proc)
        signal.signal(signal.SIGTERM, self._signal_handler)

        baseline_config = create_default_config(self.dataset)
        baseline_opts = TrainingOptions(
            batch_size=1,
            training_timeout_minutes=experiment_timeout_minutes,
        )
        self.baseline_dict = {
            **ConfigSerializer.config_to_dict(baseline_config),
            **ConfigSerializer.training_options_to_dict(baseline_opts),
        }

    def _kill_active_proc(self):
        pgid = self._active_pgid
        if pgid is None:
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    def _signal_handler(self, signum, frame):
        self._kill_active_proc()
        sys.exit(1)

    @staticmethod
    def _config_hash(model_dict: dict, training_dict: dict) -> str:
        return hashlib.md5(
            json.dumps({**model_dict, **training_dict}, sort_keys=True).encode()
        ).hexdigest()

    def _already_run(self, model_dict: dict, training_dict: dict) -> bool:
        h = self._config_hash(model_dict, training_dict)
        return any(
            self._config_hash(e.model_config, e.training_config) == h
            for e in self.state.experiments
        )

    def run(self):
        start_id = len(self.state.experiments)
        budget_msg = f"  Budget: ${self.budget_usd:.2f}" if self.budget_usd else ""
        print(
            f"[autoparam] Starting from experiment {start_id}. Target: {self.max_experiments}. Timeout: {self.timeout}min each.{budget_msg}"
        )
        if self.budget_usd and not self._llm_disabled:
            self._daily_spend_at_start = fetch_openrouter_daily_usage()
            if self._daily_spend_at_start >= 0:
                self._log(
                    f"OpenRouter daily spend at start: ${self._daily_spend_at_start:.4f}"
                )
        consecutive_failures = 0

        for exp_id in range(start_id, self.max_experiments):
            if _gpus_are_stuck():
                self._log("ERROR: GPUs show high utilization but no running processes — likely a zombie GPU context. Reboot required. Aborting.")
                break
            self._log(f"=== Experiment {exp_id + 1}/{self.max_experiments} ===")

            if self.budget_usd and not self._llm_disabled:
                daily = fetch_openrouter_daily_usage()
                if daily >= 0:
                    spent = daily - self._daily_spend_at_start
                    self._log(
                        f"OpenRouter spend this session: ${spent:.4f} / ${self.budget_usd:.2f}"
                    )
                    if spent >= self.budget_usd:
                        self._log(
                            f"Budget ${self.budget_usd:.2f} reached (${spent:.4f} spent). Stopping."
                        )
                        break

            import random as _random
            MAX_DEDUP_ATTEMPTS = 20
            proposed = None
            reasoning = None
            model_dict = None
            training_dict = None
            config = None
            training_options = None
            config_error = None
            is_extension = False

            for attempt in range(MAX_DEDUP_ATTEMPTS):
                is_extension = False
                extension = _pick_extension_candidate(self.state) if attempt == 0 else None
                if extension is not None:
                    cand = extension
                    cand_reason = cand.pop("reasoning")
                    is_extension = True
                    src = f"Extending top run: {cand_reason}"
                elif self._llm_disabled or exp_id % RANDOM_EXPLORE_EVERY == 0:
                    parent = _pick_mutation_parent(self.state)
                    if parent is not None and _random.random() < MUTATE_PROB:
                        cand = mutate_config_dict(parent)
                        cand_reason = cand.pop("reasoning")
                        src = f"Mutated top run: {cand_reason}"
                    else:
                        cand = random_config_dict()
                        cand_reason = cand.pop("reasoning")
                        src = (
                            "Random exploration (LLM disabled)"
                            if self._llm_disabled
                            else f"Random exploration (every {RANDOM_EXPLORE_EVERY} experiments)"
                        )
                else:
                    try:
                        cand = self.proposer.propose(self.state, self.baseline_dict)
                        cand_reason = cand.pop("reasoning", "(no reasoning provided)")
                        src = f"Reasoning: {cand_reason}"
                    except Exception as e:
                        msg = str(e)
                        if "402" in msg or "insufficient" in msg.lower() or "credit" in msg.lower():
                            self._llm_disabled = True
                            self._log(f"LLM credits exhausted ({e}). Switching to random-only mode.")
                        else:
                            self._log(f"LLM proposal failed ({e}), using random fallback.")
                        cand = random_config_dict()
                        cand_reason = cand.pop("reasoning", f"LLM failed: {e}")
                        src = f"Reasoning: {cand_reason}"

                try:
                    cfg_obj = ConfigSerializer.dict_to_config(cand, self.dataset)
                    to_obj = ConfigSerializer.dict_to_training_options(cand, self.timeout)
                    m_dict = ConfigSerializer.config_to_dict(cfg_obj)
                    t_dict = ConfigSerializer.training_options_to_dict(to_obj)
                except Exception as e:
                    config_error = e
                    proposed = cand
                    reasoning = cand_reason
                    break

                if self._already_run(m_dict, t_dict):
                    self._log(f"Duplicate config (attempt {attempt + 1}/{MAX_DEDUP_ATTEMPTS}), retrying.")
                    continue

                proposed = cand
                reasoning = cand_reason
                config = cfg_obj
                training_options = to_obj
                model_dict = m_dict
                training_dict = t_dict
                config_error = None
                self._log(src)
                break
            else:
                self._log(f"Could not find non-duplicate config after {MAX_DEDUP_ATTEMPTS} attempts; skipping experiment.")
                continue

            if config_error is not None:
                import traceback
                self._log(f"Config error: {config_error}\n{traceback.format_exc()}")
                self._record(
                    exp_id,
                    proposed,
                    proposed,
                    reasoning,
                    "failed",
                    f"Config error: {config_error}",
                )
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._log(f"Stopping early: {consecutive_failures} consecutive failures.")
                    return
                continue

            try:
                num_gpus = max(1, self.nproc_per_node)
                bs = getattr(training_options, "batch_size", 1) or 1
                est_gb, n_params = _estimate_footprint_gb_meta(config, bs, num_gpus)
                budget_gb = _total_gpu_memory_gb()
                if est_gb > budget_gb:
                    self._log(
                        f"Skipping: estimated {est_gb:.2f} GB > budget {budget_gb:.2f} GB "
                        f"(params={n_params/1e6:.1f}M, bs={bs}, gpus={num_gpus})"
                    )
                    self._record(
                        exp_id,
                        model_dict,
                        training_dict,
                        reasoning,
                        "failed",
                        f"Model too large: estimated {est_gb:.2f} GB > {budget_gb:.2f} GB budget",
                    )
                    consecutive_failures += 1
                    if consecutive_failures >= self.max_consecutive_failures:
                        self._log(f"Stopping early: {consecutive_failures} consecutive failures.")
                        return
                    continue
            except Exception as e:
                self._log(f"Meta-device size estimate failed (continuing anyway): {e}")

            self._log(
                f"Config: {json.dumps(model_dict)}  training: {json.dumps(training_dict)}"
            )

            exp_name = f"autoparam_{exp_id:03d}"
            timestamp_start = datetime.now().isoformat()
            score, status, error_message = {}, "failed", None

            config_data = {
                "dataset_name": self.dataset.name,
                "exp_name": exp_name,
                "timeout_minutes": training_options.training_timeout_minutes,
                "model_config": model_dict,
                "training_config": training_dict,
                "distributed_strategy": self.distributed_strategy.name,
                "is_extension": is_extension,
                "checkpoint_tag": f"autoparam-{self.dataset.name}-{exp_name}" if is_extension else None,
            }
            config_path = None
            result_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, dir="/tmp"
                ) as f:
                    json.dump(config_data, f)
                    config_path = f.name
                result_path = config_path.replace(".json", "_result.json")

                executor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoparam_executor.py")
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={self.nproc_per_node}",
                    "--standalone",
                    "--max-restarts=0",
                    "--monitor-interval=5",
                    executor,
                    "--config", config_path,
                    "--result", result_path,
                ]
                log_path = result_path.replace("_result.json", "_run.log")
                log_file = open(log_path, "w")
                print(f"[autoparam] subprocess log: {log_path}", flush=True)
                env = os.environ.copy()
                env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
                env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
                env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
                env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True, env=env)
                log_file.close()
                self._active_proc = proc
                pgid = os.getpgid(proc.pid)
                self._active_pgid = pgid
                try:
                    proc.wait(timeout=(training_options.training_timeout_minutes + 5) * 60)
                except subprocess.TimeoutExpired:
                    print(f"Experiment subprocess timed out after {training_options.training_timeout_minutes + 5} minutes, killing")
                except KeyboardInterrupt:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                    raise
                finally:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    self._active_proc = None
                    self._active_pgid = None
                    time.sleep(15)

                if os.path.exists(result_path):
                    with open(result_path) as f:
                        result_data = json.load(f)
                    score = result_data.get("score", {})
                    status = result_data.get("status", "failed")
                    error_message = result_data.get("error_message")
                else:
                    status = "failed"
                    error_message = f"Executor exited with code {proc.returncode}, no result written"

                if status == "failed":
                    self._log(f"Training failed: {error_message}")
                    consecutive_failures += 1
                else:
                    steps = score.get("steps_to_threshold", -1)
                    params_m = score.get("params_count", 0) / 1e6
                    pmem = score.get("peak_memory_gb", 0)
                    tokens = score.get("tokens_seen", 0)
                    self._log(
                        f"Result: accuracy={score['final_accuracy']:.2f}%  "
                        f"ppl={score.get('perplexity', 0):.1f}  "
                        f"slope={score.get('accuracy_slope', 0):.4f}  "
                        f"steps_to_{int(STEPS_TO_ACCURACY_THRESHOLD)}pct={'never' if steps < 0 else steps}  "
                        f"stability={score['stability_score']:.3f}  "
                        f"params={params_m:.1f}M  peak_mem={pmem:.2f}GB  tokens={tokens:,}  "
                        f"val_acc={score.get('val_accuracy', float('nan')):.2f}%  "
                        f"val_loss={score.get('val_loss', float('nan')):.3f}  "
                        f"gap={score.get('overfit_gap', float('nan')):.2f}"
                    )
            except Exception as e:
                import traceback
                error_message = str(e)
                self._log(f"Failed to launch executor: {e}\n{traceback.format_exc()}")
                consecutive_failures += 1
            finally:
                for p in [config_path, result_path]:
                    if p and os.path.exists(p):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

            if status == "success":
                consecutive_failures = 0

            if consecutive_failures >= self.max_consecutive_failures:
                self._log(
                    f"Stopping early: {consecutive_failures} consecutive failures."
                )
                break

            self.state.add_experiment(
                ExperimentRecord(
                    experiment_id=exp_id,
                    name=exp_name,
                    model_config=model_dict,
                    training_config=training_dict,
                    score=score,
                    status=status,
                    error_message=error_message,
                    timestamp_start=timestamp_start,
                    timestamp_end=datetime.now().isoformat(),
                    llm_reasoning=reasoning,
                )
            )
            if status == "success" and is_extension and self.state.best_experiment_id == exp_id:
                try:
                    _promote_best_tag(self.dataset.name, f"autoparam-{self.dataset.name}-{exp_name}")
                    self._log(f"Promoted #{exp_id} to best-{self.dataset.name} tag")
                except Exception as e:
                    self._log(f"Failed to promote best tag: {e}")
            plot_progress(self.state, self.plot_path)

            best = self.state.best_record
            if best:
                self._log(
                    f"Best so far: #{best.experiment_id}  "
                    f"acc={best.score.get('final_accuracy', 0):.2f}%  "
                    f"ppl={best.score.get('perplexity', 0):.1f}  "
                    f"slope={best.score.get('accuracy_slope', 0):.4f}"
                )

        self._print_summary()

    def _log(self, text: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        print(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _record(self, exp_id, model_dict, training_dict, reasoning, status, error):
        self.state.add_experiment(
            ExperimentRecord(
                experiment_id=exp_id,
                name=f"autoparam_{exp_id:03d}",
                model_config=model_dict,
                training_config=training_dict,
                score={},
                status=status,
                error_message=error,
                timestamp_start=datetime.now().isoformat(),
                timestamp_end=datetime.now().isoformat(),
                llm_reasoning=reasoning,
            )
        )

    def _print_summary(self):
        successes = self.state.successful_experiments()
        print(f"\n[autoparam] ═══ Summary ═══")
        print(
            f"Total : {len(self.state.experiments)}  Successful : {len(successes)}  Failed : {len(self.state.experiments) - len(successes)}"
        )

        if not successes:
            return

        top = sorted(
            successes, key=lambda e: e.score.get("final_accuracy", 0), reverse=True
        )[:5]
        print(f"\n── Top {len(top)} ──")
        for rank, e in enumerate(top, 1):
            s = e.score
            steps = s.get("steps_to_threshold", -1)
            print(
                f"  #{rank}  exp={e.experiment_id:03d}  "
                f"acc={s.get('final_accuracy', 0):.2f}%  "
                f"ppl={s.get('perplexity', 0):.1f}  "
                f"slope={s.get('accuracy_slope', 0):.4f}  "
                f"steps_to_{int(STEPS_TO_ACCURACY_THRESHOLD)}pct={'never' if steps < 0 else steps}  "
                f"stability={s.get('stability_score', 0):.3f}"
            )
            print(f"       model    : {json.dumps(e.model_config)}")
            print(f"       training : {json.dumps(e.training_config)}")
            print(f"       reasoning: {e.llm_reasoning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous hyperparameter optimization"
    )
    parser.add_argument(
        "--dataset", default=os.environ.get("TARGET_DATASET", "fineweb-256")
    )
    parser.add_argument("--max-experiments", type=int, default=1_000)
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=int(os.environ.get("TRAINING_TIME_MINUTES", TRAINING_TIME_MINUTES)),
    )
    parser.add_argument(
        "--check-spend",
        action="store_true",
        help="Print today's OpenRouter spend and exit",
    )
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--distributed-strategy",
        default="fsdp",
        choices=["none", "ddp", "fsdp"],
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=max(1, torch.cuda.device_count()),
        help="Number of GPUs per node for the executor (default: all available GPUs)",
    )
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--llm-model", default=LLM_MODEL)
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Disable the OpenRouter LLM proposer and use random search only",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default="5.00",
        metavar="USD",
        help="Stop when OpenRouter daily spend exceeds this amount (in USD)",
    )
    args = parser.parse_args()

    if args.check_spend:
        daily = fetch_openrouter_daily_usage()
        if daily < 0:
            print("Failed to fetch OpenRouter usage (check OPENROUTER_API_KEY).")
        else:
            print(f"OpenRouter spend today: ${daily:.4f}")
        exit(0)

    strategy = DistributedStrategy[args.distributed_strategy.upper()]

    AutoparamLoop(
        dataset_name=args.dataset,
        max_experiments=args.max_experiments,
        experiment_timeout_minutes=args.timeout_minutes,
        state_path=args.state,
        llm_model=args.llm_model,
        budget_usd=args.budget,
        distributed_strategy=strategy,
        nproc_per_node=args.nproc_per_node,
        max_consecutive_failures=args.max_consecutive_failures,
        random_only=args.random_only,
    ).run()
