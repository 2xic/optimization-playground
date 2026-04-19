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


_GPU_MEMORY_HEADROOM = 0.75


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

LLM_MODEL = "anthropic/claude-opus-4-5"
LLM_MAX_TOKENS = 1024
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HISTORY_WINDOW = 15

SEARCH_SPACE_DESCRIPTION = """
Searchable hyperparameter space (use ONLY the values listed):

Model config:
  dim_embeddings:          [128, 256, 384, 512, 768, 1024]
  num_attention_heads:     [4, 8, 12, 16, 32]   ← dim_embeddings MUST be divisible by this
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

Optimizer:
  optimizer_type:          adam | adamw | rmsprop | muon | muon_hybrid
  lr:                      float in [0.0001, 0.002]
  weight_decay:            [0, 0.01, 0.1]
  max_grad_norm:           [0, 0.5, 1.0, 5.0]   ← 0 = disabled; controls gradient clipping
  beta1:                   float in [0.85, 0.95]   (adam/adamw only)
  beta2:                   float in [0.90, 0.999]  (adam/adamw only)
  alpha:                   [0.9, 0.95, 0.99]       (rmsprop only)
  momentum:                [0, 0.1, 0.9]           (rmsprop only)
  Note: muon/muon_hybrid ignore beta1/beta2/alpha/momentum — only lr/weight_decay apply
  Note: muon_hybrid uses Muon for hidden Linear weights, AdamW for embeddings/lm_head (best of both)

Scheduler:
  scheduler_type:          none | noam | warmup_exp_decay | step_exp | cosine | trapezoidal
  warmup_steps:            [500, 1000, 2000, 4000]
  flat_steps:              [1000, 5000, 10000, 20000]   (trapezoidal only)
  decay_steps:             [10000, 50000, 100000]
  min_lr_ratio:            [0.01, 0.05, 0.1]   ← floor as fraction of initial lr

Training:
  batch_size:              [16, 32, 64, 128]
  accumulation_steps:      [1, 2, 4, 8]
"""

_SYSTEM_PROMPT = f"""You are an expert ML researcher running autonomous hyperparameter optimization \
of a PyTorch transformer language model trained on web text (next-token prediction / causal LM).

Your goal: find configurations that maximize accuracy AND training stability.

{SEARCH_SPACE_DESCRIPTION}

Hard constraints:
- dim_embeddings MUST be divisible by num_attention_heads
- All enum values must match exactly (case-sensitive)
- lr must be between 0.0001 and 0.002
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
            "batch_size": opts.batch_size,
            "accumulation_steps": opts.accumulation_steps,
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
        return d

    @classmethod
    def dict_to_config(cls, d: dict, dataset):
        config = create_default_config(dataset)
        config.dim_embeddings = int(d["dim_embeddings"])
        config.num_attention_heads = int(d["num_attention_heads"])
        config.num_transformer_layers = int(d["num_transformer_layers"])
        config.dropout = float(d["dropout"])
        config.feed_forward_layer = int(d["feed_forward_layer"])
        config.bias = bool(d.get("bias", False))
        config.hc_n = int(d.get("hc_n", 4))
        config.transformer_layer = TransformerLayerType[d["transformer_layer"]]
        config.positional_embedding = PositionalEmbeddingType[d["positional_embedding"]]
        config.normalization_layer = NormalizationLayerType[d["normalization_layer"]]
        config.attention_type = AttentionType[d.get("attention_type", "DEFAULT")]
        config.qk_norm = bool(d.get("qk_norm", False))
        return cls._validate_and_repair(config)

    @classmethod
    def dict_to_training_options(cls, d: dict, timeout_minutes: int, distributed_strategy: DistributedStrategy = DistributedStrategy.FSDP, device: Optional[torch.device] = None) -> TrainingOptions:
        opt_type = d.get("optimizer_type", "adamw")
        lr = float(d.get("lr", 3e-4))
        wd = float(d.get("weight_decay", 0.01))

        max_grad_norm = float(d.get("max_grad_norm", 0))
        if opt_type in ("adam", "adamw"):
            opt_cls = AdamConfig if opt_type == "adam" else AdamWConfig
            optimizer = opt_cls(
                lr=lr,
                betas=(float(d.get("beta1", 0.90)), float(d.get("beta2", 0.95))),
                weight_decay=wd,
                max_grad_norm=max_grad_norm,
            )
        elif opt_type == "rmsprop":
            optimizer = RMSpropConfig(
                lr=lr,
                alpha=float(d.get("alpha", 0.99)),
                weight_decay=wd,
                momentum=float(d.get("momentum", 0)),
            )
        elif opt_type == "muon_hybrid":
            optimizer = MuonConfig(lr=lr, hybrid=True)
        else:
            optimizer = MuonConfig(lr=lr)

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
            training_timeout_minutes=timeout_minutes,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            record_interval_steps=50,
            distributed_strategy=distributed_strategy,
        )
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
    }


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
        if best:
            bs = best.score
            best_section = (
                f"Best: #{best.experiment_id} acc={bs.get('final_accuracy', 0):.2f}%  "
                f"ppl={bs.get('perplexity', 0):.1f}  "
                f"slope={bs.get('accuracy_slope', 0):.4f}  "
                f"stability={bs.get('stability_score', 0):.3f}\n"
                f"  model={json.dumps(best.model_config)}\n"
                f"  training={json.dumps(best.training_config)}"
            )

        return f"""## Baseline configuration
{json.dumps(baseline_dict, indent=2)}

## Experiment history (last {len(recent)})
{chr(10).join(history_lines) if history_lines else "  (none yet)"}

## Current best
{best_section}

## Task
Accuracy is currently plateaued near 30%. The goal is to break past this ceiling. \
Prioritize configurations likely to exceed 30%: larger models (dim>=512, layers>=8), \
muon_hybrid optimizer (Muon for hidden layers + AdamW for embeddings — strongest option), \
LLAMA3/DEEPSEEK architectures with ROTARY_POSITION_ENCODING, and \
trapezoidal scheduler (warmup→flat→decay). Avoid incremental tweaks to configs \
already stuck at 30%. Reason about what fundamentally changes the learning dynamics.

Respond with JSON matching this schema exactly:
{{
  "reasoning": "<your explanation>",
  "dim_embeddings": <int>, "num_attention_heads": <int>, "num_transformer_layers": <int>,
  "dropout": <float>, "feed_forward_layer": <int>, "bias": <bool>, "hc_n": <int>,
  "transformer_layer": "<string>", "positional_embedding": "<string>", "normalization_layer": "<string>", "attention_type": "<string>", "qk_norm": <bool>,
  "optimizer_type": "<string>", "lr": <float>, "beta1": <float>, "beta2": <float>,
  "weight_decay": <float>, "max_grad_norm": <float>, "alpha": <float>, "momentum": <float>,
  "scheduler_type": "<string>", "warmup_steps": <int>, "flat_steps": <int>, "decay_steps": <int>, "min_lr_ratio": <float>,
  "batch_size": <int>, "accumulation_steps": <int>
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

    ids = [e.experiment_id for e in successes]
    accuracy = [e.score["final_accuracy"] for e in successes]
    loss_vals = [
        e.score["final_loss"]
        for e in successes
        if math.isfinite(e.score.get("final_loss", float("inf")))
    ]
    loss_ids = [
        e.experiment_id
        for e in successes
        if math.isfinite(e.score.get("final_loss", float("inf")))
    ]

    _, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ids, accuracy, "b-o", label="Accuracy (%)", markersize=5)

    if loss_ids:
        ax2 = ax.twinx()
        ax2.plot(
            loss_ids,
            loss_vals,
            "r--",
            marker="^",
            label="Loss",
            markersize=4,
            alpha=0.6,
        )
        ax2.set_ylabel("Loss", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper right")

    best = state.best_record
    if best:
        ax.axvline(
            x=best.experiment_id,
            color="red",
            linestyle=":",
            alpha=0.6,
            label=f"Best (#{best.experiment_id})",
        )

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Autoparam: optimization progress over time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
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
        self.proposer = LLMProposer(model=llm_model)
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
            batch_size=32,
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
        if self.budget_usd:
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

            if self.budget_usd:
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

            if exp_id % RANDOM_EXPLORE_EVERY == 0:
                proposed = random_config_dict()
                reasoning = proposed.pop("reasoning")
                self._log(
                    f"Random exploration (every {RANDOM_EXPLORE_EVERY} experiments)"
                )
            else:
                try:
                    proposed = self.proposer.propose(self.state, self.baseline_dict)
                    reasoning = proposed.pop("reasoning", "(no reasoning provided)")
                    self._log(f"Reasoning: {reasoning}")
                except Exception as e:
                    self._log(f"LLM proposal failed ({e}), using baseline.")
                    proposed = dict(self.baseline_dict)
                    reasoning = f"LLM failed: {e}"

            try:
                config = ConfigSerializer.dict_to_config(proposed, self.dataset)
                training_options = ConfigSerializer.dict_to_training_options(
                    proposed, self.timeout
                )
                model_dict = ConfigSerializer.config_to_dict(config)
                training_dict = ConfigSerializer.training_options_to_dict(
                    training_options
                )
            except Exception as e:
                import traceback
                self._log(f"Config error: {e}\n{traceback.format_exc()}")
                self._record(
                    exp_id,
                    proposed,
                    proposed,
                    reasoning,
                    "failed",
                    f"Config error: {e}",
                )
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._log(f"Stopping early: {consecutive_failures} consecutive failures.")
                    return
                continue

            if self._already_run(model_dict, training_dict):
                self._log("Skipping duplicate config.")
                self._record(
                    exp_id,
                    model_dict,
                    training_dict,
                    reasoning,
                    "failed",
                    "Duplicate config",
                )
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    self._log(f"Stopping early: {consecutive_failures} consecutive failures.")
                    return
                continue

            self._log(
                f"Config: {json.dumps(model_dict)}  training: {json.dumps(training_dict)}"
            )

            exp_name = f"autoparam_{exp_id:03d}"
            timestamp_start = datetime.now().isoformat()
            score, status, error_message = {}, "failed", None

            config_data = {
                "dataset_name": self.dataset.name,
                "exp_name": exp_name,
                "timeout_minutes": self.timeout,
                "model_config": model_dict,
                "training_config": training_dict,
                "distributed_strategy": self.distributed_strategy.name,
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
                    executor,
                    "--config", config_path,
                    "--result", result_path,
                ]
                log_path = result_path.replace("_result.json", "_run.log")
                log_file = open(log_path, "w")
                print(f"[autoparam] subprocess log: {log_path}", flush=True)
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
                log_file.close()
                self._active_proc = proc
                pgid = os.getpgid(proc.pid)
                self._active_pgid = pgid
                try:
                    proc.wait(timeout=(self.timeout + 5) * 60)
                except subprocess.TimeoutExpired:
                    print(f"Experiment subprocess timed out after {self.timeout + 5} minutes, killing")
                except KeyboardInterrupt:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                    raise
                finally:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        pass
                    self._active_proc = None
                    self._active_pgid = None
                    time.sleep(3)

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
                    self._log(
                        f"Result: accuracy={score['final_accuracy']:.2f}%  "
                        f"ppl={score.get('perplexity', 0):.1f}  "
                        f"slope={score.get('accuracy_slope', 0):.4f}  "
                        f"steps_to_{int(STEPS_TO_ACCURACY_THRESHOLD)}pct={'never' if steps < 0 else steps}  "
                        f"stability={score['stability_score']:.3f}"
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
    ).run()
