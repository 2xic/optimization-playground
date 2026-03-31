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
import os
import re
import argparse
import hashlib
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List

import torch

from dotenv import load_dotenv

load_dotenv()

from experiments import (
    execute,
    create_default_config,
    NAMED_DATASETS,
    TRAINING_TIME_MINUTES,
)
from training.model import (
    Model,
    TransformerLayerType,
    PositionalEmbeddingType,
    NormalizationLayerType,
)
from training.trainer import TrainingOptions
from training.optimizer import (
    AdamConfig,
    AdamWConfig,
    RMSpropConfig,
    MuonConfig,
    NoamScheduler,
    WarmupExpDecay,
    StepExponentialLR,
)
from utils.plot import Results
import matplotlib.pyplot as plt

ACCURACY_WEIGHT = 0.70
STABILITY_WEIGHT = 0.30
STABILITY_TAIL_FRACTION = 0.25

LLM_MODEL = "anthropic/claude-opus-4-5"
LLM_MAX_TOKENS = 1024
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HISTORY_WINDOW = 15

SEARCH_SPACE_DESCRIPTION = """
Searchable hyperparameter space (use ONLY the values listed):

Model config:
  dim_embeddings:          [128, 256, 384, 512, 768]
  num_attention_heads:     [4, 8, 12, 16]   ← dim_embeddings MUST be divisible by this
  num_transformer_layers:  [2, 4, 6, 8, 12]
  dropout:                 [0.0, 0.05, 0.1, 0.2]
  feed_forward_layer:      [512, 1024, 2048, 4096]
  bias:                    [true, false]
  hc_n:                    [2, 4, 8]   ← only relevant for OLMO_HYPER_CONNECTIONS variants

Architecture (exact enum names):
  transformer_layer:       SIMPLE | GPT2 | LLAMA2 | LLAMA3 | DEEPSEEK | OLMO |
                           OLMO_HYPER_CONNECTIONS | OLMO_CONSTRAINED_HYPER_CONNECTIONS |
                           OLMO_IDENTITY_HYPER_CONNECTIONS | SIMPLE_NO_ATTENTION |
                           BERT | SIMPLE_ATTENTION_AT_HOME
  positional_embedding:    NN_EMBEDDING | SINUSOIDAL | ROTARY_POSITION_ENCODING | NONE
  normalization_layer:     LAYER_NORM | DyT

Optimizer:
  optimizer_type:          adam | adamw | rmsprop | muon
  lr:                      float in [0.0001, 0.002]
  weight_decay:            [0, 0.01, 0.1]
  max_grad_norm:           [0, 0.5, 1.0, 5.0]   ← 0 = disabled; controls gradient clipping
  beta1:                   float in [0.85, 0.95]   (adam/adamw only)
  beta2:                   float in [0.90, 0.999]  (adam/adamw only)
  alpha:                   [0.9, 0.95, 0.99]       (rmsprop only)
  momentum:                [0, 0.1, 0.9]           (rmsprop only)

Scheduler:
  scheduler_type:          none | noam | warmup_exp_decay | step_exp
  warmup_steps:            [500, 1000, 2000, 4000]
  decay_steps:             [10000, 50000, 100000]
  min_lr_ratio:            [0.01, 0.05, 0.1]   ← floor as fraction of initial lr (warmup_exp_decay/step_exp only)

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
        }

    @staticmethod
    def training_options_to_dict(opts: TrainingOptions) -> dict:
        opt = opts.optimizer
        opt_type = type(opt).__name__.lower().replace("config", "")
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
        if isinstance(sched, WarmupExpDecay):
            d["scheduler_type"] = "warmup_exp_decay"
            d["warmup_steps"] = sched.warmup_steps
            d["decay_steps"] = sched.decay_steps
            d["min_lr_ratio"] = sched.min_lr_ratio
        elif isinstance(sched, NoamScheduler):
            d["scheduler_type"] = "noam"
            d["warmup_steps"] = sched.warmup_steps
        elif isinstance(sched, StepExponentialLR):
            d["scheduler_type"] = "step_exp"
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
        return cls._validate_and_repair(config)

    @classmethod
    def dict_to_training_options(cls, d: dict, timeout_minutes: int) -> TrainingOptions:
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
                d_model=int(d.get("dim_embeddings", 256)),
                warmup_steps=int(d.get("warmup_steps", 1000)),
            )
        elif sched_type == "step_exp":
            scheduler = StepExponentialLR(
                decay_steps=int(d.get("decay_steps", 50000)),
                min_lr_ratio=min_lr_ratio,
            )

        return TrainingOptions(
            batch_size=int(d.get("batch_size", 32)),
            accumulation_steps=int(d.get("accumulation_steps", 1)),
            training_timeout_minutes=timeout_minutes,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            record_interval_steps=50,
        )

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
                "stability_score": 0.0,
                "combined_score": 0.0,
                "raw_variance": float("inf"),
            }

        tail = avg[-max(1, int(len(avg) * STABILITY_TAIL_FRACTION)) :]
        final_accuracy = statistics.mean(tail)
        variance = statistics.variance(tail) if len(tail) > 1 else 0.0
        stability_score = 1.0 / (1.0 + variance**0.5)
        combined_score = (
            ACCURACY_WEIGHT * (final_accuracy / 100.0)
            + STABILITY_WEIGHT * stability_score
        )
        return {
            "final_accuracy": round(final_accuracy, 4),
            "stability_score": round(stability_score, 4),
            "combined_score": round(combined_score, 4),
            "raw_variance": round(variance, 6),
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
            score = record.score.get("combined_score", -1.0)
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
                history_lines.append(
                    f"  #{exp.experiment_id} [ok] acc={s.get('final_accuracy', 0):.2f}% "
                    f"stability={s.get('stability_score', 0):.3f} "
                    f"combined={s.get('combined_score', 0):.4f} | "
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
            best_section = (
                f"Best: #{best.experiment_id} combined={best.score.get('combined_score', 0):.4f}\n"
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
Propose the next experiment. Use the history to guide exploration — try promising \
directions, avoid repeating failures, and reason about what has worked so far.

Respond with JSON matching this schema exactly:
{{
  "reasoning": "<your explanation>",
  "dim_embeddings": <int>, "num_attention_heads": <int>, "num_transformer_layers": <int>,
  "dropout": <float>, "feed_forward_layer": <int>, "bias": <bool>, "hc_n": <int>,
  "transformer_layer": "<string>", "positional_embedding": "<string>", "normalization_layer": "<string>",
  "optimizer_type": "<string>", "lr": <float>, "beta1": <float>, "beta2": <float>,
  "weight_decay": <float>, "max_grad_norm": <float>, "alpha": <float>, "momentum": <float>,
  "scheduler_type": "<string>", "warmup_steps": <int>, "decay_steps": <int>, "min_lr_ratio": <float>,
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
    stability = [e.score["stability_score"] * 100 for e in successes]
    combined = [e.score["combined_score"] * 100 for e in successes]

    _, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ids, accuracy, "b-o", label="Accuracy (%)", markersize=5)
    ax.plot(ids, combined, "g-o", label="Combined score (×100)", markersize=5)
    ax.plot(
        ids,
        stability,
        "orange",
        linestyle="--",
        marker="s",
        label="Stability (×100)",
        markersize=4,
        alpha=0.7,
    )

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
    ax.set_ylabel("Score")
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
        experiment_timeout_minutes: int = 20,
        state_path: str = "autoparam_state.json",
        llm_model: str = LLM_MODEL,
    ):
        self.max_experiments = max_experiments
        self.log_path = state_path.replace(".json", ".log")
        self.timeout = experiment_timeout_minutes
        self.state = AutoparamState(state_path)
        self.proposer = LLMProposer(model=llm_model)
        self.dataset = NAMED_DATASETS[dataset_name]
        self.plot_path = os.path.join("plots", dataset_name, "autoparam_progress.png")
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)

        baseline_config = create_default_config(self.dataset)
        baseline_opts = TrainingOptions(
            batch_size=32, training_timeout_minutes=experiment_timeout_minutes
        )
        self.baseline_dict = {
            **ConfigSerializer.config_to_dict(baseline_config),
            **ConfigSerializer.training_options_to_dict(baseline_opts),
        }

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
        print(
            f"[autoparam] Starting from experiment {start_id}. Target: {self.max_experiments}. Timeout: {self.timeout}min each."
        )
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5

        for exp_id in range(start_id, self.max_experiments):
            self._log(f"=== Experiment {exp_id + 1}/{self.max_experiments} ===")

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
                self._record(
                    exp_id,
                    proposed,
                    proposed,
                    reasoning,
                    "failed",
                    f"Config error: {e}",
                )
                consecutive_failures += 1
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
                continue

            self._log(
                f"Config: {json.dumps(model_dict)}  training: {json.dumps(training_dict)}"
            )

            exp_name = f"autoparam_{exp_id:03d}"
            timestamp_start = datetime.now().isoformat()
            score, status, error_message = {}, "failed", None

            try:
                _, results = execute(
                    self.dataset, exp_name, Model(config), training_options
                )
                score = StabilityMetric.compute(results)
                no_data = (
                    len(results.accuracy.min_max_avg) == 0
                    and len(results.step_accuracy.min_max_avg) == 0
                )
                if no_data:
                    status = "failed"
                    error_message = (
                        "No training data collected (dataloader may be failing)"
                    )
                    self._log(f"Training failed: {error_message}")
                    torch.cuda.empty_cache()
                    consecutive_failures += 1
                else:
                    status = "success"
                    self._log(
                        f"Result: accuracy={score['final_accuracy']:.2f}%  "
                        f"stability={score['stability_score']:.3f}  "
                        f"combined={score['combined_score']:.4f}"
                    )
            except torch.cuda.OutOfMemoryError as e:
                error_message = f"CUDA out of memory: {e}"
                self._log(f"Training failed (OOM — clearing CUDA cache): {error_message}")
                torch.cuda.empty_cache()
                consecutive_failures += 1
            except Exception as e:
                import traceback

                error_message = str(e)
                self._log(f"Training failed: {e}\n{traceback.format_exc()}")
                consecutive_failures += 1

            if status == "success":
                consecutive_failures = 0

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
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
                    f"combined={best.score.get('combined_score', 0):.4f}  "
                    f"acc={best.score.get('final_accuracy', 0):.2f}%"
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
            successes, key=lambda e: e.score.get("combined_score", 0), reverse=True
        )[:5]
        print(f"\n── Top {len(top)} ──")
        for rank, e in enumerate(top, 1):
            s = e.score
            print(
                f"  #{rank}  exp={e.experiment_id:03d}  combined={s.get('combined_score', 0):.4f}  acc={s.get('final_accuracy', 0):.2f}%  stability={s.get('stability_score', 0):.3f}"
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
    parser.add_argument("--max-experiments", type=int, default=40)
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=int(os.environ.get("TRAINING_TIME_MINUTES", TRAINING_TIME_MINUTES)),
    )
    parser.add_argument("--state", default="autoparam_state.json")
    parser.add_argument("--llm-model", default=LLM_MODEL)
    args = parser.parse_args()

    AutoparamLoop(
        dataset_name=args.dataset,
        max_experiments=args.max_experiments,
        experiment_timeout_minutes=args.timeout_minutes,
        state_path=args.state,
        llm_model=args.llm_model,
    ).run()
