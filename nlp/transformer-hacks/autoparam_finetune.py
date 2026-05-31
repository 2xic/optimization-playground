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
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Optional

import torch

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments import NAMED_DATASETS, TRAINING_TIME_MINUTES, create_default_config
from training.trainer import TrainingOptions, DistributedStrategy
from autoparam import (
    AutoparamState,
    ExperimentRecord,
    LLMProposer,
    ConfigSerializer,
    SEARCH_SPACE_DESCRIPTION,
    LLM_MODEL,
    STEPS_TO_ACCURACY_THRESHOLD,
    RANDOM_EXPLORE_EVERY,
    MUTATE_PROB,
    HISTORY_WINDOW,
    _promote_best_tag,
    _pick_extension_candidate,
    _pick_mutation_parent,
    mutate_config_dict,
    random_config_dict,
    _gpus_are_stuck,
    fetch_openrouter_daily_usage,
    _estimate_footprint_gb_meta,
    _total_gpu_memory_gb,
    plot_progress,
)
from utils.load_mode_from_checkpoint import load_modeL_tag
from utils.checkpoints import StorageBox, should_checkpoint
from scheduler.cooperative import install_shutdown_handler, shutdown_requested

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
    def __init__(self, model: str = LLM_MODEL):
        super().__init__(model=model)

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


class FinetuneAutoparamLoop:
    def __init__(
        self,
        init_tag: str,
        max_experiments: int = 50,
        experiment_timeout_minutes: int = 60,
        state_path: str = "autoparam_finetune_state.json",
        llm_model: str = LLM_MODEL,
        budget_usd: Optional[float] = None,
        distributed_strategy: DistributedStrategy = DistributedStrategy.FSDP,
        nproc_per_node: int = 1,
        max_consecutive_failures: int = 5,
        random_only: bool = False,
    ):
        self.init_tag = init_tag
        self.max_experiments = max_experiments
        self.budget_usd = budget_usd
        self.distributed_strategy = distributed_strategy
        self.nproc_per_node = nproc_per_node
        self.max_consecutive_failures = max_consecutive_failures
        self.timeout = experiment_timeout_minutes
        self.log_path = state_path.replace(".json", ".log")
        self.state = AutoparamState(state_path)
        self._llm_disabled = random_only
        self.proposer = None if random_only else SFTLLMProposer(model=llm_model)
        self.dataset = NAMED_DATASETS["smoltalk-256"]
        self.plot_path = os.path.join("plots", CHAT_TAG, "autoparam_progress.png")
        self.locked_arch = _load_pretrain_arch(self.init_tag)
        print(f"[autoparam-ft] Locked arch from pretrain: {self.locked_arch}", flush=True)

        self._active_proc = None
        self._active_pgid = None
        self._active_stop_file = None
        self._stop_requested_at = None
        self._stop_grace_seconds = int(os.environ.get("AUTOPARAM_STOP_GRACE_SECONDS", "1800"))
        import atexit
        atexit.register(self._kill_active_proc)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._forward_sigusr1)

        baseline_config = create_default_config(self.dataset)
        baseline_opts = TrainingOptions(
            batch_size=32,
            training_timeout_minutes=experiment_timeout_minutes,
        )
        self.baseline_dict = {
            **ConfigSerializer.config_to_dict(baseline_config),
            **ConfigSerializer.training_options_to_dict(baseline_opts),
        }
        for k, v in self.locked_arch.items():
            if k in self.baseline_dict:
                self.baseline_dict[k] = v

    def _apply_locked_arch(self, cand: dict):
        for k, v in self.locked_arch.items():
            if k == "vocab_size":
                continue
            cand[k] = v

    def _apply_locked_arch_to_config(self, cfg_obj):
        from enum import Enum
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

    def _forward_sigusr1(self, signum, frame):
        from scheduler.cooperative import _flag
        _flag.set()
        self._stop_requested_at = time.time()
        path = self._active_stop_file
        if path:
            try:
                open(path, "w").close()
            except OSError:
                pass

    @staticmethod
    def _config_hash(model_dict, training_dict):
        return hashlib.md5(
            json.dumps({**model_dict, **training_dict}, sort_keys=True).encode()
        ).hexdigest()

    def _already_run(self, model_dict, training_dict):
        h = self._config_hash(model_dict, training_dict)
        return any(
            self._config_hash(e.model_config, e.training_config) == h
            for e in self.state.experiments
        )

    def _log(self, text: str):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _record(self, exp_id, model_dict, training_dict, reasoning, status, error):
        self.state.add_experiment(
            ExperimentRecord(
                experiment_id=exp_id,
                name=f"autoparam_ft_{exp_id:03d}",
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

    def run(self):
        import random as _random
        start_id = len(self.state.experiments)
        budget_msg = f"  Budget: ${self.budget_usd:.2f}" if self.budget_usd else ""
        print(
            f"[autoparam-ft] Init checkpoint: {self.init_tag}. "
            f"Starting from experiment {start_id}. Target: {self.max_experiments}.{budget_msg}",
            flush=True,
        )
        if self.budget_usd and not self._llm_disabled:
            self._daily_spend_at_start = fetch_openrouter_daily_usage()
            if self._daily_spend_at_start >= 0:
                self._log(f"OpenRouter daily spend at start: ${self._daily_spend_at_start:.4f}")

        consecutive_failures = 0

        for exp_id in range(start_id, self.max_experiments):
            if shutdown_requested():
                self._log("Shutdown signal received — exiting cleanly between experiments.")
                break
            if _gpus_are_stuck():
                self._log("ERROR: GPUs stuck — aborting.")
                break
            self._log(f"=== Experiment {exp_id + 1}/{self.max_experiments} ===")

            if self.budget_usd and not self._llm_disabled:
                daily = fetch_openrouter_daily_usage()
                if daily >= 0:
                    spent = daily - self._daily_spend_at_start
                    self._log(f"OpenRouter spend this session: ${spent:.4f} / ${self.budget_usd:.2f}")
                    if spent >= self.budget_usd:
                        self._log(f"Budget reached. Stopping.")
                        break

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
                            else f"Random exploration (every {RANDOM_EXPLORE_EVERY})"
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
                            self._log(f"LLM credits exhausted ({e}). Switching to random-only.")
                        else:
                            self._log(f"LLM proposal failed ({e}), using random fallback.")
                        cand = random_config_dict()
                        cand_reason = cand.pop("reasoning", f"LLM failed: {e}")
                        src = f"Reasoning: {cand_reason}"

                self._apply_locked_arch(cand)
                try:
                    cfg_obj = ConfigSerializer.dict_to_config(cand, self.dataset)
                    self._apply_locked_arch_to_config(cfg_obj)
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
                self._log(f"Could not find non-duplicate config after {MAX_DEDUP_ATTEMPTS} attempts; skipping.")
                continue

            if config_error is not None:
                import traceback
                self._log(f"Config error: {config_error}\n{traceback.format_exc()}")
                self._record(exp_id, proposed, proposed, reasoning, "failed", f"Config error: {config_error}")
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
                        exp_id, model_dict, training_dict, reasoning, "skipped",
                        f"Model too large: estimated {est_gb:.2f} GB > {budget_gb:.2f} GB budget",
                    )
                    continue
            except Exception as e:
                self._log(f"Meta-device size estimate failed (continuing anyway): {e}")

            self._log(f"Config: {json.dumps(model_dict)}  training: {json.dumps(training_dict)}")

            exp_name = f"autoparam_ft_{exp_id:03d}"
            timestamp_start = datetime.now().isoformat()
            score, status, error_message = {}, "failed", None
            run_tag = f"autoparam-finetune-{exp_name}"

            config_data = {
                "exp_name": exp_name,
                "timeout_minutes": training_options.training_timeout_minutes,
                "model_config": model_dict,
                "training_config": training_dict,
                "distributed_strategy": self.distributed_strategy.name,
                "is_extension": is_extension,
                "checkpoint_tag": run_tag if should_checkpoint(is_extension) else None,
                "init_from_tag": self.init_tag,
            }

            config_path = None
            result_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp") as f:
                    json.dump(config_data, f)
                    config_path = f.name
                result_path = config_path.replace(".json", "_result.json")
                stop_file = config_path.replace(".json", ".stop")

                executor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoparam_finetune_executor.py")
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
                print(f"[autoparam-ft] subprocess log: {log_path}", flush=True)
                env = os.environ.copy()
                env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
                env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
                env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
                env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                env["AUTOPARAM_STOP_FILE"] = stop_file
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True, env=env)
                log_file.close()
                self._active_proc = proc
                pgid = os.getpgid(proc.pid)
                self._active_pgid = pgid
                self._active_stop_file = stop_file
                self._stop_requested_at = None
                hard_deadline = time.time() + (training_options.training_timeout_minutes + 5) * 60
                try:
                    while True:
                        try:
                            proc.wait(timeout=5)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        now = time.time()
                        if now >= hard_deadline:
                            self._log("Experiment subprocess past hard deadline, killing")
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except OSError:
                                pass
                            break
                        if shutdown_requested() and not os.path.exists(stop_file):
                            try:
                                open(stop_file, "w").close()
                            except OSError:
                                pass
                            try:
                                os.killpg(pgid, signal.SIGUSR1)
                            except OSError:
                                pass
                            if self._stop_requested_at is None:
                                self._stop_requested_at = now
                        if self._stop_requested_at and now - self._stop_requested_at >= self._stop_grace_seconds:
                            self._log(
                                f"Stop requested >{self._stop_grace_seconds}s ago, escalating to SIGKILL"
                            )
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except OSError:
                                pass
                            break
                except KeyboardInterrupt:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                    raise
                finally:
                    if proc.poll() is None:
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
                    self._active_stop_file = None
                    self._stop_requested_at = None
                    if os.path.exists(stop_file):
                        try:
                            os.unlink(stop_file)
                        except OSError:
                            pass
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
                    params_m = score.get("params_count", 0) / 1e6
                    pmem = score.get("peak_memory_gb", 0)
                    tokens = score.get("tokens_seen", 0)
                    warn = score.get("partial_load_warning")
                    warn_str = f"  WARNING: {warn}" if warn else ""
                    self._log(
                        f"Result: val_acc={score.get('val_accuracy', float('nan')):.2f}%  "
                        f"val_loss={score.get('val_loss', float('nan')):.3f}  "
                        f"slope={score.get('accuracy_slope', 0):.4f}  "
                        f"params={params_m:.1f}M  peak_mem={pmem:.2f}GB  tokens={tokens:,}{warn_str}"
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
                self._log(f"Stopping early: {consecutive_failures} consecutive failures.")
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
            if status == "success":
                try:
                    acc = score.get("val_accuracy", score.get("final_accuracy", -1.0))
                    slope = score.get("accuracy_slope", 0.0)
                    val_loss = score.get("val_loss")
                    import math as _math
                    composite = acc + 0.5 * max(0.0, slope * 500)
                    if val_loss is not None and _math.isfinite(val_loss):
                        composite -= 2.0 * float(val_loss)
                    promoted = _promote_best_tag(
                        CHAT_TAG,
                        run_tag,
                        composite,
                        accuracy=float(acc),
                        metadata={"experiment_id": exp_id, "exp_name": exp_name, "score": score},
                    )
                    if promoted:
                        self._log(f"Promoted #{exp_id} to best for {run_tag} (score={composite:.4f})")
                    else:
                        self._log(f"#{exp_id} not promoted (score={composite:.4f} did not beat stored best)")
                except Exception as e:
                    self._log(f"Failed to promote best tag: {e}")
            try:
                plot_progress(self.state, self.plot_path)
            except Exception as e:
                self._log(f"plot_progress failed: {e}")

            best = self.state.best_record
            if best:
                self._log(
                    f"Best so far: #{best.experiment_id}  "
                    f"val_acc={best.score.get('val_accuracy', best.score.get('final_accuracy', 0)):.2f}%  "
                    f"val_loss={best.score.get('val_loss', 0):.3f}"
                )

        self._print_summary()

    def _print_summary(self):
        successes = self.state.successful_experiments()
        print(f"\n[autoparam-ft] ═══ Summary ═══")
        print(
            f"Total : {len(self.state.experiments)}  Successful : {len(successes)}  "
            f"Failed : {len(self.state.experiments) - len(successes)}"
        )
        if not successes:
            return
        top = sorted(successes, key=lambda e: e.score.get("final_accuracy", 0), reverse=True)[:5]
        print(f"\n── Top {len(top)} ──")
        for rank, e in enumerate(top, 1):
            s = e.score
            print(
                f"  #{rank}  exp={e.experiment_id:03d}  "
                f"val_acc={s.get('val_accuracy', s.get('final_accuracy', 0)):.2f}%  "
                f"val_loss={s.get('val_loss', 0):.3f}  "
                f"slope={s.get('accuracy_slope', 0):.4f}"
            )
            print(f"       model    : {json.dumps(e.model_config)}")
            print(f"       training : {json.dumps(e.training_config)}")
            print(f"       reasoning: {e.llm_reasoning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous SFT hyperparameter optimization")
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--pretrain-tag", help="Explicit pretrained checkpoint tag")
    init_group.add_argument("--use-best-of", metavar="DATASET",
                            help="Resolve to best-<DATASET> at startup (e.g. fineweb-256)")
    parser.add_argument("--max-experiments", type=int, default=50)
    parser.add_argument("--budget", type=float, default=5.00, metavar="USD")
    parser.add_argument("--state-file", default="autoparam_finetune_state.json")
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
