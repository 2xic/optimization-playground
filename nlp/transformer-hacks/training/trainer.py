from .model import Config, TransformerLayerType
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
try:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy as FSDP2MixedPrecision
    _FSDP2_AVAILABLE = True
except ImportError:
    _FSDP2_AVAILABLE = False
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable, Optional, Union
from tqdm import tqdm
from dataclasses import dataclass, field
from enum import Enum
import os
import time
import shutil
from abc import ABC
from .objectives import BaseObjective
from .rho_loss import RhoLossConfig, RhoLossTagConfig, RhoLossEmaConfig, RhoLossSnapshotConfig  # noqa: F401
from .optimizer import BaseOptimizerConfig, AdamConfig, Scheduler
from .device_caps import supports_bf16, supports_amp, supports_torch_compile
from utils.metrics import MetricsTracker
from utils.checkpoints import StorageBoxCheckpoint, Stats, TrainingMetadata
from scheduler.cooperative import shutdown_requested
import signal as _signal

def _trainer_sigusr1(_signum, _frame):
    print(f"[trainer pid={os.getpid()}] SIGUSR1 received — setting shutdown flag", flush=True)
    from scheduler.cooperative import _flag
    _flag.set()

_signal.signal(_signal.SIGUSR1, _trainer_sigusr1)
from datetime import datetime
from .adaptive_batching import AdaptiveBatchSizer
from utils.web_dataloader import WebDataloader
from utils.mixture_dataloader import WebDataloaderMixture

_torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert _torch_version >= (2, 4), f"PyTorch >= 2.4 required (got {torch.__version__}); needed for DTensor-aware clip_grad_norm_ and FSDP2."

torch.backends.cudnn.benchmark = True

DEBUG = False
# This slows down the training a lot ...
VALIDATION_DEBUG = False

BETA_1 = 0.90
BETA_2 = 0.95


@dataclass
class IntervalMetrics:
    sum_loss: float = 0.0
    sum_accuracy: float = 0.0
    count_rows: int = 0
    step_count: int = 0

    def update(self, loss, accuracy, rows):
        self.sum_loss += loss
        self.sum_accuracy += accuracy
        self.count_rows += rows
        self.step_count += 1

    def compute(self):
        avg_loss = self.sum_loss / max(self.step_count, 1)
        acc_pct = (self.sum_accuracy / self.count_rows * 100) if self.count_rows > 0 else 0
        return avg_loss, acc_pct

    def reset(self):
        self.sum_loss = 0.0
        self.sum_accuracy = 0.0
        self.count_rows = 0
        self.step_count = 0


class ModelStateSaver:
    def __init__(self, name):
        self.name = name

    def save(self, model, optimizer, epoch, loss):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(checkpoint, self.get_file_path())
        torch.save(checkpoint, self.get_file_path_epoch(epoch))
        tqdm.write("save checkpoint")

    def load_model_state(self, model: torch.nn.Module):
        state = torch.load(self.get_file_path())
        model.load_state_dict(state["model_state_dict"])
        return model

    def get_file_path(self):
        dir_name = os.path.join(os.path.dirname(__file__), self.name)
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(dir_name, "checkpoint.pth")

    def get_file_path_epoch(self, epoch):
        dir_name = os.path.join(os.path.dirname(__file__), self.name)
        os.makedirs(dir_name, exist_ok=True)
        return os.path.join(dir_name, f"{epoch}_checkpoint.pth")


class TrainingTimer:
    def __init__(self, minutes):
        self.start = time.time()
        self.minutes = minutes

    def done(self):
        return (time.time() - self.start) > self.minutes * 60

    def reset(self):
        self.start = time.time()


@dataclass
class EpochData:
    model: torch.nn.Module


@dataclass
class BatchData:
    model: torch.nn.Module


class DistributedStrategy(Enum):
    NONE = "none"
    DDP = "ddp"
    FSDP = "fsdp"

    @staticmethod
    def from_env() -> "DistributedStrategy":
        val = os.environ.get("PARALLEL_STRATEGY", "NONE").upper()
        return DistributedStrategy[val] if val in DistributedStrategy.__members__ else DistributedStrategy.NONE


@dataclass
class TrainingOptions:
    batch_size: Optional[int] = None
    epochs: int = 100
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    epoch_callback: Optional[Callable[[EpochData], None]] = None
    batch_callback: Optional[Callable[[BatchData], None]] = None
    training_timeout_minutes: Optional[int] = None
    # Optimizer configuration
    lr_scheduler: Optional[Scheduler] = None
    optimizer: BaseOptimizerConfig = field(default_factory=AdamConfig)
    # accumulation_steps
    accumulation_steps: int = 1
    record_interval_steps: int = 0
    val_interval_steps: int = 0
    val_max_batches: int = 50
    val_loader: Optional[Union["WebDataloader", "WebDataloaderMixture"]] = None
    # misc
    enable_checkpoints: bool = False
    checkpoint_tag: Optional[str] = None
    enable_metrics: bool = False
    distributed_strategy: DistributedStrategy = field(default_factory=DistributedStrategy.from_env)
    rho_loss: Optional["RhoLossConfig"] = None

    @property
    def sampling_timeout_minutes(self):
        if self.training_timeout_minutes is None:
            return None
        return self.training_timeout_minutes  # * 3

    # Additional metadata
    metadata: TrainingMetadata = field(default_factory=TrainingMetadata)


def debug_print(*args):
    if DEBUG:
        print(*args)


def create_config(vocab_size, padding_index, sequence_length):
    return Config(
        sequence_length=sequence_length,
        dim_embeddings=32,
        num_attention_heads=4,
        num_transformer_layers=4,
        padding_index=padding_index,
        vocab_size=vocab_size,
        transformer_layer=TransformerLayerType.GPT2,
    )


def default_batch_adapter(batch):
    return batch["x_tokens"], batch["y_tokens"]


def batch_iterator(dataset, adapter=default_batch_adapter):
    for batch in dataset:
        yield adapter(batch)


class BaseTrainer(ABC):
    def __init__(
        self,
        optimizer: Optional[BaseOptimizerConfig],
        lr_scheduler: Optional[Scheduler] = None,
    ):
        self.start = time.time()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_batch_num = 0
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_tracker = MetricsTracker(run_id=self._run_id, enabled=False)
        self.checkpoints_tracker = None
        self.sizer = AdaptiveBatchSizer(
            initial_batch=None,
            target_utilization=0.75,
            safety_margin=0.15,
            window_size=128,
        )
        self.static_shapes = os.environ.get("TRAINER_STATIC_SHAPES") == "1"
        # Do a checkpoint each hour.
        self.start = time.time()
        self.checkpoint_interval = 60 * 60
        self.min_checkpoint_seconds = float(os.environ.get("MIN_CHECKPOINT_MINUTES", "5")) * 60
        self.last_checkpoint = time.time()
        self._val_iter = None
        self._rho_selector = None
        self.batch_adapter = default_batch_adapter

    def _maybe_init_rho_loss(self, training_options: "TrainingOptions", main_model=None, dtype=None):
        if self._rho_selector is not None:
            return
        cfg = training_options.rho_loss
        if cfg is None or cfg.ratio >= 1.0:
            return
        from .rho_loss import RhoLossSelector
        self._rho_selector = RhoLossSelector.build(
            cfg, main_model if main_model is not None else self.model, training_options.device, dtype=dtype,
        )
        self.log(f"RHO-Loss enabled: mode={self._rho_selector.mode} ratio={cfg.ratio}")

    def _run_validation(self, model, objective, training_options: TrainingOptions):
        val_loader = training_options.val_loader
        if val_loader is None:
            return
        was_training = model.training
        model.eval()
        device = training_options.device
        sum_loss = torch.zeros((), dtype=torch.float64, device=device)
        sum_correct = torch.zeros((), dtype=torch.float64, device=device)
        sum_tokens = torch.zeros((), dtype=torch.float64, device=device)
        sum_batches = torch.zeros((), dtype=torch.float64, device=device)
        try:
            with torch.no_grad():
                for _ in range(max(1, training_options.val_max_batches)):
                    if self._val_iter is None:
                        self._val_iter = iter(val_loader)
                    try:
                        batch = next(self._val_iter)
                    except StopIteration:
                        self._val_iter = iter(val_loader)
                        try:
                            batch = next(self._val_iter)
                        except StopIteration:
                            break
                    X, y = self.batch_adapter(batch)
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    y_pred = model(X)
                    loss = objective.forward(y_pred, y)
                    correct, total = objective.evaluator(y_pred, y)
                    sum_loss += loss.detach().double()
                    sum_correct += correct.detach().double()
                    sum_tokens += total.detach().double()
                    sum_batches += 1.0
            if dist.is_initialized():
                stacked = torch.stack([sum_loss, sum_correct, sum_tokens, sum_batches])
                dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                sum_loss, sum_correct, sum_tokens, sum_batches = stacked.tolist()
            else:
                sum_loss = sum_loss.item()
                sum_correct = sum_correct.item()
                sum_tokens = sum_tokens.item()
                sum_batches = sum_batches.item()
            avg_loss = sum_loss / max(sum_batches, 1.0)
            acc_pct = (sum_correct / sum_tokens * 100.0) if sum_tokens > 0 else 0.0
            training_options.metadata.plots.record_val(
                loss=avg_loss, accuracy=acc_pct, train_step=self.total_batch_num
            )
        finally:
            if was_training:
                model.train()

    def train(
        self,
        model,
        objective,
        loader: WebDataloader,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        if dist.is_initialized() and dist.get_rank() != 0:
            progress = lambda x: x  # noqa: E731

        rank = dist.get_rank() if dist.is_initialized() else 0
        if training_options.enable_metrics:
            self.metrics_tracker = MetricsTracker(
                run_id=self._run_id,
                dataset_name=loader.name,
                rank=rank,
            )
        if training_options.enable_checkpoints and self.checkpoints_tracker is None:
            self.checkpoints_tracker = StorageBoxCheckpoint(
                run_id=self._run_id,
                host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
                username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
                password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
            )

        sum_loss = 0
        sum_accuracy = 0
        count_rows = 0
        epoch_batch_count = 0
        interval = IntervalMetrics()
        progress = progress(loader)
        model.train()
        has_tqdm_loader = isinstance(progress, tqdm)
        self.metrics_tracker.dataset_name = loader.name
        iterator = batch_iterator(progress, self.batch_adapter)

        if training_options.metadata.epoch == loader.epoch:
            sum_loss = training_options.metadata.sum_loss
            sum_accuracy = training_options.metadata.sum_accuracy
            count_rows = training_options.metadata.count_rows
            self.total_batch_num = training_options.metadata.total_batch_num
            epoch_batch_count = training_options.metadata.epoch_batch_count

        while True:
            try:
                X, y = next(iterator)
                local_has_data = 1
            except StopIteration:
                X, y = None, None
                local_has_data = 0
            if dist.is_initialized():
                flag = torch.tensor(local_has_data, device=training_options.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if flag.item() == 0:
                    break
            elif not local_has_data:
                break
            loss, accuracy, rows = self.forward(
                model, objective, X, y, training_options
            )

            if has_tqdm_loader:
                acc_pct = (sum_accuracy / count_rows * 100) if count_rows > 0 else 0
                avg_loss = sum_loss / max(epoch_batch_count, 1)
                postfix = {
                    "loss": f"{avg_loss:.2f}",
                    "acc": f"{acc_pct:.1f}%",
                    "batch": f"{loader._batches_consumed}/{loader.total_batches}",
                    "time": f"{self.trained_minutes}/{training_options.training_timeout_minutes}m",
                    "q": loader.batch_queue.qsize(),
                }
                if loader._failed_fetches > 0:
                    postfix["failed_fetches"] = loader._failed_fetches
                progress.set_postfix(postfix)

            if (
                time.time() - self.last_checkpoint > self.checkpoint_interval
                and training_options.enable_checkpoints
            ):
                self.checkpoint(
                    training_options, model, sum_loss, sum_accuracy, count_rows, epoch_batch_count
                )

            if not self.static_shapes and self.sizer.record_step(training_options.device):
                # Update the batch size if needed
                training_options.batch_size = self.sizer.get_batch_size()
                loader.set_batch_size(self.sizer.get_batch_size())
                if isinstance(progress, tqdm):
                    progress.total = len(loader)

            sum_loss += loss.item()
            sum_accuracy += accuracy.item()
            count_rows += rows.item()
            epoch_batch_count += 1

            if training_options.record_interval_steps > 0:
                interval.update(loss.item(), accuracy.item(), rows.item())
                if interval.step_count % training_options.record_interval_steps == 0:
                    avg_loss, acc_pct = interval.compute()
                    training_options.metadata.plots.record_step(loss=avg_loss, accuracy=acc_pct)
                    interval.reset()

            if (
                training_options.val_interval_steps > 0
                and training_options.val_loader is not None
                and self.total_batch_num > 0
                and self.total_batch_num % training_options.val_interval_steps == 0
            ):
                self._run_validation(model, objective, training_options)

            training_options.metadata.epoch = loader.epoch
            training_options.metadata.batches_consumed = loader._batches_consumed
            training_options.metadata.sum_accuracy = sum_accuracy
            training_options.metadata.sum_loss = sum_loss
            training_options.metadata.count_rows = count_rows
            training_options.metadata.total_batch_num = self.total_batch_num
            training_options.metadata.epoch_batch_count = epoch_batch_count

            timeout = int(self.has_timeout(training_options) or shutdown_requested())
            if dist.is_initialized():
                t = torch.tensor(timeout, device=training_options.device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                timeout = t.item()
            if timeout:
                self.log("Hit timeout or shutdown")
                break

        return (
            sum_accuracy,
            sum_loss,
            count_rows,
            epoch_batch_count,
        )

    def checkpoint(
        self, training_options: TrainingOptions, model, sum_loss, sum_accuracy, sum_rows, batch_count=1
    ):
        elapsed = time.time() - self.start
        if elapsed < self.min_checkpoint_seconds:
            self.log(f"skip checkpoint: trained {elapsed/60:.1f}m < {self.min_checkpoint_seconds/60:.0f}m")
            return
        stats = Stats(
            loss_average=(sum_loss / max(batch_count, 1)),
            accuracy_pct=(sum_accuracy / max(sum_rows, 1) * 100),
            runtime_seconds=time.time() - self.start,
            steps=self.total_batch_num,
            dataset=self.metrics_tracker.dataset_name,
            metadata=training_options.metadata,
        )
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            StateDictOptions,
        )
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=opts)
        optim_sd = get_optimizer_state_dict(model, self.optimizer, options=opts)
        underlying = model.module if hasattr(model, "module") else model
        config = underlying.config
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            self.checkpoints_tracker.checkpoint(
                model_sd, optim_sd, config, stats,
            )
            if training_options.checkpoint_tag is not None:
                self.checkpoints_tracker.tag(
                    tag_name=training_options.checkpoint_tag, stats=stats
                )
        if dist.is_initialized():
            dist.barrier()
        self.last_checkpoint = time.time()

    def forward(self, model, objective, X, y, training_options: TrainingOptions):
        X, y = (
            X.to(training_options.device, non_blocking=True),
            y.to(
                training_options.device,
                non_blocking=True,
            ),
        )
        if self._rho_selector is not None:
            X, y = self._rho_selector.select(X, y, model, objective)
        y_predicted = model(X)
        loss = objective(y_predicted, y)

        (loss / training_options.accumulation_steps).backward()
        if (self.total_batch_num + 1) % training_options.accumulation_steps == 0:
            max_grad_norm = getattr(training_options.optimizer, "max_grad_norm", 0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self._rho_selector is not None:
                self._rho_selector.update(model)
        self.total_batch_num += 1
        # Report metrics.
        if objective.has_evaluator:
            (accuracy, rows) = objective.evaluator(y_predicted, y)
            return loss, accuracy, rows
        return loss, 0, 0

    def has_timeout(self, training_options: TrainingOptions):
        if training_options.training_timeout_minutes is None:
            return False
        training_time = self.trained_minutes
        return training_time >= training_options.training_timeout_minutes

    @property
    def is_main_rank(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def log(self, msg):
        if self.is_main_rank:
            tqdm.write(msg)

    @property
    def trained_minutes(self):
        return (time.time() - self.start) // 60


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        objective: BaseObjective,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Scheduler] = None,
        name=None,
    ):
        super().__init__(optimizer, lr_scheduler)
        self.name = name
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.state_saver = ModelStateSaver(name) if name is not None else None

    def apply_distributed_strategy(self, training_options: TrainingOptions, is_fsdp: bool):
        blocks = []
        if training_options.distributed_strategy == DistributedStrategy.DDP and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[dist.get_rank()])
        elif is_fsdp:
            mp_dtype = best_autocast_dtype()
            bf16 = mp_dtype == torch.bfloat16
            blocks = [
                layer
                for mod in self.model.modules()
                if isinstance(mod, torch.nn.ModuleList)
                for layer in mod
            ]
            if _FSDP2_AVAILABLE:
                mp_policy = FSDP2MixedPrecision(param_dtype=mp_dtype if bf16 else None, reduce_dtype=mp_dtype)
                for layer in blocks:
                    fully_shard(layer, mp_policy=mp_policy)
                fully_shard(self.model, mp_policy=mp_policy)
                self.model.to(training_options.device)
                self.optimizer = training_options.optimizer.create_optimizer(self.model.parameters())
                if self.lr_scheduler is not None:
                    self.lr_scheduler.create_scheduler(self.optimizer)
            else:
                from torch.distributed.fsdp.wrap import ModuleWrapPolicy
                mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
                wrap_policy = ModuleWrapPolicy({type(layer) for layer in blocks}) if blocks else None
                self.model = FSDP(self.model, device_id=dist.get_rank(), mixed_precision=mp_policy, use_orig_params=True, auto_wrap_policy=wrap_policy)
        return blocks

    def train(
        self,
        dataloader: WebDataloader,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self.log(f"Training on {training_options.device}")
        apply_runtime_optimizations()
        is_fsdp = (
            training_options.distributed_strategy == DistributedStrategy.FSDP
            and dist.is_initialized()
        )
        if not is_fsdp:
            self.model.to(training_options.device)
        self._maybe_init_rho_loss(training_options)
        blocks = self.apply_distributed_strategy(training_options, is_fsdp)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(training_options.device)
        resume = getattr(self, "resume", None)
        if resume is not None:
            opt_state = resume.get("optimizer_state")
            if opt_state is not None:
                from torch.distributed.checkpoint.state_dict import (
                    set_optimizer_state_dict,
                    StateDictOptions,
                )
                set_optimizer_state_dict(
                    self.model, self.optimizer, opt_state,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )
                for st in self.optimizer.state.values():
                    for k, v in st.items():
                        if isinstance(v, torch.Tensor):
                            st[k] = v.to(training_options.device)
            step = int(resume.get("step", 0))
            self.total_batch_num = step
            training_options.metadata.total_batch_num = step
            self.log(f"resumed optimizer + step from {step}")
            self.resume = None
        if is_fsdp:
            resident_mb = torch.cuda.memory_allocated(training_options.device) / 1024**2
            self.log(f"[fsdp] world_size={dist.get_world_size()} fsdp2={_FSDP2_AVAILABLE} blocks={len(blocks)} resident_after_shard={resident_mb:.0f}MB")
        self.sizer.current_batch = training_options.batch_size
        self.log("Starting to train now!")
        dataloader.load_state_dict(
            batches_consumed=training_options.metadata.batches_consumed,
            epoch=training_options.metadata.epoch,
        )

        for epoch in range(training_options.epochs):
            sum_epoch_accuracy, sum_epoch_loss, sum_epoch_rows, epoch_batch_count = super().train(
                self.model, self.objective, dataloader, training_options, progress
            )
            dataloader.set_epoch(epoch + 1)
            accuracy_pct = sum_epoch_accuracy / max(sum_epoch_rows, 1) * 100
            avg_loss = sum_epoch_loss / max(epoch_batch_count, 1)
            self.log(f"Epoch {epoch} | acc={accuracy_pct:.2f}% loss={avg_loss:.4f}")
            timeout = int(self.has_timeout(training_options) or shutdown_requested())
            if dist.is_initialized():
                t = torch.tensor(timeout, device=training_options.device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                timeout = t.item()
            if timeout:
                self.log("Hit timeout or shutdown")
                break
            training_options.metadata.plots.record_epoch(
                loss=avg_loss, accuracy=accuracy_pct
            )
        if training_options.enable_checkpoints:
            self.log("Storing checkpoints ...")
            self.checkpoint(
                training_options,
                self.model,
                sum_epoch_loss,
                sum_epoch_accuracy,
                sum_epoch_rows,
                epoch_batch_count,
            )
            self.checkpoints_tracker.flush()

        return (
            training_options.metadata.plots.accuracies,
            training_options.metadata.plots.losses,
            training_options.metadata.plots.step_accuracies,
            training_options.metadata.plots.step_losses,
            training_options.metadata.plots.epoch_at_step,
            training_options.metadata.plots.val_step_accuracies,
            training_options.metadata.plots.val_step_losses,
            training_options.metadata.plots.val_at_step,
        )


def check_bf16_support():
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    if supports_bf16():
        print(f"✅ BF16 supported (Compute Capability: {major}.{minor})")
        return True
    else:
        print(
            f"❌ BF16 not supported (Compute Capability: {major}.{minor}, need >= 8.0)"
        )
        return False


def apply_runtime_optimizations():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")


def best_autocast_dtype(device=None):
    if supports_bf16(device):
        return torch.bfloat16
    if supports_amp(device):
        return torch.float16
    return torch.float32


class GradScalerTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        objective: BaseObjective,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Scheduler] = None,
        name=None,
    ):
        super().__init__(model, objective, optimizer, lr_scheduler, name)
        self._original_model = model
        self.scaler = GradScaler("cuda")
        self.type = best_autocast_dtype()
        self.last_time = None
        self._training_options: Optional[TrainingOptions] = None

    @property
    def use_fsdp(self) -> bool:
        return (
            self._training_options is not None
            and self._training_options.distributed_strategy == DistributedStrategy.FSDP
            and dist.is_initialized()
        )

    def train(
        self,
        dataset,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self._training_options = training_options
        if self.use_fsdp:
            self._original_model.to(training_options.device)
        else:
            self._original_model.to(training_options.device).to(self.type)
        il_dtype = None if self.use_fsdp else self.type
        self._maybe_init_rho_loss(training_options, main_model=self._original_model, dtype=il_dtype)
        apply_runtime_optimizations()
        self.maybe_compile(training_options)
        return super().train(dataset, training_options, progress)

    def maybe_compile(self, training_options: TrainingOptions):
        can_compile = shutil.which("cc") or shutil.which("gcc")
        fsdp_compile_disabled = self.use_fsdp and os.environ.get("DISABLE_FSDP_COMPILE") == "1"
        if can_compile and torch.cuda.is_available() and not fsdp_compile_disabled:
            if supports_torch_compile(training_options.device):
                try:
                    compiled = torch.compile(self.model, dynamic=not self.static_shapes)
                    warm_batch = training_options.batch_size if self.static_shapes else 1
                    dummy = torch.zeros(warm_batch, self._original_model.config.sequence_length, dtype=torch.long, device=training_options.device)
                    with torch.no_grad():
                        compiled(dummy)
                    del dummy
                    self.model = compiled
                except Exception as e:
                    tqdm.write(f"Skipping torch.compile: {e}")
                    del compiled
                    torch.cuda.empty_cache()
                    self.model = self._original_model
            else:
                cap = torch.cuda.get_device_capability(training_options.device)
                tqdm.write(f"Skipping torch.compile: CUDA Capability {cap[0]}.{cap[1]} < 7.0")
        elif not can_compile:
            tqdm.write("Skipping torch.compile: no C compiler found")

    def forward(
        self, model, objective, X, y, training_options: TrainingOptions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        now = time.time()
        with autocast("cuda", dtype=self.type):
            with self.metrics_tracker.span("to_device"):
                X, y = (
                    X.to(training_options.device, non_blocking=True),
                    y.to(
                        training_options.device,
                        non_blocking=True,
                    ),
                )
            if self._rho_selector is not None:
                with self.metrics_tracker.span("rho_select"):
                    X, y = self._rho_selector.select(X, y, model, objective)
            with self.metrics_tracker.span("forward"):
                y_predicted = model(X)
            with self.metrics_tracker.span("objective"):
                loss: torch.Tensor = objective(
                    y_predicted,
                    y,
                )
        with self.metrics_tracker.span("optimize"):
            self.scaler.scale(loss / training_options.accumulation_steps).backward()
            if (
                self.total_batch_num + 1
            ) % training_options.accumulation_steps == 0:
                max_grad_norm = getattr(training_options.optimizer, "max_grad_norm", 0)
                if max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= prev_scale
                self.optimizer.zero_grad(set_to_none=True)
                if stepped and self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                if self._rho_selector is not None:
                    self._rho_selector.update(self._original_model)

        self.total_batch_num += 1

        if objective.has_evaluator:
            with self.metrics_tracker.span("evaluator"):
                (accuracy, rows) = objective.evaluator(y_predicted, y)

            if dist.is_initialized():
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"non-finite loss: {loss.item()}")
                accuracy_tensor = accuracy.detach().float().reshape(1)
                rows_tensor = rows.detach().float().reshape(1)
                loss_tensor = loss.detach().float().reshape(1)

                dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(rows_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

                accuracy = accuracy_tensor
                rows = rows_tensor
                loss = loss_tensor / dist.get_world_size()

            metrics = {
                "loss": loss,
                "accuracy": accuracy / rows * 100,
            }
            if self.last_time is not None:
                elapsed = now - self.last_time
                metrics["samples_per_second"] = rows / elapsed
                metrics["batches_per_second"] = 1 / (now - self.last_time)
            self.metrics_tracker.log(**metrics)
            self.last_time = now
            return loss, accuracy, rows
        return loss, 0, 0
