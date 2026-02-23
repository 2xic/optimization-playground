from .model import Config, TransformerLayerType
import torch
from typing import Callable, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
import os
import time
from utils.performance_benchmarker import Timer
from .objectives import BaseObjective
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from abc import ABC
from .optimizer import BaseOptimizerConfig, AdamConfig, Scheduler
from dataclasses import dataclass, field
from utils.metrics import MetricsTracker
from utils.checkpoints import StorageBoxCheckpoint, Stats, TrainingMetadata
from datetime import datetime
from .adaptive_batching import AdaptiveBatchSizer
from utils.web_dataloader import WebDataloader
from typing import List
import shutil

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
    optimizer: BaseOptimizerConfig = field(default_factory=lambda: AdamConfig())
    # accumulation_steps
    accumulation_steps: int = 1
    record_interval_steps: int = 0
    # misc
    enable_checkpoints: bool = False
    checkpoint_tag: Optional[str] = None

    @property
    def sampling_timeout_minutes(self):
        if self.training_timeout_minutes is None:
            return None
        return self.training_timeout_minutes  # * 3

    # Additional metadata
    metadata: TrainingMetadata = field(default_factory=lambda: TrainingMetadata())


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


def batch_iterator(dataset):
    for batch in dataset:
        yield batch["x_tokens"], batch["y_tokens"]


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
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_tracker = MetricsTracker(
            run_id=run_id,
            dataset_name=None,
            rank=dist.get_rank() if dist.is_initialized() else 0,
        )
        self.checkpoints_tracker = StorageBoxCheckpoint(
            run_id=run_id,
            host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
            username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
            password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
        )
        self.sizer = AdaptiveBatchSizer(
            initial_batch=None,
            target_utilization=0.90,
            safety_margin=0.10,
            window_size=128,
        )
        # Do a checkpoint each hour.
        self.start = time.time()
        self.checkpoint_interval = 60 * 60
        self.last_checkpoint = time.time()

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

        sum_loss = 0
        sum_accuracy = 0
        count_rows = 0
        epoch_batch_count = 0
        interval = IntervalMetrics()
        progress = progress(loader)
        model.train()
        has_tqdm_loader = isinstance(progress, tqdm)
        self.metrics_tracker.dataset_name = loader.name
        iterator = batch_iterator(progress)

        if training_options.metadata.epoch == loader.epoch:
            sum_loss = training_options.metadata.sum_loss
            sum_accuracy = training_options.metadata.sum_accuracy
            count_rows = training_options.metadata.count_rows
            self.total_batch_num = training_options.metadata.total_batch_num
            epoch_batch_count = training_options.metadata.epoch_batch_count

        for _, (X, y) in enumerate(iterator):
            # print("index", index, (X.shape, y.shape))
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

            if self.sizer.record_step(training_options.device):
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

            training_options.metadata.epoch = loader.epoch
            training_options.metadata.batches_consumed = loader._batches_consumed
            training_options.metadata.sum_accuracy = sum_accuracy
            training_options.metadata.sum_loss = sum_loss
            training_options.metadata.count_rows = count_rows
            training_options.metadata.total_batch_num = self.total_batch_num
            training_options.metadata.epoch_batch_count = epoch_batch_count

            if self.has_timeout(training_options):
                self.log("Hit timeout")
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
        stats = Stats(
            loss_average=(sum_loss / max(batch_count, 1)),
            accuracy_pct=(sum_accuracy / max(sum_rows, 1) * 100),
            runtime_seconds=time.time() - self.start,
            steps=self.total_batch_num,
            dataset=self.metrics_tracker.dataset_name,
            metadata=training_options.metadata,
        )
        self.checkpoints_tracker.checkpoint(
            model,
            self.optimizer,
            model.config,
            stats,
        )
        if training_options.checkpoint_tag is not None:
            self.checkpoints_tracker.tag(
                tag_name=training_options.checkpoint_tag, stats=stats
            )
        self.last_checkpoint = time.time()

    def forward(self, model, objective, X, y, training_options: TrainingOptions):
        X, y = (
            X.to(training_options.device, non_blocking=True),
            y.to(
                training_options.device,
                non_blocking=True,
            ),
        )
        y_predicted = model(X)
        loss = objective(y_predicted, y)

        loss.backward()
        if (self.total_batch_num + 1) % training_options.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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

    def train(
        self,
        dataloader: WebDataloader,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self.log(f"Training on {training_options.device}")
        self.model.to(training_options.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(training_options.device)
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
            training_options.metadata.plots.record_epoch(
                loss=avg_loss, accuracy=accuracy_pct
            )
            if self.has_timeout(training_options):
                self.log("Hit timeout")
                break
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
        )


def check_bf16_support():
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    # Ampere (8.0) or newer supports BF16
    if major >= 8:
        print(f"✅ BF16 supported (Compute Capability: {major}.{minor})")
        return True
    else:
        print(
            f"❌ BF16 not supported (Compute Capability: {major}.{minor}, need >= 8.0)"
        )
        return False


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
        self.type = torch.bfloat16 if check_bf16_support() else torch.float16
        self.last_time = None

    def train(
        self,
        dataset,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self._original_model.to(training_options.device)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.set_float32_matmul_precision("high")
        can_compile = shutil.which("cc") or shutil.which("gcc")
        if can_compile and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(training_options.device)
            if capability[0] >= 7:
                try:
                    compiled = torch.compile(self.model, dynamic=True)
                    dummy = torch.zeros(1, self._original_model.config.sequence_length, dtype=torch.long, device=training_options.device)
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
                tqdm.write(f"Skipping torch.compile: CUDA Capability {capability[0]}.{capability[1]} < 7.0")
        elif not can_compile:
            tqdm.write("Skipping torch.compile: no C compiler found")
        self.optimizer.fused = True
        return super().train(dataset, training_options, progress)

    def forward(
        self, model, objective, X, y, training_options: TrainingOptions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            with self.metrics_tracker.span("forward"):
                y_predicted = model(X)
            with self.metrics_tracker.span("objective"):
                loss: torch.Tensor = objective(
                    y_predicted,
                    y,
                )
            with self.metrics_tracker.span("optimize"):
                self.scaler.scale(loss).backward()
                if (
                    self.total_batch_num + 1
                ) % training_options.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.total_batch_num += 1

            if objective.has_evaluator:
                with self.metrics_tracker.span("evaluator"):
                    (accuracy, rows) = objective.evaluator(y_predicted, y)

                if dist.is_initialized():
                    accuracy_tensor = torch.tensor(
                        [accuracy], device=training_options.device, dtype=torch.float32
                    )
                    rows_tensor = torch.tensor(
                        [rows], device=training_options.device, dtype=torch.float32
                    )

                    dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rows_tensor, op=dist.ReduceOp.SUM)

                    accuracy = accuracy_tensor
                    rows = rows_tensor

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
