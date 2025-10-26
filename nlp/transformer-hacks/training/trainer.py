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
from .optimizer import Optimizer, AdamConfig, Scheduler
from dataclasses import dataclass, field

torch.backends.cudnn.benchmark = True

DEBUG = False
# This slows down the training a lot ...
VALIDATION_DEBUG = False

BETA_1 = 0.90
BETA_2 = 0.95


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
        print("save checkpoint")

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
    optimizer: Optimizer = field(default_factory=lambda: AdamConfig())

    @property
    def sampling_timeout_minutes(self):
        if self.training_timeout_minutes is None:
            return None
        return self.training_timeout_minutes * 3


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


def timer_iterator(dataset):
    with Timer("create_iterator"):
        iterator = iter(dataset)
    for _ in range(len(dataset)):
        try:
            with Timer("timer_iterator"):
                X, y = next(iterator)
            yield X, y
        except StopIteration:
            break


class BaseTrainer(ABC):
    def __init__(
        self, optimizer: Optional[Optimizer], lr_scheduler: Optional[Scheduler] = None
    ):
        self.start = time.time()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(
        self,
        model,
        objective,
        dataset,
        training_options,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        if dist.is_initialized() and dist.get_rank() != 0:
            progress = lambda x: x  # noqa: E731

        sum_loss = 0
        sum_accuracy = 0
        count_rows = 0
        progress = progress(dataset)
        has_tqdm_loader = isinstance(progress, tqdm)
        iterator = timer_iterator(progress)
        for _, (X, y) in enumerate(iterator):
            loss, accuracy, rows = self.forward(
                model, objective, X, y, training_options
            )

            if has_tqdm_loader:
                progress.set_description(
                    "Loss {loss}, Accuracy {acc}".format(
                        acc=(sum_accuracy / count_rows * 100) if count_rows > 0 else 0,
                        loss=sum_loss,
                    )
                )

            sum_loss += loss.item()
            sum_accuracy += accuracy.item()
            count_rows += rows.item()

            if self.has_timeout(training_options):
                print("Hit timeout")
                break
        return sum_accuracy / count_rows * 100 if count_rows > 0 else None, sum_loss

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

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Report metrics.
        if objective.has_evaluator:
            (accuracy, rows) = objective.evaluator(y_predicted, y)
            return loss, accuracy, rows
        return loss, 0, 0

    def has_timeout(self, training_options: TrainingOptions):
        if training_options.training_timeout_minutes is None:
            return False
        delta = time.time() - self.start
        return delta // 60 >= training_options.training_timeout_minutes


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
        dataset,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self.model.to(training_options.device)

        epochs_accuracy = []
        epochs_loss = []
        loader = dataset.iter(batch_size=training_options.batch_size)
        for epoch in range(training_options.epochs):
            epoch_accuracy, epoch_loss = super().train(
                self.model, self.objective, loader, training_options, progress
            )
            print(f"Epoch {epoch} done, accuracy {epoch_accuracy}, loss {epoch_loss}")
            epochs_accuracy.append(epoch_accuracy)
            epochs_loss.append(epoch_loss)
            # Check timeout
            if self.has_timeout(training_options):
                print("Hit timeout")
                break
        return epochs_accuracy, epochs_loss


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

    def train(
        self,
        dataset,
        training_options: TrainingOptions,
        progress=lambda x: tqdm(x, mininterval=1),
    ):
        self._original_model.to(training_options.device)
        self.model = self.model  # torch.compile(self.model, mode="reduce-overhead")
        # Try to enable the fuzed model.
        self.optimizer.fused = True
        return super().train(dataset, training_options, progress)

    def forward(
        self, model, objective, X, y, training_options: TrainingOptions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast("cuda", dtype=self.type):
            X, y = (
                X.to(training_options.device, non_blocking=True),
                y.to(
                    training_options.device,
                    non_blocking=True,
                ),
            )
            y_predicted = model(X)
            loss: torch.Tensor = objective(
                y_predicted,
                y,
            )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if objective.has_evaluator:
                (accuracy, rows) = objective.evaluator(y_predicted, y)
                return loss, accuracy, rows
            return loss, 0, 0
