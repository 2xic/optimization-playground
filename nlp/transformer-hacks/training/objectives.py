import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Callable


class BaseObjective(nn.Module, ABC):
    @abstractmethod
    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        pass

    @property
    @abstractmethod
    def has_evaluator(self) -> bool:
        pass


class NextTokenPrediction(BaseObjective):
    def __init__(
        self,
        padding_index: int,
        vocab_size: int,
        sampler: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ):
        super().__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.sampler = sampler

    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            y_predicted.view(-1, self.vocab_size),
            y.view(-1),
            ignore_index=self.padding_index,
        )

    @property
    def has_evaluator(self):
        return self.sampler is not None

    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        y_sample_next = self.sampler(y_predicted[:, -1, :])
        y_next = y[:, -1]

        assert y_sample_next.shape == y_next.shape
        accuracy = (y_sample_next == y_next).sum()
        rows = y_next.shape.numel()

        return (accuracy, rows)
