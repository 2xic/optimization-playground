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
        flat_pred = y_predicted.view(-1, y_predicted.shape[-1])[..., :self.vocab_size]
        flat_y = y.view(-1)
        chunk_size = 2048
        if flat_pred.shape[0] <= chunk_size:
            return torch.nn.functional.cross_entropy(flat_pred.float(), flat_y, ignore_index=self.padding_index)
        total_loss = y_predicted.new_zeros((), dtype=torch.float32)
        total_valid = 0
        for i in range(0, flat_pred.shape[0], chunk_size):
            chunk_pred = flat_pred[i:i + chunk_size].float()
            chunk_y = flat_y[i:i + chunk_size]
            valid = int((chunk_y != self.padding_index).sum())
            if valid > 0:
                total_loss = total_loss + torch.nn.functional.cross_entropy(
                    chunk_pred, chunk_y, ignore_index=self.padding_index, reduction='sum'
                )
                total_valid += valid
        return total_loss / max(1, total_valid)

    @property
    def has_evaluator(self):
        return self.sampler is not None

    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        """
        y_sample_next = self.sampler(y_predicted[:, -1, :])
        y_next = y[:, -1]

        assert y_sample_next.shape == y_next.shape
        accuracy = (y_sample_next == y_next).sum()
        rows = y_next.shape.numel()

        return (accuracy, rows)
        """
        y_pred_flat = y_predicted.view(-1, y_predicted.shape[-1])[..., :self.vocab_size]
        y_flat = y.view(-1)

        y_sample = self.sampler(y_pred_flat)
        valid_mask = y_flat != self.padding_index

        correct_predictions = (y_sample == y_flat) & valid_mask
        accuracy = correct_predictions.sum()
        total_valid_tokens = valid_mask.sum()
        return accuracy, total_valid_tokens
