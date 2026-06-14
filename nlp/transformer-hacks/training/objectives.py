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

    def per_sequence_loss(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("per_sequence_loss not implemented for this objective")


class NextTokenPrediction(BaseObjective):
    def __init__(
        self,
        padding_index: int,
        vocab_size: int,
        sampler: Optional[Callable[[torch.Tensor], torch.Tensor]],
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.sampler = sampler
        self.label_smoothing = label_smoothing

    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        flat_pred = y_predicted.view(-1, y_predicted.shape[-1])[..., :self.vocab_size]
        flat_y = y.view(-1)
        chunk_size = 2048
        if flat_pred.shape[0] <= chunk_size:
            return torch.nn.functional.cross_entropy(flat_pred.float(), flat_y, ignore_index=self.padding_index, label_smoothing=self.label_smoothing)
        total_loss = y_predicted.new_zeros((), dtype=torch.float32)
        total_valid = 0
        for i in range(0, flat_pred.shape[0], chunk_size):
            chunk_pred = flat_pred[i:i + chunk_size].float()
            chunk_y = flat_y[i:i + chunk_size]
            valid = int((chunk_y != self.padding_index).sum())
            if valid > 0:
                total_loss = total_loss + torch.nn.functional.cross_entropy(
                    chunk_pred, chunk_y, ignore_index=self.padding_index, reduction='sum', label_smoothing=self.label_smoothing
                )
                total_valid += valid
        return total_loss / max(1, total_valid)

    @property
    def has_evaluator(self):
        return self.sampler is not None

    def per_sequence_loss(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = y_predicted[..., :self.vocab_size].float()
        B, T, V = logits.shape
        flat = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            y.reshape(-1),
            ignore_index=self.padding_index,
            reduction='none',
        ).reshape(B, T)
        mask = (y != self.padding_index).float()
        return (flat * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)

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


class BinaryFeedbackClassification(BaseObjective):
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def _flatten(self, y_predicted: torch.Tensor) -> torch.Tensor:
        if y_predicted.dim() > 1:
            y_predicted = y_predicted.squeeze(-1)
        return y_predicted.float()

    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self._flatten(y_predicted)
        target = y.float().view(-1)
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw
        )

    @property
    def has_evaluator(self):
        return True

    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        logits = self._flatten(y_predicted)
        target = y.float().view(-1)
        pred = (logits > 0).to(target.dtype)
        correct = (pred == target).sum()
        total = torch.tensor(target.numel(), device=logits.device)
        return correct, total


class MultiClassClassification(BaseObjective):
    def __init__(self, label_smoothing: float = 0.0, class_weights: Optional[list] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def _weight(self, logits: torch.Tensor) -> Optional[torch.Tensor]:
        if self.class_weights is None:
            return None
        return torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)

    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = y_predicted.float()
        target = y.long().view(-1)
        return torch.nn.functional.cross_entropy(
            logits, target,
            weight=self._weight(logits),
            label_smoothing=self.label_smoothing,
        )

    @property
    def has_evaluator(self):
        return True

    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        logits = y_predicted.float()
        target = y.long().view(-1)
        pred = logits.argmax(dim=-1)
        correct = (pred == target).sum()
        total = torch.tensor(target.numel(), device=logits.device)
        return correct, total


class TripletContrastive(BaseObjective):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def _split(self, y_predicted: torch.Tensor):
        b3, _ = y_predicted.shape
        assert b3 % 3 == 0, f"triplet expects 3*B rows, got {b3}"
        b = b3 // 3
        return y_predicted[:b], y_predicted[b:2 * b], y_predicted[2 * b:]

    def forward(self, y_predicted: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        anchor, positive, negative = self._split(y_predicted)
        return torch.nn.functional.triplet_margin_loss(
            anchor.float(), positive.float(), negative.float(), margin=self.margin
        )

    @property
    def has_evaluator(self):
        return True

    def evaluator(self, y_predicted: torch.Tensor, y: torch.Tensor):
        anchor, positive, negative = self._split(y_predicted)
        sim_pos = (anchor * positive).sum(-1)
        sim_neg = (anchor * negative).sum(-1)
        correct = (sim_pos > sim_neg).sum()
        total = torch.tensor(anchor.shape[0], device=anchor.device)
        return correct, total
