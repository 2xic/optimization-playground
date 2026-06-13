import torch
import torch.nn as nn

from training.model import Model


class ClassificationHeadModel(nn.Module):
    def __init__(self, base: Model, num_classes: int = 1):
        super().__init__()
        self.base = base
        self.config = base.config
        self.head = nn.Linear(self.config.dim_embeddings, num_classes)
        self.padding_index = self.config.padding_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.base.get_hidden_states(x)
        mask = (x != self.padding_index).to(hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        logits = self.head(pooled)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits


class TripletEmbeddingModel(nn.Module):
    def __init__(self, base: Model):
        super().__init__()
        self.base = base
        self.config = base.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.embed(x)
