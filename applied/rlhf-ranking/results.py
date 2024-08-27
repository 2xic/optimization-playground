import torch
from dataclasses import dataclass

@dataclass
class Results:
    item_id: int
    item_score: float
    item_tensor: torch.Tensor

@dataclass
class Input:
    item_id: int
    item_tensor: torch.Tensor

