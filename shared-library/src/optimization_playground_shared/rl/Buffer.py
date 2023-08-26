from dataclasses import dataclass
import torch
from typing import List

@dataclass
class EnvStateActionPairs:
    state: torch.Tensor
    next_state: torch.Tensor
    action: int
    reward: float

class Buffer:
    def __init__(self) -> None:
        self.entries: List[EnvStateActionPairs] = []

    def add(self, entry: EnvStateActionPairs):
        self.entries.append(entry)

    def extend(self, buffer):
        self.entries += buffer.entries
        return self
