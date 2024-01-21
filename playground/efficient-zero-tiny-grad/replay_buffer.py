from dataclasses import dataclass
from torch import Tensor
import random
from typing import List

@dataclass
class Predictions:
    # State = State
    state: Tensor
    action: int 
    # Value out
    environment_reward: float
    next_state: Tensor
    # policy 
    state_distribution: Tensor

class ReplayBuffer:
    def __init__(self) -> None:
        self.entries: List[Predictions] = []
        self.max_entries = 1_00

    def add_to_entries(self, prediction: Predictions):
        self.entries.append(prediction)
        if self.max_entries < len(self.entries): 
            self.entries = random.sample(self.entries, self.max_entries)
