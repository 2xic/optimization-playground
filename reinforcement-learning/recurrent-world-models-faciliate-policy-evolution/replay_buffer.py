import torch
import random

class ReplayBuffer:
    def __init__(self, sequential=False) -> None:
        self.items = None
        self.max_size = 256
        self.index = 0
        self.sequential = sequential

    def add(self, item):
        if self.items is None:
            self.items = item
        elif self.items.shape[0] < self.max_size:
            self.items = torch.concat(
                (self.items,
                item),
                dim=0
            )
        elif self.sequential:
            self.items[self.index] = item[0]
            self.index = (self.index + 1) % self.max_size
        elif random.randint(0, 3) == 2:
            self.items[self.index] = item[0]
            self.index = (self.index + 1) % self.max_size
