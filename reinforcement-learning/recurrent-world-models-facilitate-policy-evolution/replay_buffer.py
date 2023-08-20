import torch
import random

class ReplayBuffer:
    def __init__(self, sequential=False, overflow=True, max_size=256) -> None:
        self.items = None
        self.max_size = max_size
        self.batch_size = max_size
        self.index = 0
        self.sequential = sequential
        self.overflow = overflow

    def add(self, item):
        if self.items is None:
            self.items = item
        elif not self.is_filled:
            self.items = torch.concat(
                (self.items,
                item),
                dim=0
            )
        elif self.overflow:
            if self.sequential:
                self.items[self.index] = item[0]
                self.index = (self.index + 1) % self.max_size
            elif random.randint(0, 3) == 2:
                self.items[self.index] = item[0]
                self.index = (self.index + 1) % self.max_size

    def reset(self):
        self.items = None

    @property
    def is_filled(self):
        if self.items is None:
            return False
        return int(self.items.shape[0]) == self.max_size

    def __iter__(self):
        for i in range(0, self.items.shape[0], self.batch_size):
            batch = self.items[i:i+self.batch_size]
            if batch.shape[0] == self.batch_size:
                yield batch
