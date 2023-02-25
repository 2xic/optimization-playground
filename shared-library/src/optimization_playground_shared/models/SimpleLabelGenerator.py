from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SimpleGenerator import SimpleGenerator

class SimpleLabelGenerator(SimpleGenerator):
    def __init__(self, z):
        super().__init__(z)
        self.labels = nn.Sequential(
            nn.Linear(1, z),
            nn.LeakyReLU()
        )

    def forward(self, x, y):
        y = y.reshape((-1, 1))
        label = self.labels(y)
        return super().forward(x + label)
