from torch import nn
import torch.nn as nn
from .SimpleGenerator import SimpleGenerator

class SimpleLabelGenerator(SimpleGenerator):
    def __init__(self, z, input_shape=(1, 28, 28)):
        super().__init__(z, input_shape)
        self.labels = nn.Sequential(
            nn.Linear(1, z),
            nn.LeakyReLU()
        )

    def forward(self, x, y):
        y = y.reshape((-1, 1))
        label = self.labels(y)
        return super().forward(x + label)
