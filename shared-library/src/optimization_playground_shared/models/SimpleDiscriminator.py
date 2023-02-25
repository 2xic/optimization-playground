import torch.nn as nn
from .BasicConvModel import BasicConvModel

class SimpleDiscriminator(BasicConvModel):
    def __init__(self, n_channels=1):
        super().__init__(n_channels)
        self.out = nn.Sequential(
          nn.Linear(256, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 64),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(64, 1),
          nn.Sigmoid(),
        )
