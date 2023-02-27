import torch.nn as nn
from .BasicConvModel import BasicConvModel

class SimpleLabelDiscriminator(BasicConvModel):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__(input_shape)
        self.out = nn.Sequential(
          self.fc1,
          nn.Linear(256, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 64),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(64, 11),
          nn.Sigmoid(),
        )

    def forward(self, x):
        output = super().forward(x)
        is_generated = output[:, 0]
        labels = output[:, 1:]
        return (
          is_generated,
          labels
        )
