import torch.nn as nn
from .BasicConvModel import BasicConvModel

class SimpleDiscriminator(BasicConvModel):
    def __init__(self, image_shape=(1, 28, 28)):
        super().__init__(image_shape)
        self.out = nn.Sequential(
          nn.Linear(256, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 64),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(64, 1),
          nn.Sigmoid(),
        )
