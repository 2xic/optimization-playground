import torch.nn as nn
from .BasicConvModel import BasicConvModel
import torch
import torch.nn.functional as F

class SimpleConditionalDiscriminator(BasicConvModel):
    def __init__(self, image_shape=(1, 28, 28), classes=10):
        super().__init__(image_shape)
        self.out = nn.Sequential(
          nn.Linear(256, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 64),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(64, classes),
        )
        self.out_labels = nn.Sequential(
          nn.Linear(256, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 64),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        y = self.out_labels(x)
        x = self.out(x)
        return y, x
