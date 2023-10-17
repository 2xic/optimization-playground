import torch.nn as nn
from .BasicConvModel import BasicConvModel
import torch
import torch.nn.functional as F

class SimpleConditionalDiscriminator(BasicConvModel):
    def __init__(self, image_shape=(1, 28, 28), classes=10):
        super().__init__(image_shape)
        self.out = nn.Sequential(
            self.fc1,
            nn.ReLU(),
#            nn.Linear(256, 128),
#            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Tanh(),
        )
        self.out_labels = nn.Sequential(
          nn.Linear(64, classes),
        )
        self.is_real_labels = nn.Sequential(
          nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = self.out(x)
        is_real_labels = F.relu(self.is_real_labels(x))
        out_labels = F.relu(self.out_labels(x))
        return is_real_labels, out_labels
