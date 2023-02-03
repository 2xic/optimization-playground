from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch

class Predictor(nn.Module):
    def __init__(self, output=10, output_relu=True):
        super().__init__()
        self.fc = nn.Linear(128, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, output)
        self.output_relu = output_relu

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.batch_norm(x)
        if self.output_relu:
            x = F.relu(self.output(x))
        else:
            x = (self.output(x))
        return x
