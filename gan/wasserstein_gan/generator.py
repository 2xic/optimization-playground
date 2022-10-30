from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.fc1 = nn.Linear(z, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 28 * 28)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = x.reshape((x.shape[0], 1, 28, 28))
        return x
