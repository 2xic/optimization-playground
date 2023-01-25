from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 256)
     #   self.fc_t = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 28 * 28)
        self.embedding = nn.Embedding(1001, 24 * 24)

    def forward(self, x, t):
        t = self.embedding(t).reshape((x.shape[0], 1, 24, 24))
        x = self.pool(
            F.tanh(self.conv1(x)) +\
            F.tanh(t)
        )
        x = self.pool(F.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.reshape((x.shape[0], 1, 28, 28))
        return x
