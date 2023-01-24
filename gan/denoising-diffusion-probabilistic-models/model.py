from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc_t = nn.Linear(1, 256)

        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 28 * 28)

    def forward(self, x, t):
        x = F.tanh(self.fc1(x) + self.fc_t(t))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
       # x = x.reshape((x.shape[0], 1, 28, 28))
        return x
