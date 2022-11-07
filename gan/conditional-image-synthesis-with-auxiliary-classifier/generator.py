from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.fc1 = nn.Linear(z, 128)
        self.fc1_labels = nn.Linear(10, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 28 * 28)

    def forward(self, x, labels):
        labels = labels.float()
        x = F.relu(self.fc1(x))
        x_labels = F.relu(self.fc1_labels(labels))

        x = torch.concat((x, x_labels), dim=1)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.reshape((x.shape[0], 1, 28, 28))
        return x
