from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn

class BasicConvModel(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        if n_channels == 1:
            self.fc1 = nn.Linear(16 * 4 * 4, 256)
        elif n_channels == 3:
            self.fc1 = nn.Linear(16 * 5 * 5, 256)
        else:
            raise Exception(f"Unknown output for n_channels={n_channels}")
        self.out = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 63),
            nn.ReLU(),
            nn.Dropout(p=0.01),
            nn.Linear(63, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = self.out(x)
        return x
