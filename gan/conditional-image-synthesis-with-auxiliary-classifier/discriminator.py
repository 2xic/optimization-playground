from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        # is_real + classes -> using mist
        self.source = nn.Linear(128, 1)
        self.labels = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        source = F.relu(self.source(x))
        labels = F.relu(self.labels(x))

        return source, labels
