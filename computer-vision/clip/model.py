from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(250000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1024)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc3(x))

        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = VisionNet()
        self.embedding = nn.Embedding(
            2048,
            8
        )
        self.embedding_fc = nn.Linear(1024, 1024)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        x = self.vision(image)
        y = torch.relu(
            self.embedding_fc(self.embedding(text).reshape((text.shape[0], -1)))
        )

        return x, y, self.logit_scale
