from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_t = self.make_embedding(24 * 24, 1024)

        self.embedding = nn.Embedding(1001, 24 * 24)

        self.fc1 = nn.Linear(256, 256)
        self.fc1_t = self.make_embedding(24 * 24, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc2_t = self.make_embedding(24 * 24, 84)
        self.fc3 = nn.Linear(84, 28 * 28)
        self.fc3_t = self.make_embedding(24 * 24, 28 * 28)

    def forward(self, x, t):
        t = self.embedding(t).reshape((x.shape[0], 1, 24, 24))
        x = self.pool(F.tanh(self.conv1(x) + t))

        t = t.reshape((x.shape[0], 24 * 24))
        x = self.pool(F.tanh(self.conv2(x) + self.conv2_t(t).reshape((x.shape[0], 16, 8, 8))))

        x = torch.flatten(x, 1)    
        x = F.sigmoid(self.fc1(x) + self.fc1_t(t))
        x = F.sigmoid(self.fc2(x) + self.fc2_t(t))
        x = F.sigmoid(self.fc3(x) + self.fc3_t(t))
        x = x.reshape((x.shape[0], 1, 28, 28))
        return x
    
    def make_embedding(self, shape_in, shape_out):
        return nn.Sequential(*[
            nn.Linear(shape_in, shape_out),
            nn.Tanh(),
            nn.Linear(shape_out, shape_out)
        ])
