from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        return x
