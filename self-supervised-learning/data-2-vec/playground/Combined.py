import torch.nn as nn
import torch

class CombinedModel(nn.Module):
    def __init__(self, components):
        super().__init__()
        self.combined = torch.nn.Sequential(
            *components
        )

    def forward(self, x):
        return self.combined(x)
