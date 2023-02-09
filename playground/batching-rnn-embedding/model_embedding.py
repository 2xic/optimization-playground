import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 7
        self.input_size = 5 
        self.embedding = nn.Embedding(100, self.hidden_size, padding_idx=0)

        self.linear_out = nn.Linear(
            self.input_size * self.hidden_size, 
            self.input_size
        )

    def forward(self, x):
        input_shape = x.shape[-1]
        assert input_shape == self.input_size
        x = self.embedding(x)
        x = x.reshape((x.shape[0], input_shape * self.hidden_size))
        x = self.linear_out(x)
        x = (F.sigmoid(x) * 9)
        return x
