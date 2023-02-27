from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math

class SimpleGenerator(nn.Module):
    def __init__(self, z, input_shape=(1, 28, 28)):
        super().__init__()
        self.input_shape = input_shape
        self.out = nn.Sequential(
          nn.Linear(z, 128),
          nn.LeakyReLU(),
          nn.Linear(128, 256),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(256, 512),
          nn.LeakyReLU(),
          nn.Linear(512, math.prod(input_shape)),
          nn.Tanh(),
        )

    def forward(self, x):
        x = self.out(x) 
        x = x.reshape((x.shape[0], ) + self.input_shape)
        return x
