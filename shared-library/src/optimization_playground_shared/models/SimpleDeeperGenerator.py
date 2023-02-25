from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SimpleGenerator import SimpleGenerator

class SimpleDeeperGenerator(SimpleGenerator):
    def __init__(self, z):
        super().__init__(z)
        self.out = nn.Sequential(
          nn.Linear(z, 64),
          nn.LeakyReLU(),
          nn.Linear(64, 128),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(128, 256),
          nn.LeakyReLU(),
          nn.Dropout(p=0.01),
          nn.Linear(256, 512),
          nn.LeakyReLU(),
          nn.Linear(512, 728),
          nn.LeakyReLU(),
          nn.Linear(728, 28 * 28),
          nn.Tanh(),
        )
