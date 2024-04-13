from torch import nn
import torch.nn as nn
import math

class SimpleGenerator(nn.Module):
    def __init__(self, z, input_shape=(1, 28, 28), normalize=False):
        super().__init__()
        self.z = z
        self.input_shape = input_shape
        self.out = nn.Sequential(
          *self._block(z, 128),
          *self._block(128, 256, normalize=normalize),
          nn.Dropout(p=0.01),
          *self._block(256, 512, normalize=normalize),
          nn.Linear(512, math.prod(input_shape)),
          nn.Tanh(),
        )

    def _block(self, in_feature, out_feat_feature, normalize=True):
        layers = [nn.Linear(in_feature, out_feat_feature)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat_feature, 0.8))
        layers.append(nn.LeakyReLU(inplace=True))
        return layers

    def forward(self, x):
        x = self.out(x) 
        x = x.reshape((x.shape[0], ) + self.input_shape)
        return x
