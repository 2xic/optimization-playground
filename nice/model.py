from layer import AddictiveCouplingLayer, ScalingLayer
import torch
import torch.nn as nn
from helper import get_prior, generate_mask


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.d = 28
        self.D = 28 * 28
        # Neural network input is (d), and output (D - d)
        # D = size of input
        # d = partition 1 size
        self.layer_1 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=False))
        self.layer_2 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=True))
        self.layer_3 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=False))
        self.layer_4 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=True))

        self.scale = ScalingLayer()

        self.layers = [
            self.layer_1,
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.scale
        ]

    def prior(self, h_d):
        return get_prior(h_d)

    def forward(self, x):
        z = x
        log_determinant_jacobian = 0
        for i in self.layers:
            z = i.forward(z)
        z = (self.prior(z)) + log_determinant_jacobian
        return z

    def split_backward(self, Z):
        x_1, x_2 = self.layer_1.split(Z)
        results = self.backward(x_1, x_2)
        return results

    def backward(self, z):
        x = z
        for j in reversed(self.layers):
            x = j.backward(z)
        return x
