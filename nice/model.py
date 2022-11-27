from layer import AddictiveCouplingLayer, ScalingLayer
import torch
import torch.nn as nn
from helper import get_prior, generate_mask


class Model(nn.Module):
    def __init__(self, device) -> None:
        super(Model, self).__init__()

        self.d = 28
        self.D = 28 * 28
        # Neural network input is (d), and output (D - d)
        # D = size of input
        # d = partition 1 size
        self.layer_1 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=False, device=device))
        self.layer_2 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=True, device=device))
        self.layer_3 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=False, device=device))
        self.layer_4 = AddictiveCouplingLayer(
            self.d, self.D, generate_mask(start_odd=True, device=device))

        self.scale = ScalingLayer()

        self.layers = [
            self.layer_1,
            self.layer_2,
            self.layer_3,
            self.layer_4,
            self.scale
        ]

    def forward(self, x):
        z = x
        log_determinant_jacobian = 0
        for i in self.layers:
            z = i.forward(z)
        log_determinant_jacobian += torch.sum(self.scale.scale_vector)
        likelihood = torch.sum(get_prior(z), dim=1) + log_determinant_jacobian
        return likelihood

    def backward(self, z):
        x = self.scale.backward(z)
        for j in reversed(self.layers):
            x = j.backward(x)
        return x
