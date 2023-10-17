from torch import nn
import torch
import torch.nn as nn


class SimpleConditionalGenerator(nn.Module):
    def __init__(self, z):
        super().__init__()
        first_layer_out_size = 64
        self.input_shape = (1, 28, 28)
        self.z_input = nn.Linear(z, first_layer_out_size)
        self.class_input = nn.Sequential(*[
            nn.Linear(10, 128),
            nn.Tanh(),
            nn.Linear(128, first_layer_out_size),
        ])
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(first_layer_out_size * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 728),
            nn.Tanh(),
            nn.Linear(728, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, x, classes):
        x = torch.concat(
            (
                self.z_input(x),
                self.class_input(classes)
            ),
            dim=1
        )
        x = self.out(x)
        x = x.reshape((x.shape[0], ) + self.input_shape)
        return x
