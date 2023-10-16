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
            nn.Sigmoid(),
            nn.Linear(128, first_layer_out_size),
        ])
        self.out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(first_layer_out_size * 2, 128),
            nn.Sigmoid(),
            nn.Dropout(p=0.01),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 728),
            nn.Sigmoid(),
            nn.Linear(728, 28 * 28),
            nn.Sigmoid(),
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
