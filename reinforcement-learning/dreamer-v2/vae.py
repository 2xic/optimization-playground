import torch.nn as nn
from optimization_playground_shared.models.BasicConvModel import get_output_shape
import torch

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        out = x.view(self.shape)
        return out


class Z(nn.Module):
    def __init__(self, hidden_dim, z_shape):
        super(Z, self).__init__()
        self.mean = nn.Linear(hidden_dim, z_shape)
        self.var = nn.Linear(hidden_dim, z_shape)

    def forward(self, x):
        mean = self.mean(x)
        log_var = self.var(x)

        return mean, log_var


class SimpleVaeModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), z_size=128, conv_shape=None):
        super().__init__()
        self.z_size = z_size
        self.channels = input_shape[0]
        if conv_shape is None:
            self.conv_shape = [
                32,
                64,
                128,
            ]
        else:
            self.conv_shape = conv_shape
        self.encoder = self.get_encoder(input_shape)


    def encode(self, image, hidden_state):
        (image_encoder, hidden_state_encoder, output) = self.encoder
        z_image = image_encoder(image)
        z_hidden = hidden_state_encoder(hidden_state)
        z_combined = z_image + z_hidden
        return output(z_combined)

    def get_encoder(self, input_shape):
        last_channel = input_shape[0]
        encoders = []
        for i in self.conv_shape:
            encoders.append(
                nn.Conv2d(
                    last_channel,
                    out_channels=i,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
            )
            encoders.append(
                nn.BatchNorm2d(i)
            )
            encoders.append(
                nn.LeakyReLU(),
            )
            last_channel = i
        output_shape = ((
            get_output_shape(input_shape,
                             encoders)
        ))
        hidden_dim = 256
        fc1 = nn.Linear(
            self.conv_shape[-1] * output_shape[0] * output_shape[1],
            hidden_dim
        )
        image_encoder = torch.nn.Sequential(*encoders+[
            torch.nn.Flatten(1),
            fc1,
        ])
        hidden_state_encoder = torch.nn.Sequential(*[
            torch.nn.Linear(64, hidden_dim),
            nn.Sigmoid(),
        ])
        
        return image_encoder, hidden_state_encoder, torch.nn.Sequential(*[
            nn.LeakyReLU(),
            Z(hidden_dim, self.z_size)
        ])
