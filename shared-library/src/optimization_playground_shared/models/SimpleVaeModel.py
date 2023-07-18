import torch.nn as nn
from .BasicConvModel import get_output_shape
import torch

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        out = x.view(self.shape)
        print("===")
        print(x.shape)
        print(out.shape)
        return out


class SimpleVaeModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.z_size = 512
        self.conv_shape = [
            32,
            64,
            128,
            256,
            512,
            1024
        ]
        self.encoder = self.get_encoder(input_shape)
        self.decoder = self.get_decoder(input_shape)

    def forward(self, image):
        z = self.encode(image)
        raise Exception("Missing decoder")

    def encode(self, image):
        return self.encoder(image)

    def decode(self, z):
        return self.decoder(z)

    def get_decoder(self, input_shape):
        self.decoder_input = nn.Linear(self.z_size, self.conv_shape[-1] * 4)
        decoders = [
            self.decoder_input,
            Reshape(-1, self.conv_shape[-1], 2, 2),
        ]
        reversed = self.conv_shape[::-1]
        for index, conv_shape in enumerate(reversed[:-1]):
            decoders.append(
                nn.ConvTranspose2d(
                    conv_shape,
                    reversed[index + 1],
                    kernel_size=6,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
            )
            decoders.append(
                nn.Tanh(),
            )
        output = nn.Sequential(
            nn.ConvTranspose2d(reversed[-1],
                               reversed[-1],
                               kernel_size=9,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Conv2d(reversed[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        return torch.nn.Sequential(
            *decoders + [output]
        )

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
                nn.Tanh(),
            )
            last_channel = i
        output_shape = ((
            get_output_shape(input_shape,
                             encoders)
        ))
        fc1 = nn.Linear(
            self.conv_shape[-1] * output_shape[0] * output_shape[1], self.z_size)
        return torch.nn.Sequential(*encoders+[
            torch.nn.Flatten(1),
            fc1,
            nn.Tanh(),
        ])
