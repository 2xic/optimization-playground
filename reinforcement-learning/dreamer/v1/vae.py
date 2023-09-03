import torch.nn as nn
from optimization_playground_shared.models.BasicConvModel import get_output_shape
import torch
import torch.nn as nn
from .config import Config

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
        mean = torch.nn.Tanh()(self.mean(x))
        log_var = torch.nn.Tanh()(self.var(x))

        return mean, log_var


class SimpleVaeModel(nn.Module):
    def __init__(self, config: Config, input_shape=(1, 28, 28), z_size=128, conv_shape=None):
        super().__init__()
        self.config = config
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
        # size of the hidden state to model        
        self.hidden_size = config.z_size + 1
        (image_encoder, hidden_state_encoder, output) = self.get_encoder(input_shape)
        self.image_encoder = image_encoder
        self.hidden_state_encoder = hidden_state_encoder
        self.output = output
        self.decoder = self.get_decoder()


    def encode(self, image, hidden_state):
        z_image = self.image_encoder(image)
        z_hidden = self.hidden_state_encoder(hidden_state)
        z_combined = z_image + z_hidden
        return self.output(z_combined)

    def decode(self, image):
        return self.decoder(image)

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
            torch.nn.Linear(self.hidden_size, hidden_dim),
            nn.Sigmoid(),
        ])
        
        return image_encoder.to(self.config.device),  \
                hidden_state_encoder.to(self.config.device), \
                torch.nn.Sequential(*[
                    Z(hidden_dim, self.z_size)
                ]).to(self.config.device)

    def get_decoder(self):
        # z_size = latent vector + hidden_size
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
                )
            )
            decoders.append(
                nn.BatchNorm2d(reversed[index + 1])
            )
            decoders.append(
                nn.LeakyReLU(),
            )
        output = nn.Sequential(
            nn.ConvTranspose2d(reversed[-1],
                               reversed[-1],
                               kernel_size=9,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Conv2d(reversed[-1], out_channels=self.channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        return torch.nn.Sequential(
            *decoders + [output]
        )