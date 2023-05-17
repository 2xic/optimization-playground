from torch import nn
import torch
import torch.nn as nn
from constants import Constants


class Yolo(nn.Module):
    def __init__(self, constants: Constants):
        super().__init__()

        OUTPUT = constants.tensor_grid_size
        leak = 0.1

        # first block
        self.conv_1 = nn.Sequential(*[
            nn.Conv2d(3, 192, (7, 7), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.MaxPool2d((2, 2), stride=2),
            nn.LeakyReLU(leak),
        ])

        # second block
        self.conv_2 = nn.Sequential(*[
            nn.Conv2d(192, 128, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.MaxPool2d((2, 2), stride=2),
            nn.LeakyReLU(leak),
        ])

        # third block
        self.conv_3 = nn.Sequential(*[
            nn.Conv2d(128, 256, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.MaxPool2d((2, 2), stride=2),
            nn.LeakyReLU(leak),
        ])

        # forth block
        self.conv_4 = nn.Sequential(*[
            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            # ----
            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(1024, 512, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.MaxPool2d((2, 2), stride=2),
            nn.LeakyReLU(leak),
        ])

        # fifth block
        self.conv_5 = nn.Sequential(*[
            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(1024, 512, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(1024, 1024, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),

            # ----
            nn.Conv2d(1024, 1024, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(1024, 1024, (3, 3), stride=(2, 2)),
            nn.LeakyReLU(leak),
            nn.MaxPool2d((2, 2), stride=2),
            nn.LeakyReLU(leak),
        ])

        # sixth block
        self.conv_5 = nn.Sequential(*[
            # TODO: I think this should be 1024 also ?
            nn.Conv2d(512, 1024, (3, 3), stride=(1, 1)),
            nn.LeakyReLU(leak),
            nn.Conv2d(1024, 1024, (3, 3), stride=(2, 2)),
            nn.LeakyReLU(leak),
        ])

        self.linear = nn.Sequential(*[
            nn.Linear(102400, 512),
            nn.LeakyReLU(leak),
            nn.Dropout(p=0.3),
            nn.Linear(512, 4096),
            nn.LeakyReLU(leak),
            nn.Linear(4096, OUTPUT),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        """
        Some notes
        - output layer has normalized coordinates between 0 and 1
        """
        return x

    # configure loss : MSE
    # ^ loss has special configuration
    #   - "increase the loss from bounding box coordinate predictions and decrease the loss from confidence predictions for boxes that don’t contain objects"
    #    - see full loss on page 4
    #
