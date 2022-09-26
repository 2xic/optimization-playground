from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class Yolo(nn.Module):
    def __init__(self):
        super().__init__()

        # first block
        self.conv_1 = nn.Sequential(*[
            nn.Conv2d(64, 192, (7, 7), stride=(1, 1)),
            nn.MaxPool2d((2, 2), stride=2)
        ])

        # second block
        self.conv_2 = nn.Sequential(*[ 
            nn.Conv2d(192, 128, (3, 3), stride=(1, 1)),
            nn.MaxPool2d((2, 2), stride=2) 
        ])

        # third block
        self.conv_3 = nn.Sequential(*[
            nn.Conv2d(128, 256, (1, 1), stride=(1, 1)),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1)),
            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),

            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),
            nn.MaxPool2d((2, 2), stride=2),
        ])

        # forth block
        self.conv_4 = nn.Sequential(*[
            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.Conv2d(512, 256, (3, 3), stride=(1, 1)),

            nn.Conv2d(256, 512, (1, 1), stride=(1, 1)),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1)),

            # ----
            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.Conv2d(1024, 512, (3, 3), stride=(1, 1)),

            nn.MaxPool2d((2, 2), stride=2),
        ])

        # fifth block
        self.conv_5 = nn.Sequential(*[
            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.Conv2d(1024, 512, (3, 3), stride=(1, 1)),

            nn.Conv2d(512, 1024, (1, 1), stride=(1, 1)),
            nn.Conv2d(1024, 1024, (3, 3), stride=(1, 1)),

            # ----
            nn.Conv2d(1024, 1024, (3, 3), stride=(1, 1)),
            nn.Conv2d(1024, 1024, (3, 3), stride=(2, 2)),

            nn.MaxPool2d((2, 2), stride=2),
        ])

        # sixth block
        self.conv_5 = nn.Sequential(*[
            # TODO: I think this should be 1024 also ? 
            nn.Conv2d(512, 1024, (3, 3), stride=(1, 1)),
            nn.Conv2d(1024, 1024, (3, 3), stride=(2, 2)),
        ])

        self.linear = nn.Sequential(*[
            nn.Linear(102400, 512),
            nn.Linear(512, 4096),
            nn.Linear(4096, 7* 7 * 30),
        ])


    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        return x
