from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channel, out_channel, kernel_size=3, stride=1):
    return nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        bias=False,
    )

class ResidualBlock(nn.Module):
    def __init__(self, input, output) -> None:
        super().__init__()
        
        self.conv_1 = conv_layer(input, output)
        self.conv_2 = conv_layer(output, output)

    def forward(self, x):
        residual = x

        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x += residual

        return F.relu(x)

class PlainModelResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = ResidualBlock(16, 16)
        self.fc1 = nn.Linear(16 * 30 * 30, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        #x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def loss(self, X, y):
        forward = self.forward(X)
        loss = torch.nn.NLLLoss()(forward, y)
        return loss
