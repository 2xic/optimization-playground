from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
from optimization_playground_shared.models.BasicConvModel import get_output_shape

class WeightLayer(nn.Module):
    def __init__(self, width, height, z=64):
        super().__init__()
        self.z_list = nn.ParameterList()

        self.shape = (width, height)
        self.width = width
        self.height = height
        self.z = z

        for _ in range(height):
            for _ in range(width):
                param = torch.fmod(torch.randn((
                    z
                )), 2)
                self.z_list.append(nn.Parameter(param))

    def forward(self, hyper_net):
        z_tensor = torch.stack([z for z in self.z_list], dim=0) 
        z_reshaped = z_tensor.reshape(self.width, self.height, -1)
        output = hyper_net(z_reshaped)

        return output


class MainNet(nn.Module):
    def __init__(self, input_shape, hyper_net):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 4)
        self.conv_2_weight = WeightLayer(
            width=1,
            height=1,
            z=64
        )
        self.hyper_net = hyper_net
        self.output_shape = ((
            get_output_shape(input_shape,
            [self.conv1, self.pool, self.conv2, self.pool])
        ))
        self.fc1 = nn.Linear(16 * self.output_shape[0] * self.output_shape[1], 256)
        self.out = nn.Sequential(
            self.fc1,
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(F.conv2d(
            x,
            self.conv_2_weight(self.hyper_net)
        )))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = self.out(x)
        return x
