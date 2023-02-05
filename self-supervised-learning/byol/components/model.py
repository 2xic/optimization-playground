from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Tested with resnet, but does not help.
        Authors used resnet.
        """
        #self.model_conv = torchvision.models.resnet50(pretrained=False)
        #num_ftrs = self.model_conv.fc.in_features
        #self.model_conv.fc = nn.Linear(num_ftrs, 8 * 8)

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv_1_batch_norm = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv_2_batch_norm = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 8 * 8)
        self.batch_norm_out = nn.BatchNorm1d(8 * 8)

    def forward(self, x):
        #return self.model_conv(x)
        x = self.pool(self.conv_1_batch_norm(F.relu(self.conv1(x))))
        x = self.pool(self.conv_2_batch_norm(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.batch_norm(F.relu(self.fc1(x)))
        x = (F.sigmoid(self.fc2(x)))

        return x
