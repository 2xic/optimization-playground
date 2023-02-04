from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

augmentations = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.33),
    torchvision.transforms.RandomVerticalFlip(p=0.33),
    torchvision.transforms.RandomGrayscale(p=0.33)
)

class PlainModelBatchNormAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv_1_batch_norm = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv_2_batch_norm = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        self.batch_norm_out = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.pool(self.conv_1_batch_norm(F.relu(self.conv1(x))))
        x = self.pool(self.conv_2_batch_norm(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = self.batch_norm(F.relu(self.fc1(x)))
        x = self.batch_norm_out(F.relu(self.fc2(x)))
        return F.log_softmax(x, dim=1)

    def loss(self, X, y):
        forward = self.forward(augmentations(X))
        loss = torch.nn.NLLLoss()(forward, y)
        return loss
