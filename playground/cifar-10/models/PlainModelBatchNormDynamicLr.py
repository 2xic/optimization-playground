from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainModelBatchNormDynamicLr(nn.Module):
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

    def set_optimizer(self, optimizer):
        def learning_rate(epoch):
            if epoch < 10:
                return 0.3
            elif epoch < 25:
                return 0.2
            elif epoch < 35:
                return 0.15
            elif epoch < 45:
                return 0.10
            elif epoch < 55:
                return 0.05
            elif epoch < 65:
                return 0.001
            else:
                return 3e-4
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate)

    def forward(self, x):
        x = self.pool(self.conv_1_batch_norm(F.relu(self.conv1(x))))
        x = self.pool(self.conv_2_batch_norm(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        #x = F.dropout(x, p=0.1)
        x = self.batch_norm(F.relu(self.fc1(x)))
        x = self.batch_norm_out(F.relu(self.fc2(x)))
        return F.log_softmax(x, dim=1)

    def on_epoch_end(self):
        self.scheduler.step()

    def loss(self, X, y):
        forward = self.forward(X)
        loss = torch.nn.NLLLoss()(forward, y)
        return loss
