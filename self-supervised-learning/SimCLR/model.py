from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from loss import Loss
import random

DEBUG = False


def debug_assert(condition):
    if DEBUG:
        return condition
    return True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc3(x))
        return x

class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_in = nn.Linear(100, 64)
        self.projection_out = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.projection_in(x))
        return x

class SimClrTorch(nn.Module):
    def __init__(self, model, projection, debug=False):
        super().__init__()
        self.model = model
        self.projection = projection
        self.debug = debug

    def forward(self, X):
        return self.projection(self.model(X))

    def _forward(self, batch):
        x, y = batch
        model_device = next(self.projection.parameters()).device
        x = x.to(model_device)
        y = y.to(model_device)

        z_k_1 = self.forward(x)
        z_k_2 = self.forward(y)

        if self.debug:
            assert debug_assert((z_k_1).isnan().any() == False), z_k_1
            assert debug_assert((z_k_2.isnan()).any() == False), z_k_2

        loss_value = Loss().loss(z_k_1, z_k_2)

        if random.randint(0, 10) == 5 and self.debug:
            print(z_k_1[0])
            print(z_k_2[0])
            print(loss_value)
            exit(0)

        return loss_value

class SimClrModel(pl.LightningModule):
    def __init__(self, model, projection, debug=False):
        super().__init__()
        self.model = model
        self.projection = projection
        self.debug = debug

    def forward(self, X):
        return self.projection(self.model(X))

    def training_step(self, batch):
        x, y = batch
        model_device = next(self.projection.parameters()).device
        x = x.to(model_device)
        y = y.to(model_device)

        self.timer_start()
        z_k_1 = self.forward(x)
        z_k_2 = self.forward(y)

        if self.debug:  
            assert debug_assert((z_k_1).isnan().any() == False), z_k_1
            assert debug_assert((z_k_2.isnan()).any() == False), z_k_2

        loss_value = Loss().loss(z_k_1, z_k_2)

        if random.randint(0, 10) == 5 and self.debug:
            print(z_k_1[0])
            print(z_k_2[0])
            print(loss_value)
            exit(0)

        self.log("train_loss", loss_value)

        return loss_value

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def timer_start(self):
        self.start = time.time()

    def timer_end(self, action):
        print(action + " : " + str(time.time() - self.start))
