from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import time

from loss import Loss

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_in = nn.Linear(100, 64)
        self.projection_out = nn.Linear(64, 100)

    def forward(self, x):
        x = F.relu(self.projection_in(x))
        x = self.projection_out(x)
        return x


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

        self.timer_start()
        z_k_1 = self.forward(x)
        z_k_2 = self.forward(y)
        
        assert debug_assert((z_k_1).isnan().any() == False), z_k_1
        assert debug_assert((z_k_2.isnan()).any() == False), z_k_2

        loss_value = Loss().loss(z_k_1, z_k_2)
    
        self.log("train_loss", loss_value)

        return loss_value

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def timer_start(self):
        self.start = time.time()

    def timer_end(self, action):
        print(action + " : " + str(time.time() - self.start))


class SimpleModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = nn.functional.cross_entropy(z, y)
        batch_size = x.shape[0]
        predictions = torch.argmax(z, dim=1)
        # if all are non zero accuracy = batch size
        accuracy = batch_size - torch.count_nonzero(predictions - y)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy / float(batch_size))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
