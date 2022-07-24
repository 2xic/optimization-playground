from torch import optim, nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# just using the example model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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




