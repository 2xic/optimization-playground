from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

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
        x = F.dropout(x, p=0.1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SimpleModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.batch = 0

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_predicted = self.model(x)
        loss = nn.functional.cross_entropy(y_predicted, y)
        accuracy = self.get_acc(y_predicted, y)

        if self.batch % 10 == 0:
            with open("train_accuracy.txt", "a") as file:
                file.write(f"{accuracy}\n")
        self.batch += 1

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_predicted = self.model(x)
        loss = nn.functional.cross_entropy(y_predicted, y)

        accuracy = self.get_acc(y_predicted, y)

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)

        if self.batch % 10 == 0:
            with open("test_accuracy.txt", "a") as file:
                file.write(f"{accuracy}\n")
        self.batch += 1

    def get_acc(self, predicted, truht):
        batch_size = truht.shape[0]
        predictions = torch.argmax(predicted, dim=1)

        # if all are non zero accuracy = batch size
        accuracy = batch_size - torch.count_nonzero(predictions - truht)
        return accuracy / float(batch_size)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
