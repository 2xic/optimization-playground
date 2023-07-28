from torch import nn
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F

class TransferLearning(pl.LightningModule):
    def __init__(self, backend, output_size=100):
        super().__init__()
        
        self.backend = backend.eval()
        for params in list(self.backend.parameters())[:3]:
            params.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.backend(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        z = self.forward(x)
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
