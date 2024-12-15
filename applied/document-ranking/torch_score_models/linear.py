"""
Simple linear model for torch models 

Input is embedding and output is the score.
"""
import torch.nn as nn
from torch import optim
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader
import torch

class Model(nn.Module):
    def __init__(self, embedding_dimensions):
        super().__init__()
        self.layer = nn.Sequential(*[
            nn.Linear(embedding_dimensions, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ])

    def forward(self, x):
        return self.layer(x)


class ModelInterface:
    def __init__(self):
        self.model = None
        self.optimizer = None

    def fit(self, X, y):
        X = torch.tensor(X).float()
        y = torch.tensor(y).float().reshape((-1, 1))
        if self.model is None:
            self.model = Model(X.shape[-1])
            self.optimizer = optim.Adam(self.model.parameters())
        loss = nn.L1Loss()
        trainer = TrainingLoop(self.model, self.optimizer, loss)
        self.model.train()
        dataloader = get_dataloader(
            (X, y)
        )
        for _ in range(100):
            trainer.train(dataloader)

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X).float()
        y = self.model(X).reshape((-1)).tolist()
     #   print(y)
        return y
