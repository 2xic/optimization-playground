import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import BaseTorchModel

class ConvTorchModel(BaseTorchModel):
    def __init__(self, features, outcomes) -> None:
        super().__init__(features, outcomes)

    def get_model(self, features, outcomes):
        return nn.Sequential(*[
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
            ),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(features, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outcomes),
            nn.LogSoftmax(dim=1)
        ])


    def forward(self, X):
        if type(X) == list:
            X = torch.tensor(X)
        
        X = X.unsqueeze(1)
        return self.model(X)
