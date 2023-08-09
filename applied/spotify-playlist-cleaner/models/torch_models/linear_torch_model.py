import torch.nn as nn
from .core import BaseTorchModel

class LinearTorchModel(BaseTorchModel):
    def __init__(self, features, outcomes) -> None:
        super().__init__(features, outcomes)

    def get_model(self, features, outcomes):
        return nn.Sequential(*[
            nn.Linear(features, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, outcomes),
            nn.LogSoftmax(dim=1)
        ])
