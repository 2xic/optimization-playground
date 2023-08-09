import torch
import torch.nn.functional as F
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.training_loops.TrainingLoopPlot import TrainingLoopPlot
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader

class BaseTorchModel(torch.nn.Module):
    def __init__(self, features, outcomes) -> None:
        super().__init__()
        self.model = self.get_model(features, outcomes)

    def get_model(self):
        raise Exception("Should be implemented in parent")

    def forward(self, X):
        if type(X) == list:
            X = torch.tensor(X)
        return self.model(X)
    
    def predict(self, X):
        return torch.argmax(
            self.forward(X),
            dim=1
        )
    
    def fit(self, X, y ):
        X = torch.tensor(X)
        y = torch.tensor(y)

        optimizer = torch.optim.Adam(self.parameters())
        dataloader = get_dataloader((X, y))

        TrainingLoopPlot(TrainingLoop(
            self,
            optimizer
        )).fit(dataloader, 3_000)
