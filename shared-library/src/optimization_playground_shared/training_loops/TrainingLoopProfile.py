import torch
import torch.nn as nn
from ..utils.Timer import Timer

class TrainingLoopProfile:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss = nn.NLLLoss()
        self.epoch = 1
    
    def eval(self, dataloader):
        with torch.no_grad():
            (_loss, acc) = self._iterate(dataloader, train=False)
            return acc
    
    def train(self, dataloader):
        return self._iterate(dataloader, train=True)

    def _iterate(self, dataloader, train=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.to(device)
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0

        for (X, y) in dataloader:
            with Timer("move to device"):
                X = X.to(device)
                y = y.to(device)
            y_pred = None
            with Timer("infer"):
                y_pred = self.model(X)

            if train:
                loss = None
                with Timer("loss"):
                   loss = self.loss(y_pred, y)

                with Timer("backward + step"):
                    self.optimizer.zero_grad(
                    set_to_none=True
                    )
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss 
            
            with Timer("acc"):
                accuracy += (torch.argmax(y_pred, 1) == y).sum()
                length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        return (
            total_loss,
            accuracy
        )
