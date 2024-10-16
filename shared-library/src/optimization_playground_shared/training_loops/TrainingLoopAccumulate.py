import torch
import torch.nn as nn
from tqdm import tqdm

class TrainingLoopAccumulate:
    def __init__(self, model, optimizer, loss= nn.NLLLoss(), callback=None):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.accumulate_steps = 32
        self.epoch = 0
        self.iterator_loop = lambda x, _train: x
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.callback = callback

    def eval(self, dataloader):
        with torch.no_grad():
            (_loss, acc) = self._iterate(dataloader, train=False)
            return acc
    
    def train(self, dataloader, callback=None):
        return self._iterate(dataloader, train=True, callback=callback)

    def use_tqdm(self):
        self.iterator_loop = lambda x, train: tqdm(x, desc="Training" if train else "Testing")
        return self

    def _iterate(self, dataloader, train=True, callback=None):
        self.model.to(self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        accuracy = torch.tensor(0.0, device=self.device)
        length = 0

        has_nan_loss = None
        training_loop = self.iterator_loop(dataloader, train)
        for index, (X, y) in enumerate(training_loop):
            if callback is not None:
                X, y = callback(X, y)

            X = X.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(X)

            assert torch.all(y_pred != torch.nan), "Found nan in output"

            if train:
                loss = self.loss(y_pred, y)
                if not torch.isnan(loss):
                    loss.backward()
                    total_loss += loss
                # Do the step and zero_grad
                if (0 < index) and index % self.accumulate_steps:
                    self._step()
            
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += y.shape[0]
            # Fallback
            if isinstance(training_loop, tqdm):
                if has_nan_loss:
                    training_loop.set_description(f"({self.device.type}) Epoch: {self.epoch} Loss: {total_loss.item()} (last loss was nan), Accuracy: {(accuracy / length) * 100}%")                    
                else:
                    training_loop.set_description(f"({self.device.type}) Epoch: {self.epoch} Loss: {total_loss.item()}, Accuracy: {(accuracy / length) * 100}%")
        
        self._step()
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        
        return (
            total_loss,
            accuracy
        )
    
    def _forward(self, X, y):
        self.model.to(self.device)
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(X)

        if self.callback is not None:
            X, y = self.callback(X, y)

        assert torch.all(y_pred != torch.nan), "Found nan in output"

        loss = self.loss(y_pred, y)
        loss.backward()

    def _step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(
            set_to_none=True
        )
