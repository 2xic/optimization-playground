import torch
import torch.nn as nn
from tqdm import tqdm

class TrainingLoopAccumulate:
    def __init__(self, model, optimizer, loss= nn.NLLLoss()):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.accumulate_steps = 32
        self.epoch = 0
        self.iterator_loop = lambda x, _train: x
    
    def eval(self, dataloader):
        with torch.no_grad():
            (_loss, acc) = self._iterate(dataloader, train=False)
            return acc
    
    def train(self, dataloader):
        return self._iterate(dataloader, train=True)

    def use_tqdm(self):
        self.iterator_loop = lambda x, train: tqdm(x, desc="Training" if train else "Testing")
        return self

    def _iterate(self, dataloader, train=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.to(device)
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0

        has_nan_loss = None
        training_loop = self.iterator_loop(dataloader, train)
        for index, (X, y) in enumerate(training_loop):
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = self.loss(y_pred, y)
                if not torch.isnan(loss):
                    loss.backward()
                    total_loss += loss
                # Do the step and zero_grad
                if (0 < index) and index % self.accumulate_steps:
                    self._step()
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
            # Fallback
            if isinstance(training_loop, tqdm):
                if has_nan_loss:
                    training_loop.set_description(f"Loss: {total_loss.item()} (last loss was nan), Accuracy: {(accuracy / length) * 100}%")                    
                else:
                    training_loop.set_description(f"Loss: {total_loss.item()}, Accuracy: {(accuracy / length) * 100}%")
        

        self._step()
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        
        return (
            total_loss,
            accuracy
        )

    def _step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(
            set_to_none=True
        )
