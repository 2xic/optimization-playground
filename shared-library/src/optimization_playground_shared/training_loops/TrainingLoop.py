import torch
import torch.nn as nn
from tqdm import tqdm

class TrainingLoop:
    def __init__(self, model, optimizer, loss=nn.NLLLoss()):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epoch = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        print(f"Using {self.device} for inference")
        self.iterator_loop = lambda x, _train: x

    def use_tqdm(self):
        self.iterator_loop = lambda x, train: tqdm(x, desc="Training" if train else "Testing")
        return self
    
    def eval(self, dataloader):
        return self.eval_with_loss(dataloader)[1]

    def eval_with_loss(self, dataloader):
        with torch.no_grad():
            return self._iterate(dataloader, train=False)

    def train(self, dataloader, sharding=False, callback=None):
        return self._iterate(dataloader, train=True, sharding=sharding, callback=callback)

    def _iterate(self, dataloader, train, sharding, callback=None):
        device = self.device
        if not sharding:
            self.model.to(device)
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0
        has_nan_loss = False

        training_loop = self.iterator_loop(dataloader, train)
        for (X, y) in training_loop:
            if callback is not None:
                X, y = callback(X, y)
            X: torch.Tensor = X.to(device)
            y: torch.Tensor = y.to(device)
            y_pred = self.model(X)

            if not self.loss is None:
                loss = self.loss(y_pred, y)
                if torch.isnan(loss):
                    has_nan_loss = True
                else:
                    total_loss += loss
                    has_nan_loss = False
            else:
                total_loss = None

            # Nan loss can hurt training, I don't like it.
            if train and not has_nan_loss:
                assert self.loss is not None
                self.optimizer.zero_grad(
                  set_to_none=True
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            # TODO: Maybe instead add a custom accuracy metric field
            if y_pred.shape[-1] == 1:
                # check if it is within the error margin
                accuracy += ((y_pred - y).abs() < 0.001).sum()
            else:
                accuracy += (torch.argmax(y_pred, 1) == y).sum()
            # Shape of y would be equal to the number of predictions if we flatten
            length += y.shape[0]

            # Fallback
            if isinstance(training_loop, tqdm):
                if has_nan_loss:
                    training_loop.set_description(f"Loss: {total_loss.item()} (last loss was nan), Accuracy: {(accuracy / length) * 100}%")                    
                else:
                    training_loop.set_description(f"Loss: {total_loss.item()}, Accuracy: {(accuracy / length) * 100}%")
        
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        return (
            total_loss,
            accuracy
        )
