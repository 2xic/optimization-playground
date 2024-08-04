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
        print(f"Using {self.device} for training")
        self.iterator_loop = lambda x, _train: x

    def use_tqdm(self):
        self.iterator_loop = lambda x, train: tqdm(x, desc="Training" if train else "Testing")
        return self
    
    def eval(self, dataloader):
        return self.eval_with_loss(dataloader)[1]

    def eval_with_loss(self, dataloader):
        with torch.no_grad():
            return self._iterate(dataloader, train=False)

    def train(self, dataloader):
        return self._iterate(dataloader, train=True)

    def _iterate(self, dataloader, train=True):
        device = self.device
        self.model.to(device)
        total_loss = torch.tensor(0.0, device=device)
        accuracy = torch.tensor(0.0, device=device)
        length = 0

        #loop = tqdm(dataloader, desc="Training" if train else "Testing")
        training_loop = self.iterator_loop(dataloader, train)
        for (X, y) in training_loop:
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if not self.loss is None:
                loss = self.loss(y_pred, y)
                total_loss += loss
            else:
                total_loss = None

            if train:
                assert self.loss is not None
                self.optimizer.zero_grad(
                  set_to_none=True
                )
                loss.backward()
                self.optimizer.step()
                if isinstance(training_loop, tqdm):
                    training_loop.set_description(f"Loss: {total_loss.item()}, Accuracy: {(accuracy / length) * 100}%")
            
            # TODO: Maybe instead add a custom accuracy metric field
            if y_pred.shape[-1] == 1:
                # check if it is within the error margin
                accuracy += ((y_pred - y).abs() < 0.001).sum()
            else:
                accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        return (
            total_loss,
            accuracy
        )
