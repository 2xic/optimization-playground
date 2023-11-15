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
        for (X, y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            loss = self.loss(y_pred, y)
            if train:
                self.optimizer.zero_grad(
                  set_to_none=True
                )
                loss.backward()
                self.optimizer.step()
                if isinstance(dataloader, tqdm):
                    dataloader.set_description(f"Loss: {loss.item()}, Accuracy: {accuracy}")
            total_loss += loss             
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]
        accuracy = (accuracy / length) * 100 
        self.epoch += 1
        return (
            total_loss,
            accuracy
        )
