import torch
import torch.nn as nn

class TrainingLoopAccumulate:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss = nn.NLLLoss()
        self.accumulate_steps = 32
    
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

        for index, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            y_pred = self.model(X)

            if train:
                loss = self.loss(y_pred, y)
                loss.backward()

                if (0 < index) and index % self.accumulate_steps:
                    self._step()
                total_loss += loss             
            accuracy += (torch.argmax(y_pred, 1) == y).sum()
            length += X.shape[0]

        self._step()
        accuracy = (accuracy / length) * 100 
        return (
            total_loss,
            accuracy
        )

    def _step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(
            set_to_none=True
        )
