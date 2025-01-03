import torch

class AccumulateLoss:
    def __init__(self):
        self.counter = 0
        self.loss = 0

    def update(self, loss: torch.Tensor):
        if torch.isnan(loss):
            print("nan loss .... ")
        else:
            loss.backward()
            self.counter += 1
            self.loss += loss.item()

    def reset(self):
        self.counter = 0
        self.loss = 0

    def done(self):
        return self.counter >= 32
