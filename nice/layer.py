import torch
import torch.nn as nn

class AddictiveCouplingLayer(nn.Module):
    def __init__(self, d, D) -> None:
        super(AddictiveCouplingLayer, self).__init__()

        self.d = 28 * 14
        self.D = 28 * 28
#        self.model = lambda x: torch.zeros((self.D - self.d))

        self.model = nn.Sequential(
          nn.Linear(self.d, self.D - self.d),
          nn.ReLU(),
        )

    def split(self, X):
        return X[:self.d], X[self.d:]

    def forward(self, x_1, x_2, prev=None):
        assert x_1.shape[-1] == self.d
        assert x_2.shape[-1] == self.D - self.d
        
        y_1 = x_1 
        y_2 = (x_1 if prev is None else prev) + self.model(x_1.reshape((1, ) + x_1.shape).float())[0]

        return (
            y_1,
            y_2
        )

    def backward(self, y_1, y_2):
        x_1 = y_1
        x_2 = y_2 - self.model(y_1)

        return (
            x_1,
            x_2
        )
