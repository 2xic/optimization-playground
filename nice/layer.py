import torch
import torch.nn as nn

class AddictiveCouplingLayer(nn.Module):
    def __init__(self, d, D, mask) -> None:
        super(AddictiveCouplingLayer, self).__init__()

        self.d = d
        self.D = D

        self.model = nn.Sequential(
          nn.Linear(28 * 28, 1000),
          nn.Sigmoid(),
          nn.Linear(1000, 28 * 28),
#          nn.ReLU(),
        )
        self.mask = mask

    def split(self, X):
        return X[:self.d], X[self.d:]

    def forward(self, x):        
        x_1, x_2 = x * self.mask, (1 - self.mask) * x
        y_1 = x_1 
        y_2 = x_2 + self.model(x_1.reshape((1, ) + x_1.shape).float() * (1 - self.mask))[0]

        return y_1 + y_2

    def backward(self, z):
        y_1, y_2 = z * self.mask, (1 - self.mask) * z
        x_1 = y_1 
        x_2 = y_2 - self.model(y_1.reshape((1, ) + y_1.shape).float() * (1 - self.mask))[0]

        return x_1 + x_2

class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super(ScalingLayer, self).__init__()
        self.scale_vector = nn.Parameter(torch.randn(1, 28 * 28, requires_grad=True))

    def forward(self, x):
        # As mentioned in section 5.6
        return (
            torch.exp(self.scale_vector) * x
        )
    
    def backward(self, x):
        return (
            torch.exp( - self.scale_vector) * x
        )
