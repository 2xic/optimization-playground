from model import SimpleModel
import torch.nn as nn
import torch

class TripletModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.x_minus = SimpleModel()
        self.x = SimpleModel()
        self.x_plus = SimpleModel()

    def forward(self, x_minus, x, x_plus):
        x = x.reshape((1, ) + x.shape)
        x_plus = x_plus.reshape((1, ) + x_plus.shape)
        x_minus = x_minus.reshape((1, ) + x_minus.shape)
        
        x_minus = self.x_minus(x_minus)
        x = self.x(x)
        x_plus = self.x_plus(x_plus)

        x_plus_norm = torch.exp(torch.norm(x - x_plus))
        x_minus_norm = torch.exp(torch.norm(x - x_minus))
        
        d_plus = x_plus_norm / (x_plus_norm + x_minus_norm)
        d_minus = x_minus_norm / (x_plus_norm + x_minus_norm)

        return (d_plus - d_minus) #** 2
    #    print(d_plus)
  #      print(d_minus)
   #     exit(0)

    def embedded(self, x):
        if len(x.shape) == 3:
            x = x.reshape((1, ) + x.shape)
        return self.x(x)
