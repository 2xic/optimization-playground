from layer import AddictiveCouplingLayer
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.d = 2
        self.D = 4
        # Neural network input is (d), and output (D - d)
        # D = size of input
        # d = partition 1 size
        self.layer_1 = AddictiveCouplingLayer(self.d, self.D)
        self.layer_2 = AddictiveCouplingLayer(self.d, self.D)
        self.layer_3 = AddictiveCouplingLayer(self.d, self.D)
        self.layer_4 = AddictiveCouplingLayer(self.d, self.D)

        # 
        self.S = torch.zeros((16, 16))


    def split_forward(self, X):
        x_1, x_2 = self.layer_1.split(X)
        return self.forward(x_1, x_2)

    def prior(self, h_d):
        return - torch.log(
            1 + torch.exp(h_d)
        ) - torch.log(
            1 + torch.exp(-h_d)
        )

    def forward(self, x_1, x_2):
        (h_1_1, h_1_2) = self.layer_1.forward(x_1, x_2)
        (h_2_1, h_2_2) = self.layer_1.forward(
            h_1_2,
            x_2,
            prev=h_1_1
        )
        (h_3_1, h_3_2) = self.layer_1.forward(
            h_2_1,
            x_1,
            prev=h_2_2
        )
        (h_4_1, h_4_2) = self.layer_1.forward(
            h_3_2,
            x_2,
            prev=h_3_1
        )

        h = torch.concat(
            [h_4_1,
             h_4_2]
        )
        z = (self.prior(h))
    #    print(z)
        return z

    