import torch
from gated_activation import GatedActivation

class TinyWavenet(torch.nn.Module):

    def __init__(self):
        super(TinyWavenet, self).__init__()

        self.layer_1 = GatedActivation()
        """
        Add layer on layer here
        """

    def forward(self, x):
        x = self.layer_1(x)

        return x
