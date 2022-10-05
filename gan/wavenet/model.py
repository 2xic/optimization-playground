import torch
from helpers import softmax_mu
from gated_activation import GatedActivation
import torch.nn as nn


"""
Quote from the paper:

There are no pooling layers in the network, and the output of the model has the same time 
dimensionality as the input. The model outputs a categorical distribution over the next value 
x_t with a softmax layer and it is optimized to maximize the log-likelihood of the data w.r.t. 
the parameters. Because log-likelihoods are tractable, we tune hyper-parameters on a validation 
set and can easily measure if the model is overfitting or underfitting.
"""

class TinyWavenet(torch.nn.Module):

    def __init__(self):
        super(TinyWavenet, self).__init__()

        self.layer_1 = GatedActivation()
        """
        Add more layer here
        """

        """
        post process skip connections
        """
        self.post_skip_connection = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=16,
                    out_channels=128,
                    kernel_size=1,
                    padding="same"
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=128,
                    out_channels=1, 
                    kernel_size=1,
                    padding="same"
                ),
                nn.Softmax(dim=-1),
            ]
        )


    def forward(self, x):
        x = self.layer_1(x)

        x = self.post_skip_connection(x)

        return x
