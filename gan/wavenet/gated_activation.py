import torch.nn.functional as F
import torch.nn as nn
import torch

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_in = nn.Conv1d(
            in_channels=1,
            out_channels=16, 
            kernel_size=1,
            padding="same"
        )
        self.v_h = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=2,
            dilation=1,
        )
        self.v_g = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=2,
            dilation=1,
        )
        self.v_out = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
         #   padding="same"
        )

    def forward(self, h):
        x = self.v_in(h)

        z = torch.tanh(self.v_h(x)) * torch.sigmoid(self.v_g(x))

        x = self.v_out(z)   

        x = x + h[:, :, :x.shape[-1]]

        return x
