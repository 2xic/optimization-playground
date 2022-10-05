import torch.nn.functional as F
import torch.nn as nn
import torch


class GatedActivation(nn.Module):
    def __init__(self, dilation, in_channel_size, out_channel_size):
        super().__init__()

        self.v_in = nn.Conv1d(
            in_channels=in_channel_size,
            out_channels=in_channel_size,
            kernel_size=1,
            padding=1,
            bias=False,
        )
        self.v_h = nn.Conv1d(
            in_channels=in_channel_size,
            out_channels=in_channel_size,
            kernel_size=2,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.v_g = nn.Conv1d(
            in_channels=in_channel_size,
            out_channels=in_channel_size,
            kernel_size=2,
            dilation=dilation,
            padding=0,
            bias=False,
        )
        self.v_out = nn.Conv1d(
            in_channels=in_channel_size,
            out_channels=in_channel_size,
            kernel_size=1,
            padding=1,
            bias=False,
        )
        self.skip_it = nn.Conv1d(
            in_channels=in_channel_size,
            out_channels=out_channel_size,
            kernel_size=1,
            padding=1,
            bias=False,
        )

    def forward(self, h):
        x = self.v_in(h)

        z = torch.tanh(self.v_h(x)) * torch.sigmoid(self.v_g(x))

        z_x = self.skip_it(z)
        z = self.v_out(z[:,:,:-1])

        x = z + x[:, :, :z.size(2)]

#        print(z_x.shape)

        return x, z_x

