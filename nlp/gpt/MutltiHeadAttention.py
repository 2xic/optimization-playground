import torch.nn as nn
import torch
from ClassicalAttention import ClassicalAttentionLayer

class MutltiHeadAttention(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.head_0 = ClassicalAttentionLayer(
            size_in=size_in,
            size_out=size_out
        )
        self.head_1 = ClassicalAttentionLayer(
            size_in=size_in,
            size_out=size_out
        )
        self.w_out = nn.Linear(
            size_out * 2, size_out
        )

    def forward(self, x):
        heads = torch.concat((
            self.head_0(x),
            self.head_1(x)
        )).reshape((x.shape[0], -1))
        #print(heads.shape)
        #print(x.shape)
        #exit(0)

        return (self.w_out(heads))
