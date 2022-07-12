import torch
import torch.nn as nn

"""
Note it should be possible to specify n- attention heads.
Currently it's hardcoded to one.
"""
class AttentionLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        channel_size = 4
        channel_size_2 = 11

        self.q = nn.Linear(channel_size, size_out, bias=False)
        self.k = nn.Linear(channel_size, channel_size_2, bias=False)
        self.v = nn.Linear(channel_size_2, size_out, bias=False)
        self.w = nn.Linear(size_out, size_out, bias=False)

    def forward(self, x):
        dot_attention = attention(
            self.q, self.k, self.v
        ) @ self.w.weight
        return (x @ dot_attention.T)

def attention(q: torch.nn.Linear, k: torch.nn.Linear, v: torch.nn.Linear) -> torch.Tensor:
    results = torch.softmax(
        q.weight @ k.weight.T
        /
        k.weight.shape[0],
        dim=1
    )
    return results @ v.weight.T
    

