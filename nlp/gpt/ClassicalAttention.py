import torch
import torch.nn as nn

class ClassicalAttentionLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        #self.size_in, self.size_out = size_in, size_out

        self.q = nn.Linear(size_in, size_out, bias=False)
        self.k = nn.Linear(size_in, size_out, bias=False)
        self.v = nn.Linear(size_in, size_out, bias=False)
        #self.w = nn.Linear(size_out, size_out, bias=False)

    def forward(self, x):
        dot_attention = self.attention(
            (self.q(x)), 
            (self.k(x)), 
            (self.v(x))
        )
        return dot_attention

    def attention(self, q: torch.nn.Linear, k: torch.nn.Linear, v: torch.nn.Linear) -> torch.Tensor:
        results = torch.softmax(
            q @ k.T
            /
            k.shape[0],
            dim=1
        )
        return results @ v
