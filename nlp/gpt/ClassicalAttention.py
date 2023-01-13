import torch
import torch.nn as nn
import math

# shouldn't this be same as Scaled Dot-Product Attention?
#  https://paperswithcode.com/method/scaled 
# 

class ClassicalAttentionLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.qkv = nn.Linear(size_in, size_out * 3, bias=False)
        #self.k = nn.Linear(size_in, size_out, bias=False)
       # self.v = nn.Linear(size_in, size_out, bias=False)
        
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        dot_attention = self.attention(
            q, 
            k, 
            v
        )
        return dot_attention

    def attention(self, q: torch.nn.Linear, k: torch.nn.Linear, v: torch.nn.Linear) -> torch.Tensor:
        results = torch.softmax(
            q @ k.T
            /
            math.sqrt(k.shape[-1]),
            dim=-1
        )
        
        return results @ v
