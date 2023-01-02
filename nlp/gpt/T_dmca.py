# had to peak at https://github.com/lucidrains/memory-compressed-attention/
# thank you sir.

import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvCompress(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, 
            dim, 
            kernel_size=3,
            stride = 3, 
            groups = 1
        )

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

class AttentionLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        
        self.qkv = nn.Linear(size, size * 3, bias = False)
        self.out = nn.Linear(size, size, bias=False)

        self.conv = ConvCompress(
            size
        )

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        cf = 3
        t = k.shape[1]
        padding = cf - (t % cf)
        if padding != 0:
            k, v = map(lambda t: F.pad(t, (0, 0, padding, 0)), (k, v))

        k = self.conv(k)
        v = self.conv(v)
                
        dot_attention = attention(
            q, k, v,
        )
        #return dot_attention
        print(dot_attention.shape)
        out = dot_attention.transpose(1, 2).reshape(x.shape)
        return self.out(out)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = 1
    q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).transpose(1, 2), (q, k, v))
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * 256 ** -0.5
    attn = dots.softmax(dim=-1)
    out = torch.einsum('bhij,bhjd->bhid', attn, v)

    return out

if __name__ == "__main__":
    size_in = 256

    example = torch.rand((1, size_in* 2, size_in))
    
    for attention_layer in [
        AttentionLayer(size_in),
    ]:
        output = attention_layer(example)
        print(output.shape)
        print("=" * 32)
 #   print(output.shape)
    


