import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, input) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            input,
            4,
        )
        self.ffn = nn.Sequential(*[
            nn.Linear(input, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input)
        ])

        self.norm1 = nn.LayerNorm(input)
        self.norm2 = nn.LayerNorm(input)

    def forward(self, x):
        attention, _ = self.attention(x, x, x)
#        print(attention.shape)
#        print(x.shape)
        x = attention + x
        x = self.norm1(x)

        # -----
        x = self.ffn(x) + x
        x = self.norm2(x)

        return x
