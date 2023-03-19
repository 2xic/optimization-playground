import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, input) -> None:
        super().__init__()

        self.attention_1 = nn.MultiheadAttention(
            input,
            4,
        )
        self.attention_2 = nn.MultiheadAttention(
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
        self.norm3 = nn.LayerNorm(input)

    def forward(self, x, encoder=None, attn_mask=None):
        attention, _ = self.attention_1(x, x, x, attn_mask=attn_mask)

        x = attention + x
        x = self.norm1(x)

        attention, _ = self.attention_2(x, encoder, encoder, attn_mask=attn_mask)
        x = attention + x
        x = self.norm2(x)

        # -----
        x = self.ffn(x) + x
        x = self.norm3(x)

        return x
