import torch
import torch.nn as nn
from torch import Tensor
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros((max_len, d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # Make it moveable to GPU with `to` call
        # cannot also assign a variable with same name
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # return x + self.pe[:, : x.size(1)]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    _instance_count = 0  # Class variable to track instances

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position_encodings = self.get_rotary_position_embedding(
            max_seq_len=max_len,
            d_model=d_model,
        )
        # Make it moveable to GPU with `to` call
        # cannot also assign a variable with same name
        self.instance_id = RotaryPositionalEncoding._instance_count
        self.register_buffer(f"position_encodings", position_encodings, persistent=True)
        RotaryPositionalEncoding._instance_count += 1

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        cos_enc, sin_enc = (
            self.position_encodings[..., 0::2],
            self.position_encodings[..., 1::2],
        )
        x[..., 0::2] = x[..., 0::2] * cos_enc - x[..., 1::2] * sin_enc
        x[..., 1::2] = x[..., 1::2] * cos_enc + x[..., 0::2] * sin_enc
        return x

    def get_rotary_position_embedding(self, max_seq_len, d_model):
        # from https://karthick.ai/blog/2024/Rotatory-Position-Embedding-%28RoPE%29/
        angle_rates = 1 / torch.pow(
            10000, torch.arange(0, d_model, 2).float() / d_model
        )
        angles = torch.arange(max_seq_len).unsqueeze(1) * angle_rates.unsqueeze(0)
        position_encodings = torch.stack((angles.cos(), angles.sin()), dim=2).flatten(1)
        return position_encodings
