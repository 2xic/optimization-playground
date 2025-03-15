from dataclasses import dataclass
import torch.nn as nn
import torch
import math

@dataclass
class Config:
    sequence_length: int
    vocab_size: int
    dim_embeddings: int 
    num_attention_heads: int
    num_transformer_layers: int
    padding_index: int

class TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.configs = config
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.dim_embeddings, 
            num_heads=config.num_attention_heads, 
            batch_first=True
        )
        self.module = nn.Sequential(*[
            nn.Linear(self.configs.dim_embeddings, self.configs.dim_embeddings),
            nn.ReLU()
        ])
        self.layer_norm_in = nn.LayerNorm(config.dim_embeddings)
        self.layer_norm_out = nn.LayerNorm(config.dim_embeddings)

    def forward(self, X, mask):
        attn_output, _ = self.self_attention(X, X, X, attn_mask=mask)
        return self.layer_norm_out(self.module(self.layer_norm_in(X + attn_output)))

class PositionalEmbeddings(nn.Module):
    def __init__(self, sequence_length: int, dim_embeddings: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(sequence_length, dim_embeddings)

    def forward(self, x: torch.Tensor):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.positional_embeddings(positions)

class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        """
        All the transformer models are usually like this:
            1. Text and position embeddings
            2. Transformer layer
            3. Dense layer
        """
        self.embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim_embeddings,
            padding_idx=config.padding_index
        )
        self.positional_embeddings = PositionalEmbeddings(
            self.config.sequence_length + 1,
            self.config.dim_embeddings
        )
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(self.config.num_transformer_layers)
        ])

        self.output_layer = nn.Linear(self.config.dim_embeddings, self.config.vocab_size)
        self.layer_norm = nn.LayerNorm(self.config.dim_embeddings) 

    def forward(self, X: torch.Tensor):
        x = self.embeddings(X) 
        x += self.positional_embeddings(x)
        mask = torch.triu(torch.ones(self.config.sequence_length, self.config.sequence_length), diagonal=1).bool()
        for layer in self.transformer_layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return self.output_layer(x)
