from dataclasses import dataclass
import torch.nn as nn
import torch
import math

@dataclass
class Config:
    sequence_length: int
    vocab_size: int
    dim_embeddings: int 
    num_transformer_layers: int

class TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.configs = config

        self.module = nn.Sequential(*[
            nn.Linear(self.configs.dim_embeddings, self.configs.dim_embeddings),
        ])

    def forward(self, X):
        return self.module(X)
    
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embeds = nn.Embedding(
            max_len,
            d_model, 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeds(x)
    
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
            self.config.dim_embeddings
        )
        self.positional_embeddings = PositionalEmbeddings(
            self.config.dim_embeddings,
            self.config.sequence_length + 1
        )
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(self.config.num_transformer_layers)
        ])
        self.output_layer = nn.Linear(self.config.dim_embeddings, self.config.vocab_size)

    def forward(self, X):
        seq_len = X.shape[1]
        positions = torch.arange(seq_len).unsqueeze(0)

        x = self.embeddings(X)  + self.positional_embeddings(positions)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x)
