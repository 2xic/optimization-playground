import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    sequence_size: int
    # transformer decoder config
    transformer_layers: int
    attention_heads: int
    dropout: int
    feed_forward: int
    # vocab config
    padding_index: int


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x
    
class GPT2(nn.Module):
    def __init__(
        self, config: Config
    ):
        super(GPT2, self).__init__()

        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.sequence_size, config.embedding_dim)

        self.sos = torch.nn.Parameter(torch.zeros(config.embedding_dim))
        nn.init.normal_(self.sos)

        self.transformer_decoder = nn.ModuleList()
        for _ in range(config.transformer_layers):
            self.transformer_decoder.append(Block(config.embedding_dim, config.attention_heads))

        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def forward(self, x,):
        length, batch = x.shape
        h = self.token_embeddings(x)

        # prepend sos token <- which makes or breaks if the model works
        sos = torch.ones(1, batch, self.config.embedding_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for i in self.transformer_decoder:
            h = i(h)
        h = self.ln_f(h)

        logits = self.head(h)
        return logits
