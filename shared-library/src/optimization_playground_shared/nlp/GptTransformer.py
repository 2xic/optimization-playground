import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from .PositionalEncoding import PositionalEncoding
from .utils.sampling import temperature_sampling, argmax_sampling
import math

"""
GPT is a decoder only 
"""


@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    sequence_length: int
    # transformer decoder config
    transformer_layers: int
    attention_heads: int
    dropout: int
    feed_forward: int
    # vocab config
    padding_index: int


class GptTransformerModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(GptTransformerModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim, 
            nhead=config.attention_heads, 
            dim_feedforward=config.feed_forward, 
            dropout=config.dropout,
            batch_first=True,
            activation=nn.functional.gelu,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.transformer_layers,
        )
        self.layer_norm = nn.LayerNorm(self.config.embedding_dim) 
        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim, config.vocab_size, bias=False),
#            nn.Sigmoid()
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout,
            max_len=config.sequence_length,
        )
        # self.dummy_param = nn.Parameter(torch.empty(0))
        self.sequence_size = config.sequence_length
        # (SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM)
       # self.pos_encoder = nn.Embedding(config.sequence_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def raw_forward(self, X: Tensor):
        assert len(X.shape) == 2
        # embedding -> positional tokens + pos encoder
        # (batch size, sequence size, embedding_size)
       # pos = torch.arange(0, X.shape[1], dtype=torch.long, device=X.device).unsqueeze(0)
        source = self.embedding(X) + self.pos_encoder(X)
        source = self.dropout(source)
        # forward
        # (batch size, sequence size, embedding_size)
        source = self.transformer_decoder(source, torch.zeros_like(source))
        # (batch size, sequence size, embedding)
        source = self.layer_norm(source)
        source = self.output(source)
        # (batch size, sequence size, vocab_size)
        source = source.to(X.device)
        return source

    def forward(self, X: Tensor):
        results = self.raw_forward(X)
        # (batch size, sequence size, vocab_size)
        reshaped = results.view(-1, self.config.vocab_size)

        return reshaped

    def embeddings(self, X):
        values = torch.zeros((1, self.config.embedding_dim * self.config.sequence_length))
        for x in X:
            x = x.reshape((1, ) + X.shape[1:])
            assert len(X.shape) == 2
            with torch.no_grad():
                values += (self.embedding(x) + self.pos_encoder(x)).reshape(1, -1)
        return values

    def forward_argmax(self, x):
        prediction = self.forward(x)
        return prediction.argmax(dim=1)

    def predict(self, X, next_token_location=-1):
        results = self.raw_forward(X)
        return results[:, next_token_location, :].reshape((results.shape[0], -1))

    @property
    def device(self):
        return next(self.parameters()).device

    def rollout(self, seed, steps, sampling="argmax"):
        with torch.no_grad():
            output = []
            prev = None
            for index in range(steps):
                next_predicted = None
                if len(seed) <= index:
                    X = torch.full((1, self.sequence_size), self.config.padding_index).reshape(1, -1).to(self.device).long()
                    context_tensor = torch.tensor(output[-self.sequence_size:]).long()
                    X[0, :context_tensor.shape[0]] = context_tensor
                    assert prev is None or torch.all(prev[0, 1:] == X[0, :-1])

                    next_token_location = (
                        -1 if context_tensor.shape[0] == self.sequence_size else context_tensor.shape[0] - 1
                    )
                    next_token = self.predict(X, next_token_location=next_token_location)
                    next_predicted = None

                    if sampling == "argmax":
                        next_predicted = argmax_sampling(
                            next_token
                        ).item()
                    else:
                        next_predicted = temperature_sampling(
                            next_token[0]
                        ).item()
                    assert type(next_predicted) == int, next_predicted
                    output.append(next_predicted)
                    prev = X
                else:
                    next_predicted = seed[index].item()
                    assert type(next_predicted) == int, next_predicted
                    output.append(next_predicted)
            return output
