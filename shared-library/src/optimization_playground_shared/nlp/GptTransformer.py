import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from .PositionalEncoding import SinusoidalPositionalEncoding, RotaryPositionalEncoding
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


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, layers) -> None:
        super(TransformerDecoderWrapper, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, torch.zeros_like(x))
        return x


class GptTransformerModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(GptTransformerModel, self).__init__()
        self.config = config

        self._dummy_linear = nn.Linear(1, 1)

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.padding_index
        )
        layers = []
        for _ in range(config.transformer_layers):
            layers.append(
                nn.TransformerDecoderLayer(
                    d_model=config.embedding_dim,
                    nhead=config.attention_heads,
                    dim_feedforward=config.feed_forward,
                    dropout=config.dropout,
                    batch_first=True,
                    activation=nn.functional.gelu,
                )
            )

        self.transformer_decoder = TransformerDecoderWrapper(layers)
        self.layer_norm = nn.LayerNorm(self.config.embedding_dim)
        self.output = nn.Sequential(
            *[
                nn.Linear(config.embedding_dim, config.vocab_size, bias=False),
            ]
        )
        self.pos_encoder = SinusoidalPositionalEncoding(
            config.embedding_dim,
            max_len=config.sequence_length,
        )
        self.sequence_size = config.sequence_length
        # (SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM)
        self.dropout = nn.Dropout(config.dropout)
        self.use_raw_forward = False

    def with_rope_encoding(self):
        self.pos_encoder = RotaryPositionalEncoding(
            self.config.embedding_dim,
            max_len=self.config.sequence_length,
        )
        return self

    def raw_forward(self, x: Tensor):
        assert len(x.shape) == 2
        # embedding -> positional tokens + pos encoder
        # (batch size, sequence size, embedding_size)
        source = self.pos_encoder(self.embedding(x))
        source = self.dropout(source)
        # forward
        # (batch size, sequence size, embedding_size)
        source = self.transformer_decoder(source)  # , torch.zeros_like(source))
        # (batch size, sequence size, embedding)
        source = self.layer_norm(source)
        source = self.output(source)
        # (batch size, sequence size, vocab_size)
        return source

    def forward(self, x: Tensor):
        results = self.raw_forward(x)
        if self.use_raw_forward:
            return results
        # (batch size, sequence size, vocab_size)
        reshaped = results.view(-1, self.config.vocab_size)
        return reshaped

    def embeddings(self, X):
        with torch.no_grad():
            values = self.embedding(X) + self.pos_encoder(X)
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
        X_tensors = []
        with torch.no_grad():
            output = seed
            output = list(filter(lambda x: x != self.config.padding_index, output))
            for _ in range(steps):
                X = (
                    torch.full((1, self.sequence_size), self.config.padding_index)
                    .reshape(1, -1)
                    .to(self.device)
                    .long()
                )
                context_tensor = torch.tensor(output[-self.sequence_size :]).long()
                X[0, : context_tensor.shape[0]] = context_tensor
                next_predicted_token = self.raw_forward(X)
                next_predicted_token = next_predicted_token[
                    :, :: self.config.sequence_length, :
                ][0]
                next_predicted_token = torch.nn.functional.softmax(
                    next_predicted_token, dim=1
                )
                samplings = {
                    "argmax": argmax_sampling,
                    "temperature": temperature_sampling,
                }
                next_predicted_token = samplings[sampling](next_predicted_token).item()
                assert type(next_predicted_token) == int, next_predicted_token
                output.append(next_predicted_token)
                X_tensors.append(X)
        return output, X_tensors

    def rollout_output(self, seed, steps, sampling="argmax"):
        output, _ = self.rollout(seed, steps, sampling)
        return output
