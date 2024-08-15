import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from .PositionalEncoding import PositionalEncoding
from .utils.sampling import temperature_sampling, argmax_sampling

"""
GPT is a decoder only 
"""


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


class GptTransformerModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(GptTransformerModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            config.embedding_dim, config.attention_heads, config.feed_forward, config.dropout)

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.transformer_layers)

        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim *
                      config.sequence_size, config.vocab_size),
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout
        )
        # self.dummy_param = nn.Parameter(torch.empty(0))
        self.sequence_size = config.sequence_size
        # (SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM)

    def forward(self, X: Tensor):
        assert len(X.shape) == 2
        # embedding
        source = self.embedding(X) + self.pos_encoder(X)
        # forward
        transformer_out = self.transformer_decoder(source, source)
        transformer_out = transformer_out.reshape(X.shape[0], -1)
        # Remapping into
        # (SEQ_LEN, BATCH_SIZE, VOCAB_SIZE) -> (BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
        return self.output(transformer_out)  # .permute(0, 2, 1)

    def embeddings(self, X):
        values = torch.zeros((1, self.config.embedding_dim * self.config.sequence_size))
        for x in X:
            x = x.reshape((1, ) + X.shape[1:])
            assert len(X.shape) == 2
            with torch.no_grad():
                values += (self.embedding(x) + self.pos_encoder(x)).reshape(1, -1)
        return values

    def forward_argmax(self, x):
        prediction = self.forward(x)
        return prediction.argmax(dim=1)

    def rollout(self, seed, steps, device):
        with torch.no_grad():
            output = []
            for index in range(steps):
                next_predicted = None
                if (len(seed) - 1) < index:
                    X = torch.zeros(1, self.sequence_size).reshape(1, -1).to(device).long().fill_(self.config.padding_index)
                    copy = torch.tensor(output[-self.sequence_size:]).long()
                    X[0, :copy.shape[0]] = copy

                    X = self.forward(X)
                    next_predicted = temperature_sampling(
                        X
                    ).item()
                    output.append(next_predicted)
                else:
                    next_predicted = seed[index].item()
                    output.append(next_predicted)
            return output
