import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from optimization_playground_shared.nlp.PositionalEncoding import PositionalEncoding
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling, argmax_sampling

"""
GPT is a decoder only 
"""
@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    sequence_size: int
    dropout: float
    padding_index: int

class TinyModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(TinyModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim *
                      config.sequence_size, config.vocab_size),
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout
        )
        self.sequence_size = config.sequence_size

    def forward(self, X: Tensor):
        assert len(X.shape) == 2
        # embedding
        source = self.embedding(X) + self.pos_encoder(X)
        # forward
#        transformer_out = self.transformer_decoder(source, source)
        transformer_out = source.reshape(X.shape[0], -1)
        # Remapping into
        # (SEQ_LEN, BATCH_SIZE, VOCAB_SIZE) -> (BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
        return self.output(transformer_out)  # .permute(0, 2, 1)

    def forward_argmax(self, x):
        prediction = self.forward(x)
        return prediction.argmax(dim=1)

    def rollout(self, seed, steps, device=torch.device('cpu')):
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
