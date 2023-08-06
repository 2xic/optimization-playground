import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from .PositionalEncoding import PositionalEncoding

@dataclass
class Config:
    encoder_vocab: int
    decoder_vocab: int
    embedding_dim: int
    # transformer encoder / decoder config
    transformer_layers: int
    attention_heads: int
    dropout: int
    feed_forward: int
    # vocab config
    padding_index: int

class TransformerModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(TransformerModel, self).__init__()
        self.config = config

        self.encoder_embedding = nn.Embedding(config.encoder_vocab, config.embedding_dim)
        self.decoder_embedding = nn.Embedding(config.decoder_vocab, config.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(config.embedding_dim, config.attention_heads, config.feed_forward, config.dropout)
        decoder_layer = nn.TransformerDecoderLayer(config.embedding_dim, config.attention_heads, config.feed_forward, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = config.transformer_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers = config.transformer_layers)

        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim, config.decoder_vocab),
            nn.LogSoftmax(dim=1)
        ]) 
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim, 
            config.dropout
        )

        # (SEQ_LEN, BATCH_SIZE, EMBEDDING_DIM)

    def forward(self, X: Tensor, y: Tensor):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        # embedding
        source = self.encoder_embedding(X) + self.pos_encoder(X) #.permute(1, 0, 2)
        target = self.decoder_embedding(y) + self.pos_encoder(X) #.permute(1, 0, 2)
        # forward
        memory = self.transformer_encoder(source)
        transformer_out = self.transformer_decoder(target, memory)
        # Remapping into
        # (SEQ_LEN, BATCH_SIZE, VOCAB_SIZE) -> (BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
        return self.output(transformer_out).permute(0, 2, 1)

    def forward_argmax(self, x, y):
        prediction = self.forward(x, y)
        return prediction.argmax(dim=1)
    
    def rollout(self, X, steps):
        y  = []
        target = torch.zeros_like(X)
        for index in range(steps):
            y.append(
                self.forward_argmax(
                    X,
                    target
                )[0][min(index, X.shape[1] - 1)]
            )
            """
            TEMP: Hacky plz fix
            """
            if (index + 1) < X.shape[-1]:
                X[0][index + 1] = y[-1]
                target[0][index] = y[-1]
            else:
                # Shift then add at last index
                X[0][:-2] = X.clone()[0][1:-1]
                target[0][:-2] = target.clone()[0][1:-1]
                # add
                X[0][-1] = y[-1]
                target[0][-1] = y[-1]
        return y
