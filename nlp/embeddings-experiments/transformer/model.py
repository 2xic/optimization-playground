# Based off https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab, device, d_model: int = 32, nhead: int = 8,
                 nlayers: int = 6, dropout: float = 0.5):
        super().__init__()
        self.vocab = vocab
        self.device = device
        ntoken = self.vocab.size

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model, padding_idx=self.vocab.PADDING_IDX)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = torch.nn.Sigmoid(output)
        return output

    def fit(self, X):
        sz = 9
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'),
                          diagonal=1)  # .long()
        output = self.forward(X.long(), mask).reshape((X.shape[0], -1))

        labels = torch.arange(output.shape[0])
#        print(labels.shape)
#        print(output.shape)

        loss_i = torch.nn.CrossEntropyLoss()(output, labels)
        loss_t = torch.nn.CrossEntropyLoss()(output, labels)
        loss = (loss_i + loss_t)/2
        return loss

    def predict(self, text):
        mask = torch.triu(torch.ones(1, 1) * float('-inf'), diagonal=1)  

        indexes = self.vocab.get(text.lower().split(" "))
        encoded = torch.zeros((len(indexes), 32)).to(self.device)
        for index, i in enumerate(indexes):
            encoded[index][0] = i
            encoded[index][1:] = self.vocab.PADDING_IDX            
        output = self.forward(encoded.long(), mask)
        return output.reshape((1, -1))
