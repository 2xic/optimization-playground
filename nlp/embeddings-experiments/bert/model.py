# Based off https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import random


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros((max_len, d_model), requires_grad=False)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:, :x.size(1)]

    
class BertEmbedding(nn.Module):
    def __init__(self, ntoken, d_model: int, PADDING_IDX: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Figure 2 in the paper
        - Token embedding
        - Position embedding
        - Segment embedding
        """
        super().__init__()
        self.encoder = nn.Embedding(
            ntoken, 
            d_model, 
            padding_idx=PADDING_IDX
        )
        self.pos_encoder = PositionalEncoding(
            self.encoder.embedding_dim, 
            dropout
        )
        self.init_weights()
        
    def forward(self, x):
        # TODO: Add segment
        return self.pos_encoder(x) + self.encoder(x)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

class BertModel(nn.Module):
    def __init__(self, vocab, device, nhead: int = 1,
                 nlayers: int = 6, dropout: float = 0.5):
        super().__init__()
        self.vocab = vocab
        self.device = device
        ntoken = self.vocab.size
        self.SEQUENCE_SIZE = 4

        self.embedding = BertEmbedding(ntoken, self.SEQUENCE_SIZE, self.vocab.PADDING_IDX, dropout)
        encoder_layers = TransformerEncoderLayer(self.SEQUENCE_SIZE, nhead)
        # output is (N, T, E)
        # (batch, target sequence, feature number)
        # https://datascience.stackexchange.com/questions/93768/dimensions-of-transformer-dmodel-and-depth
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_vocab = nn.Sequential(*[
            nn.Linear(self.SEQUENCE_SIZE, self.vocab.size),
#            nn.LogSoftmax()
        #    nn.Sigmoid()
        ])

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
#        output = nn.Softmax(dim=2)(self.output_vocab(output))
        output = self.output_vocab(output)
        return output

    def fit(self, X, y, debug=False):
        sz = X.shape[0]# 9 if X.shape[0] >0 else X.shape[0]
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        output = self.forward(X.long(), mask)       
        

        predicted = output.view(X.shape[0] * self.SEQUENCE_SIZE, self.vocab.size)
        target = y.view(-1)

        if random.randint(0, 32) == 0:
            print("=" * 32)
            print("input ", self.vocab.get_words(X[0].tolist()))
            print("model out ", self.vocab.get_words(torch.argmax(predicted[:self.SEQUENCE_SIZE], dim=1).tolist()))
            print("expected x ", self.vocab.get_words(y[0].tolist()))
            print(output[0])
            print(y[0])
            print(predicted[0])
            print(target[0])
        elif debug:
            for i in range(sz):
                print("input ", self.vocab.get_words(X[i].tolist()))
                print("model out ", self.vocab.get_words(torch.argmax(predicted[self.SEQUENCE_SIZE*i:self.SEQUENCE_SIZE*(i + 1)], dim=1).tolist()))
                print("expected x ", self.vocab.get_words(y[i].tolist()))
                print("")

        loss = torch.nn.CrossEntropyLoss(ignore_index=self.vocab.PADDING_IDX)(predicted, target.long())
        """
        loss = torch.nn.NLLLoss(
            ignore_index=self.vocab.PADDING_IDX,
#            reduction='sum'
        )(predicted, target.long())
        """
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
