# Based off https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn
import random


class TestTransformerEncoder(nn.Module):
    def __init__(self, ntoken, device, d_model: int = 4, nhead: int = 1,
                 nlayers: int = 6, dropout: float = 0.5):
        super().__init__()
        #self.vocab = vocab
        self.device = device
        #ntoken = self.vocab.size
        self.SEQUENCE_SIZE = 4

        self.embedding =nn.Embedding(
            ntoken, 
            d_model, 
            padding_idx=-1
        )
        encoder_layers = TransformerEncoderLayer(self.SEQUENCE_SIZE, nhead)
        # output is (N, T, E)
        # (batch, target sequence, feature number)
        # https://datascience.stackexchange.com/questions/93768/dimensions-of-transformer-dmodel-and-depth
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.output_vocab = nn.Linear(self.SEQUENCE_SIZE, ntoken)
        self.ntoken = ntoken

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = nn.Sigmoid()(self.output_vocab(output))
        #output = nn.Softmax(dim=2)(output)
        return output

    def fit(self, X, y):
        sz = X.shape[0]
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        output = self.forward(X.long(), mask)

        predicted = output.view(X.shape[0] * self.SEQUENCE_SIZE, self.ntoken)
        target = y.view(-1)
        print("Output / Expected")
        print(predicted, predicted.shape)
        print(target, target.shape)
        print()

        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(predicted, target.long())
        return loss

    def predict(self, X):
        sz = X.shape[0]
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        output = self.forward(X.long(), mask)       
        return output
    