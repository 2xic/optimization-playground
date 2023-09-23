import torch
from torch import nn
from dataclasses import dataclass
import os

@dataclass
class Config:
    padding_index: int
    tokens: int
    embedding_dim = 8
    sequence_size = 1

    def __init__(self, tokens, padding_index, sequence_size):
        self.padding_index = padding_index
        self.tokens = tokens
        self.sequence_size = sequence_size

class Model(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(
            embedding_dim=config.embedding_dim,
            num_embeddings=config.tokens,
            padding_idx=config.padding_index
        )
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        self.activation = nn.Sequential(
            nn.Linear(
                512, 512
            ),
            nn.Tanh(),
            nn.Linear(
                512, config.tokens
            ),
          #  nn.Softmax(dim=1)
        )
    
    def forward(self, x, hidden=None):
        x = self.embeddings(x)
        (x, hidden) = self.lstm(x, hidden)
        x = self.activation(x)
        return x, hidden
        
    def fit(self, x, y):
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        loss = 0
        hidden = None
        for index in range(x.shape[1]):
            x_tokens = x[:, index].reshape((-1, 1))
            y_tokens = y[:, index].reshape((-1, 1))
            output, hidden = self.forward(x_tokens, hidden)
            output = output.view(
                self.config.sequence_size * x_tokens.shape[0],
                self.config.tokens
            )
            loss += torch.nn.CrossEntropyLoss(ignore_index=self.config.padding_index)(output, y_tokens.view(-1))
        return loss
    
    def predict(self, seed, predict=128, debug=False):
        assert len(seed.shape) == 2
        assert seed.shape[0] == 1, "single batch size plz"
        hidden = None
        x_token = None
        y_tokens = []
        for index in range(predict):
            if index < seed.shape[1]:
                x_token = seed[0][index].reshape((1, 1))
            output, hidden = self.forward(x_token, hidden)
            output = output.view(
                self.config.sequence_size * x_token.shape[0],
                self.config.tokens
            )
            x_token = torch.argmax(output, dim=1).reshape((1, 1))
            if debug:
                print(output)
            if seed.shape[1]< index:
                y_tokens.append(x_token.item())
        return y_tokens

    def save(self):
        torch.save({
            'params': self.state_dict()
        }, 'model.pkt')

    def load(self):
        if os.path.isfile('model.pkt'):
            try:
                self.load_state_dict(torch.load('model.pkt')['params'])
                return True
            except Exception as e:
                print(e)
        return False
