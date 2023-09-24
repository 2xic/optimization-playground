import torch
from torch import nn
from dataclasses import dataclass
import os
from collections import defaultdict

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
            input_size=config.embedding_dim,
            hidden_size=512,
            num_layers=4,
            batch_first=True
        )
        self.activation_1 = nn.Sequential(*[
            nn.Linear(
                512, 512
            ),
            nn.Tanh(),
        ])
        self.activation_2 = nn.Sequential(*[
            nn.Linear(
                512, 512
            ),
            nn.Tanh(),
        ])
        self.activation_3 = nn.Sequential(*[
            nn.Linear(
                512, 512
            ),
            nn.Tanh(),
        ])
        self.activation_4 = nn.Sequential(
            nn.Linear(
                512, config.tokens
            ),
          #  nn.Softmax(dim=1)
        )
    
    def forward(self, x, hidden=None):
        x = self.embeddings(x)
        (x, hidden) = self.lstm(x, hidden)
        x = self.activation_1(x)
        x = self.activation_2(x)
        x = self.activation_3(x)
        x = self.activation_4(x)
        return x, hidden
        
    def fit(self, x, y, debug=False):
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        loss = 0
        hidden = None
        input_tokens_count = defaultdict(int)
        output_tokens_count = defaultdict(int)
        for index in range(x.shape[1]):
            x_tokens = x[:, index].reshape((-1, 1))
            y_tokens = y[:, index].reshape((-1, 1))
            raw_output, hidden = self.forward(x_tokens, hidden)
            output = raw_output.view(
                self.config.sequence_size * x_tokens.shape[0],
                self.config.tokens
            )
            loss += torch.nn.CrossEntropyLoss(ignore_index=self.config.padding_index)(output, y_tokens.view(-1))
            if debug:
                x_token = torch.argmax(raw_output, dim=-1).reshape((-1))
                for i in x_token.tolist():
                    input_tokens_count[i] += 1
                y_tokens = y_tokens.reshape((-1))
                for i in y_tokens.tolist():
                    output_tokens_count[i] += 1
        if debug:
            print(input_tokens_count)
            print(output_tokens_count)
        return loss
    
    def predict(self, seed, predict=128, debug=False):
        assert len(seed.shape) == 2
        assert seed.shape[0] == 1, "single batch size plz"
        hidden = None
        x_token = None
        y_tokens = []
        for index in range(predict):
            # this is the seed
            if index < seed.shape[1]:
                x_token = seed[0][index].reshape((1, 1))
                y_tokens.append(x_token.item())
            else:
                y_tokens.append(x_token.item())
            output, hidden = self.forward(x_token, hidden)
            output = output.view(
                self.config.sequence_size * x_token.shape[0],
                self.config.tokens
            )
            x_token = torch.argmax(output, dim=-1).reshape((1, 1))
            if debug:
                print(output)
        y_tokens.append(x_token.item())
        return y_tokens
    
    def forward_feed(self, forward_pass):
        assert len(forward_pass.shape) == 2
        assert forward_pass.shape[0] == 1, "single batch size plz"
        hidden = None
        x_token = None
        y_tokens = []
        for index in range(forward_pass.shape[-1]):
            if index < forward_pass.shape[1]:
                x_token = forward_pass[0][index].reshape((1, 1))
            output, hidden = self.forward(x_token, hidden)
            output = output.view(
                self.config.sequence_size * x_token.shape[0],
                self.config.tokens
            )
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
