import torch
from torch import nn
from dataclasses import dataclass, asdict
import os
from collections import defaultdict

@dataclass
class Config:
    padding_index: int
    tokens: int
    embedding_dim: int
    sequence_size: int

    def __init__(self, tokens, padding_index, sequence_size=1, embedding_dim=8):
        self.padding_index = padding_index
        self.tokens = tokens
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim

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
            hidden_size=1024,
            num_layers=4,
            batch_first=True
        )
        self.activation_1 = nn.Sequential(*[
            nn.Linear(
                1024, 2048
            ),
            nn.Tanh(),
        ])
        self.activation_2 = nn.Sequential(*[
            nn.Linear(
                2048, 1024
            ),
            nn.Tanh(),
        ])
        self.activation_3 = nn.Sequential(*[
            nn.Linear(
                1024, 512
            ),
            nn.Tanh(),
        ])
        self.activation_4 = nn.Sequential(
            nn.Linear(
                512, config.tokens
            ),
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
        # Does a forward, but only predicts one token at the time. 
        # Does not use the model preidction to predict next token
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
            'params': self.state_dict(),
            'config': asdict(self.config),
        }, 'model.pkt')

    @staticmethod
    def load():
        if os.path.isfile('model.pkt'):
            try:
                data = torch.load('model.pkt')
                config = Config(**data["config"])
                print(config)
                print(data["config"])
                model = Model(config)
                model.load_state_dict(data['params'])
                return model
            except Exception as e:
                print(e)
        return None
