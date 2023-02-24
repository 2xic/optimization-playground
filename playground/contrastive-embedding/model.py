from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLstm(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.hidden_size = 2
        self.input_size = 64
        self.embedding = nn.Embedding(vocab, 7, padding_idx=0)
        self.lstm = nn.LSTM(input_size=7, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.linear_lstm = nn.Linear(
            self.input_size * self.hidden_size, 
            256
        )
        self.linear_middle = nn.Linear(
            256, 
            512
        )
        self.linear_out = nn.Linear(
            512, 
            32
        )

    def forward(self, x):
        input_shape = x.shape[-1]
        assert input_shape == self.input_size
        x = self.embedding(x)
        x, (hidden, output) = self.lstm(x)
        x = x.reshape((x.shape[0], input_shape * self.hidden_size))
        x = F.relu(self.linear_lstm(x))
        x = F.relu(self.linear_middle(x))
        x = F.relu(self.linear_out(x))
        return x
