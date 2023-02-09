from torch import nn
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 2
        self.input_size = 5 
        self.embedding = nn.Embedding(100, 7, padding_idx=0)
        self.lstm = nn.LSTM(input_size=7, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.linear_out = nn.Linear(
            self.input_size * self.hidden_size, 
            self.input_size
        )

    def forward(self, x):
        input_shape = x.shape[-1]
        assert input_shape == self.input_size
        x = self.embedding(x)
        x, (hidden, output) = self.lstm(x)
        x = x.reshape((x.shape[0], input_shape * self.hidden_size))
        x = self.linear_out(x)
        x = (F.sigmoid(x) * 9)
        return x
