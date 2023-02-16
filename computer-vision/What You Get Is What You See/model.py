from torch import optim, nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms

def get_hidden(hidden_size):
    return torch.zeros(1, 1, hidden_size)

class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 144
        self.conv1 = nn.Conv2d(4, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, x, hidden=None):
      #  print(x.shape)
        if hidden == None:
            hidden = get_hidden(self.hidden_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, 1, -1)

        output, hidden = self.gru(x, hidden)
        return output, hidden

class DecoderModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        x = self.embedding(input).view(1, 1, -1)
        x = F.relu(x)

        output, hidden = self.gru(x, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

class DecoderModelAttn(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = 0.1
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        if hidden is None:
            hidden = get_hidden(self.hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.output(output[0]), dim=1)
        return output, hidden
