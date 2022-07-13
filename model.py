from xml.sax.xmlreader import InputSource
import torch
import torch.nn as nn
from attention import AttentionLayer

class EncoderModel(torch.nn.Module):
    def __init__(self, input_size):
        super(EncoderModel, self).__init__()

        self.hidden_size = input_size #// 2
        
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(input_size * self.hidden_size, input_size)

        self.blocks = [
            AttentionLayer(input_size, input_size),
            torch.nn.Linear(input_size, input_size)
        ] * 3

    def forward(self, x, hidden=None):
        hidden = self.get_empty_hidden() if hidden is None else hidden
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        x = output.view(1, -1)
        for i in self.blocks:
            x = torch.nn.functional.normalize(i(x) + x)
        return output, hidden

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderModel(torch.nn.Module):
    def __init__(self, input_size):
        super(DecoderModel, self).__init__()

        self.hidden_size = input_size #// 2
        
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(input_size * self.hidden_size, input_size)

        self.blocks = [
            AttentionLayer(input_size, input_size),
            torch.nn.Linear(input_size, input_size)
        ] * 3

    def forward(self, x, hidden=None):
        hidden = self.get_empty_hidden() if hidden is None else hidden
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded.type(torch.float)
        hidden = hidden.type(torch.float)
    
        output, hidden = self.gru(output, hidden)
    
        x = output.view(1, -1)
        for i in self.blocks:
            x = torch.nn.functional.normalize(i(x) + x)
        return output, hidden

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

