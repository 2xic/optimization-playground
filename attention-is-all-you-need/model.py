from xml.sax.xmlreader import InputSource
import torch
import torch.nn as nn
from attention import AttentionLayer

N_LAYERS = 2
DIVIDER = 1 # 32 for large models

class EncoderModel(torch.nn.Module):
    def __init__(self, input_size, vocab_size, device=torch.device('cpu')):
        super(EncoderModel, self).__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.block_size = vocab_size // DIVIDER
        self.hidden_size = input_size // DIVIDER
        
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.block_size)

        self.blocks = nn.Sequential(*[
            AttentionLayer(self.block_size, self.block_size),
            torch.nn.Linear(self.block_size, self.block_size)
        ] * N_LAYERS)
        self.device = device

    def forward(self, x, hidden=None):
        hidden = self.get_empty_hidden() if hidden is None else hidden

        x = self.embedding(x).view(1, 1, -1)
        x, hidden = self.gru(x, hidden)
        x = x.view(1, -1)
        for i in self.blocks.children():
            x = torch.nn.functional.normalize(i(x) + x)
        return x, hidden

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.block_size, device=self.device)

class DecoderModel(torch.nn.Module):
    def __init__(self, input_size, vocab_size, device=torch.device('cpu')):
        super(DecoderModel, self).__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = input_size // DIVIDER
        self.block_size = vocab_size // DIVIDER

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.block_size)
        
        self.blocks = nn.Sequential(*[
            AttentionLayer(self.block_size, self.block_size),
            torch.nn.Linear(self.block_size, self.block_size)
        ] * N_LAYERS)
        self.last_bock = torch.nn.Linear(self.block_size, vocab_size)
        self.device = device

    def forward(self, x, encoder, hidden=None):
        hidden = self.get_empty_hidden() if hidden is None else hidden.type(torch.long)
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded.type(torch.float)
        hidden = hidden.type(torch.float)
    
        output, hidden = self.gru(output, hidden)
        x = output.view(1, -1)
#        for i in self.blocks:
        for index, i in enumerate(self.blocks.children()):
            if index == 1:
                x = torch.nn.functional.normalize(i(x) + x + encoder[0])
            else:
                x = torch.nn.functional.normalize(i(x) + x)

        x = self.last_bock(x)
        return torch.softmax(x, dim=1), hidden

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.block_size, device=self.device).type(torch.long)

