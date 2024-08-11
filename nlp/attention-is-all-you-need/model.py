from xml.sax.xmlreader import InputSource
import torch
import torch.nn as nn
from attention import AttentionLayer
from positional_encoding import encode

N_LAYERS = 4
DIVIDER = 1 # 32 for large models

class EncoderModel(torch.nn.Module):
    def __init__(self, input_size, vocab_size, device=torch.device('cpu')):
        super(EncoderModel, self).__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.block_size = vocab_size // DIVIDER
        self.hidden_size = input_size // DIVIDER
        
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.shape_2 = self.hidden_size# * self.hidden_size

        self.blocks = nn.Sequential(*[
            AttentionLayer(self.shape_2, self.shape_2),
            nn.Tanh(),
            torch.nn.Linear(self.shape_2, self.shape_2),
            nn.Tanh(),
        ] * N_LAYERS)
        self.device = device

    def forward(self, x, hidden=None):
        hidden = self.get_empty_hidden() if hidden is None else hidden

        x = self.embedding(x).reshape(x.shape[0], self.hidden_size)
        x += encode(
            self.hidden_size, 
            x.shape[0]
        )
        for i in self.blocks.children():
            x = torch.nn.functional.normalize(i(x) + x)
        return x

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.block_size, device=self.device)


#class ContextEncoding 

class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size):
        super(DecoderLayer, self).__init__()
        self.hidden_size = hidden_size

        self.blocks_in = nn.Sequential(*[
            AttentionLayer(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        ])
        self.blocks_encoder = nn.Sequential(*[
            AttentionLayer(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        ])
        self.output = nn.Sequential(*[
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        ])

    def forward(self, input, encoder):
        x = self.blocks_in(input)
        add_x = torch.nn.functional.normalize(input + x)
        # 
        x = self.blocks_encoder(add_x + encoder)
        add_x = torch.nn.functional.normalize(x + add_x)
        # 
        x = self.output(x)
        x = torch.nn.functional.normalize(x + add_x)
        return x

class DecoderModel(torch.nn.Module):
    def __init__(self, input_size, vocab_size, device=torch.device('cpu')):
        super(DecoderModel, self).__init__()

        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = input_size // DIVIDER
        self.block_size = vocab_size // DIVIDER

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.shape2  = self.hidden_size * self.hidden_size
        """
        Nooo, AttentionLayer should have the context of the encoder
        -> https://www.tensorflow.org/text/tutorials/transformer
        """
        self.blocks = nn.Sequential(*[
            DecoderLayer(self.hidden_size)#, self.hidden_size),
        ] * N_LAYERS)
        self.last_bock = torch.nn.Linear(self.hidden_size, vocab_size)
        self.device = device

    def forward(self, x, encoder, hidden=None):
        #print(x.shape)
        embedded = self.embedding(x).reshape(x.shape[0], self.hidden_size)
        x = embedded
        x += encode(
            self.hidden_size, 
            x.shape[0]
        )
        for _, i in enumerate(self.blocks.children()):
            x = i(x, encoder) #torch.nn.functional.normalize(i(x + encoder) + x)

        x = self.last_bock(x)
        return torch.softmax(x, dim=1)#, hidden

    def get_empty_hidden(self):
        return torch.zeros(1, 1, self.block_size, device=self.device).type(torch.long)

