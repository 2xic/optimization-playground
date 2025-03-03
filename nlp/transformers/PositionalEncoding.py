import numpy as np
import matplotlib.pyplot as plt
import torch

class PositionalEncoding:
    def encode_tensor(self, sequence_size, d_model):
        position = torch.arange(sequence_size)[:, torch.newaxis]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros((sequence_size, d_model))
        pe[:, 0::2] = torch.sin(torch.tensor(position * div_term))
        pe[:, 1::2] = torch.cos(torch.tensor(position * div_term))
        return pe
    
if __name__ == "__main__":
    plt.imshow(PositionalEncoding().encode_tensor(1000, 512).T + torch.ones(1000))
    plt.savefig('PositionalEncoding.png')
