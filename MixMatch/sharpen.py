import torch

def sharpen(y, T=0.3):
    return (y**(1/T) / torch.sum(y**(1/T)))

