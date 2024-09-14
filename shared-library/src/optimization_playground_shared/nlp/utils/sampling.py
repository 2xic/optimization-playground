from torch.distributions import Categorical
import torch

def temperature_sampling(values, temperature=.8):
    dist = Categorical(logits=values / temperature)
    return dist.sample()

def argmax_sampling(values: torch.Tensor):
    return values.argmax(dim=1)
