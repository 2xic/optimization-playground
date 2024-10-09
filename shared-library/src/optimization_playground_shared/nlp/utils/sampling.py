from torch.distributions import Categorical
import torch

def temperature_sampling(values, temperature=.95):
    values = values.squeeze(0)
    assert len(values.shape) == 1 
    scaled_logits = values / temperature
    probs = torch.softmax(scaled_logits, dim=0)
    dist = Categorical(probs=probs)
    return dist.sample()

def argmax_sampling(values: torch.Tensor):
    return values.argmax(dim=1)
