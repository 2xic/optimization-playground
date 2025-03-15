from torch.distributions import Categorical
import torch


def temperature_sampling(values: torch.Tensor, temperature: float = 0.95):
    scaled_logits = values / temperature    
    probs = torch.softmax(scaled_logits, dim=-1)    
    dist = Categorical(probs=probs)
    sampled_values = dist.sample()
    return sampled_values

def argmax_sampling(values: torch.Tensor):
    return values.argmax(dim=-1)
