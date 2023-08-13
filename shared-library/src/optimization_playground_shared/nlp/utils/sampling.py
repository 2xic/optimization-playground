from torch.distributions import Categorical

def temperature_sampling(values, temperature=.8):
    dist = Categorical(logits=values / temperature)
    return dist.sample()

