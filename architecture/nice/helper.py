import torch
from torch.distributions import Uniform
import torch.nn.functional as F

def get_prior(x):
    return -(F.softplus(x) + F.softplus(-x))

    return - torch.log(
        1 + torch.exp(x)
    ) - torch.log(
        1 + torch.exp(-x)
    )

def sample_z(device):
#    z = torch.rand([1, 28 * 28], device=device)
    z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample((1, 28 * 28))
    z = z.to(device)
    sample = torch.log(z) - torch.log(1. - z)
    sample = sample.reshape((28 * 28))
    return sample


def generate_mask(start_odd, device):
    # as mentioned in section 5.1
    mask = torch.zeros((28 * 28), device=device)
    mask[::2] = 1
    if start_odd:
        mask = 1 - mask
    return mask
