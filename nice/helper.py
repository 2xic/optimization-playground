import torch

def get_prior(x):
    return - torch.log(
        1 + torch.exp(x)
    ) - torch.log(
        1 + torch.exp(-x)
    )

def sample_z():
    z = torch.rand([1, 28 * 28])
    sample = torch.log(z) - torch.log(1. - z)
    sample = sample.reshape((28 * 28))
    return sample


def generate_mask(start_odd):
    # as mentioned in section 5.1
    mask = torch.zeros((28 * 28))
    mask[::2] = 1
    if start_odd:
        mask = 1 - mask
    return mask
