from torch.distributions import Independent, Normal
from functools import reduce
import math
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Described mostly in section 3.2
"""


# Timesteps between full noise and image
T = 1_000
B_0 = 10 ** -4
B_T = 0.02
size = 28

"""
The beta, lambda terms can just be precomputed and looked up
"""
betas = torch.linspace(B_0, B_T, T, device=device)
alphas = 1 - betas
alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))], device=device)

# Eq. 4
def qt_sample(X_0, t, noise):
    a_bar = alpha_bars[t].reshape(X_0.shape[0], 1, 1, 1)
    mean = (a_bar.sqrt()) * X_0
    sigma = ((1 - a_bar).sqrt()) * noise

    return (
        mean + sigma
    )

def model_sample(predicted_noise, x_t, single_t, shape):
    alpha_bar = alpha_bars[single_t]
    alpha = alphas[single_t]

    fraction = 1 / (alpha.sqrt() )

    eps_fraction = (
        (1 - alpha) /
        (1 - alpha_bar).sqrt()
    )
    mean = fraction * (x_t - eps_fraction * predicted_noise )

    if single_t > 0:
        beta = betas[single_t]
        var = beta.sqrt()

        Z = torch.randn(shape, device=device)
        return mean + var * Z
    else:
        return mean
