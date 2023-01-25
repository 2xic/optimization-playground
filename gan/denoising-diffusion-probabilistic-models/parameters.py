from torch.distributions import Independent, Normal
from functools import reduce
import math
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

"""
Described mostly in section 3.2
"""


# Timesteps between full noise and image
T = 1000

B_0 = 10 ** -4
B_T = 0.2
size = 28

loc = torch.zeros((size, size), device=device)
scale_diag = torch.ones((size, size), device=device)

"""
The beta, lambda terms can just be precomputed and looked up
"""
betas = torch.linspace(B_0, B_T, T + 1, device=device)
alphas = 1 - betas
alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))], device=device)


# Eq. 4
def qt_mean(X_0, t):
    single_shape = alpha_bars[t]
    single_shape = torch.concat((
        single_shape,
    ) * 28 * 28, dim=1).reshape((X_0.shape)).to(device)

    mean = single_shape ** 0.5 * X_0
    var = 1 - single_shape

    return mean, var
    
def qt_sample(X_0, t, noise):
    mean, var = qt_mean(X_0, t)

    return (
        mean + (var ** 0.5) * noise
    )

# Eq. 7
def mu(x_0, t, x_t):
    first = (
        math.sqrt(
            alpha_bars[(t - 1)]
        ) * betas[t] /
        (1 - alpha_bars[(t)])
    ) * x_0
    second = (
        (alphas[t]) ** 0.5 * 
        (1 - alphas[(t - 1)])
    ) / (1 - alphas[(t)]) * x_t

    return (
        first + second
    )

def sample(eps, x_t, t, z):
    single_shape = alpha_bars[t]
    single_shape = torch.concat((
        single_shape,
    ) * 28 * 28, dim=1).reshape((eps.shape)).to(device)

    single_shape_alphas = alphas[t]
    single_shape_alphas = torch.concat((
        single_shape_alphas,
    ) * 28 * 28, dim=1).reshape((eps.shape)).to(device)

    fraction = 1 / (single_shape_alphas) ** 0.5
    eps_fraction = (
        (1 - single_shape_alphas) / 
        # todo -> should this zero out ? 
        (1 - single_shape) ** 0.5 
    )
    #print(fraction.shape)
    #print(eps.shape)
    #print(eps_fraction.shape)
    mean = fraction * (x_t - eps * eps_fraction )

    single_shape_betas = betas[t]
    single_shape_betas = torch.concat((
        single_shape_betas,
    ) * 28 * 28, dim=1).reshape((eps.shape)).to(device)

    var = single_shape_betas ** 0.5 * z

    return (
        mean + var 
    )
