from torch.distributions import Independent, Normal
from functools import reduce
import math
import torch

"""
Described mostly in section 3.2
"""


# Timesteps between full noise and image
T = 5

B_0 = 10 ** -4
B_T = 0.2
size = 28

loc = torch.zeros((size, size))
scale_diag = torch.ones((size, size))

beta_t = lambda t: ((B_T - B_0) / T) * t
lambda_t = lambda t: 1 - beta_t(t)
lambda_t_sub = lambda t: reduce(lambda x, y: x * y, [lambda_t(i) for i in range(t)], 1)
sigma_t_square = lambda t: beta_t(t) # (1 - lambda_t_sub(t-1)) / (1 - lambda_t_sub(t)) * beta_t(t)
sigma_t = lambda t: math.sqrt(sigma_t_square(t))

#normal = Normal(loc, scale_diag) # Independent(Normal(loc, scale_diag), 1)
#def sample_noise(batch):
#    noise = normal.sample(sample_shape=(batch, )).reshape((batch, 28 * 28))
#    noise = torch.nan_to_num(noise)
#    return noise

# Eq. 4
def qt_mean(X_0, t):
    mean = math.sqrt(lambda_t_sub(t)) * X_0
    var = 1 - lambda_t_sub(t)

    return mean, var
    
def qt_sample(X_0, t, noise):
    #noise = torch.randn_like(X_0)
    mean, var = qt_mean(X_0, t)

    return (
        mean + (var ** 0.5) * noise
    )

# Eq. 7
def mu(x_0, t, x_t):
    first = (
        math.sqrt(
            lambda_t_sub(t - 1)
        ) * beta_t(t) /
        (1 - lambda_t_sub(t))
    ) * x_0
    second = (
        math.sqrt(lambda_t(t)) * 
        (1 - lambda_t(t - 1))
    ) / (1 - lambda_t(t)) * x_t

    return (
        first + second
    )

def sample(eps, x_t, t, z):
    fraction = 1 / (lambda_t(t)) ** 0.5
    eps_fraction = (
        (1 - lambda_t(t)) / 
        # todo -> should this zero out ? 
        ((math.sqrt(1 - lambda_t_sub(t))) + 0.00001) 
    )
    mean = fraction * (x_t - eps * eps_fraction )
    var = sigma_t(t) * z
    return (
        mean + var 
    )
