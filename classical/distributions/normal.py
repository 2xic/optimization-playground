"""
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence


https://www.johndcook.com/blog/2017/11/08/why-is-kullback-leibler-divergence-not-a-distance/
- KL divergence is the distance between two distributions
- It’s clearly zero if X and Y have the same distribution.

https://www.johndcook.com/blog/2023/11/05/kl-divergence-normal/
- 
"""
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 
import statistics 


def get_normal(data):
    mean = statistics.mean(data) 
    sd = statistics.stdev(data) 
    x = np.arange(np.min(data), np.max(data))

    return x, norm.pdf(x, mean, sd)

def kl_divergence(a, b):
    delta = 0
    for i in range(a.shape[0]):
        delta += a[i] * np.log2(a[i] / b[i])
    return delta

a, a_f = get_normal(np.arange(-20, 20, 0.01) )
b, b_f = get_normal(np.arange(-20, 20, 0.01)  + 20)
print(a_f)
print(b_f)

# There shouldn't be any error here lol ? 
# I mean... because I messed up the distance function hehe
print("Kl ", kl_divergence(a_f, b_f))
print("Kl (torch) ", torch.nn.KLDivLoss(reduction='batchmean')(torch.from_numpy(a_f), torch.from_numpy(b_f)).item())

plt.plot(a, a_f, label="A") 
plt.plot(b, b_f, label="B") 

plt.legend(loc="upper left")
plt.savefig('normal.png') 
