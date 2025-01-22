"""
https://stats.stackexchange.com/questions/306131/how-to-correctly-compute-mutual-information-python-example
https://github.com/xbeat/Machine-Learning/blob/main/Understanding%20Mutual%20Information%20in%20Machine%20Learning%20with%20Python.md

https://en.wikipedia.org/wiki/Mutual_information
https://swh.princeton.edu/~wbialek/our_papers/slonim+al_05b.pdf
"""

import numpy as np
from sklearn.metrics import mutual_info_score
import torch

x = np.random.normal(0, 1, 1000)
y = x + np.random.normal(0, 0.5, 1000)

mi = mutual_info_score(x, y)
print(f"Mutual Information: {mi:.4f}")

"""

"""
def mutual_information(x, y, bins=10):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i][j] != 0:
                p_xy = c_xy[i][j] / np.sum(c_xy)
                p_x = np.sum(c_xy[i, :]) / np.sum(c_xy)
                p_y = np.sum(c_xy[:, j]) / np.sum(c_xy)
#                kl_div = p_xy * torch.distributions.kl.kl_divergence(
#                    [p_x],
#                    [p_y],
#                )
#                mi += kl_div
                # np.log(p_xy / (p_x * p_y)) ~= kl_div
                mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi
mi = mutual_information(x, y)
print(f"Calculated Mutual Information: {mi:.4f}")