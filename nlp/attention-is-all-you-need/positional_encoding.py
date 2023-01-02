import numpy as np
import torch 


"""
Why did I not implement this to begin with ?

Genius idea!!!

Positional encoding ftw
"""
def encode(d_model, MAX_SEQ_LEN):
    PE = torch.zeros((MAX_SEQ_LEN, d_model))

    for pos in range(MAX_SEQ_LEN):
        for i in range(d_model//2):
            theta = pos / (10000 ** ((2*i)/d_model))
            PE[pos, 2 * i ] = np.sin(theta)
            PE[pos, 2 * i + 1] = np.cos(theta)
    return PE
