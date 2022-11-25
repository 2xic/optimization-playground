import torch
from model import Model



"""
x -> 
"""

X = torch.zeros((28, 28))
print(X)

model = Model().split_forward(X.reshape((28 * 28)))
