"""
The model is suppose to be a generator one, but I want to have it as a classifier
"""
from src.boltzman_machines.rbm import RBM
import torch.optim as optim
import torch

model = RBM(
    3,
    1, 
)
train_op = optim.Adam(model.parameters())

X = torch.tensor([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1] 
]).float()
y = torch.tensor([[0,1,1,0]]).T.float()

for _ in range(1_000):
    _, predicted_y = model.classify(X)
    loss = ((y - predicted_y) ** 2).mean()
    train_op.zero_grad()
    loss.backward()
    train_op.step()

_, predicted = model.classify(X)
print(predicted)
