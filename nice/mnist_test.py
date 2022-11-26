import torch
from model import Model
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from helper import sample_z, generate_mask
"""
x -> 
"""

train_ds = MNIST("./", train=True, download=True,
                 transform=transforms.ToTensor())
X = torch.zeros((28, 28))
model = Model()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(20):
    for i in range(1_000):
        # as mentinoned in section 5.6
        X,_ = train_ds[i] 
        X = X.reshape((28 * 28))
        X += torch.rand((28 * 28)) / 256
        X = torch.clamp(X, 0, 1)


        y = model.forward(X)
        
        loss = (-y.mean())
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(loss.item())    

sample = sample_z()
x = model.backward(sample)

plt.imshow(x.detach().numpy().reshape((28, 28)))
plt.show()
