from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from model import Diffusion
import random
import math
import torchvision
from parameters import *

train_ds = MNIST("./", train=True, download=True,
                    transform=transforms.ToTensor())
train_loader = DataLoader(
    train_ds, 
    batch_size=32, 
    shuffle=True
)

model = Diffusion()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
iterations = 3_000
epoch_print = 1_00


current_iteration = 0
while current_iteration < iterations:
    for (X, y) in (train_loader):
        X_0 = X.reshape((X.shape[0], 28 * 28))
        epoch_t = random.randint(1, T)

        noise = torch.randn_like(X_0)
        x_t = qt_sample(X_0, epoch_t, noise) #sample_noise(X_0.shape[0])

        #param = torch.sqrt(lambda_t_sub(epoch_t) * X_0) +\
         #       torch.sqrt(1 - lambda_t_sub(epoch_t) * noise)
        #param = torch.nan_to_num(param)

        computed = lambda_t_sub(epoch_t) **0.5 * X_0  +\
            ((1 - lambda_t_sub(epoch_t)) ** 0.5 * noise )
        y = model(
            computed, #param,
            torch.tensor([epoch_t / T]).reshape((1, -1)).float()
        )

        optimizer.zero_grad()
        error = torch.nn.MSELoss()(y, noise) # ((y - noise) ** 2).sum() / y.shape[0]
        error.backward()
        optimizer.step()

        if current_iteration % epoch_print == 0:
            print(error)

        if iterations < current_iteration:
            break
        current_iteration += 1
     #   break
    #break

noise = torch.rand_like(torch.zeros((1, 28 * 28)))
for i in range(T, 0, -1):
    Z = 0
    if i > 1:
        Z = torch.rand_like(noise)
    print(i)
    noise = sample(
        model(
            noise, 
            torch.tensor([i / T]).reshape((1, -1)).float()
        ),
        noise,
        i,
        Z
    )
    print(noise)

noise = noise.reshape((1, 28, 28)) 
noise -= torch.min(noise)
noise /= torch.max(noise)

#print(noise)
grid_image = torchvision.utils.make_grid(noise)
torchvision.utils.save_image(
    grid_image, 
    f"example.png"
)
