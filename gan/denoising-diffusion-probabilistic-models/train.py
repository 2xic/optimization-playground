import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from parameters import *
from unet_model import UNet
from model import Diffusion
from plot import plot

transform = transforms.Compose([
        transforms.ToTensor(),
    ]
)
train_ds = MNIST("./", train=True, download=True,
                    transform=transform)

#idx = train_ds.targets == 1
#train_ds.data = train_ds.data[idx]
#train_ds.targets = train_ds.targets[idx]

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True
)

model = UNet().to(device) 
#model = Diffusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.091)
iterations = 10_000
epoch_print = 5_00

"""
Reading the paper
-> Training is performed by optimizing the usual variational bound negative log likelihood
"""

current_iteration = 0
for epoch in range(20):
    for (X, y) in (train_loader):
        # Line 3 and Line 4
        X_0 = X.to(device)
        epoch_t = torch.randint(1, T, size=(X.shape[0], 1), device=device)
        noise = torch.randn(X_0.shape, device=device)

        # Returns the function of line 5 in the algorithm
        noisy_image = qt_sample(X_0, epoch_t, noise).to(device)

        y = model(
            noisy_image,
            epoch_t,
        )

        optimizer.zero_grad()
        error = torch.nn.MSELoss()(y, noise)
        error.backward()
        optimizer.step()

        if current_iteration % epoch_print == 0:
            print(current_iteration, error)

        if iterations < current_iteration:
            break
        current_iteration += 1
    print("New epoch :)")

    plot(
        X_0,
        model,
        device,
        T=T,
        sample_size=32,
        epoch=epoch
    )
