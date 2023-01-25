from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from model import Diffusion
import torchvision
from parameters import *
from unet_model import UNet

transform = transforms.Compose([
        transforms.ToTensor(),
   #     transforms.Lambda(lambda x: (x - 0.5) * 2)
    ]
)
train_ds = MNIST("./", train=True, download=True,
                    transform=transform)
train_loader = DataLoader(
    train_ds, 
    batch_size=512, 
    shuffle=True
)

model = UNet().to(device) #
#model = Diffusion().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.091)
iterations = 10_000
epoch_print = 5_00

"""
Reading the paper 
-> Training is performed by optimizing the usual variational bound negative log likelihood
"""

current_iteration = 0
#while current_iteration < iterations:
for epoch in range(20):
    for (X, y) in (train_loader):
        # X_0 sample -> Line 2
        #X_0 = X.reshape((X.shape[0], 28 * 28)).to(device)
        # Line 3 and Line 4
        X_0 = X.to(device)
        epoch_t = torch.randint(1, T, size=(X.shape[0], 1), device=device)#.float()
        noise = torch.randn_like(X_0, device=device)

        # Returns the function of line 5 in the algorithm
        # OKAY, we try with 
        x_t = (qt_sample(X_0, epoch_t, noise).to(device))

   #     print(x_t.shape)
        y = model(
            x_t, #.reshape((X.shape[0], 1, 28 , 28)),
            epoch_t,
        )
        # Optimize the optimizer.
        optimizer.zero_grad()
        error = torch.nn.MSELoss()(y, noise)
        error.backward()
        optimizer.step()

        if current_iteration % epoch_print == 0:
            print(current_iteration, error)
           # print(y)

        if iterations < current_iteration:
            break
        current_iteration += 1
    print("New epoch :)")
"""
Goes from noise to image
P(X_(t - 1) | X_t ) -> 

X_t = noise
X_0 = Image
"""
with torch.no_grad():
    sample_size = 16
    noise = (torch.rand_like(torch.zeros((sample_size, 1, 28 , 28)), device=device))
    for i in range(T, 0, -1):
        Z = 0
        if i > 1:
            Z = torch.rand_like(noise, device=device)
        noise = sample(
            model(
                noise.reshape((sample_size, 1, 28 , 28)),
                torch.zeros((sample_size, 1), device=device).fill_(i).long()
            ),
            noise,
            torch.zeros((sample_size, 1), device=device).fill_(i).long(),
            Z
        )

    noise = noise.reshape((sample_size, 1, 28, 28)) 

    grid_image = torchvision.utils.make_grid(noise[:sample_size])
    torchvision.utils.save_image(
        grid_image, 
        f"example_generated_raw.png"
    )

    noise -= torch.min(noise)
    noise /= torch.max(noise)
    noise = noise.cpu()

    grid_image = torchvision.utils.make_grid(noise[:sample_size])
    torchvision.utils.save_image(
        grid_image, 
        f"example_generated.png"
    )

    noise_example = torch.rand_like(torch.zeros((sample_size, 1, 28, 28)), device=device)
    grid_image = torchvision.utils.make_grid(noise_example)

    torchvision.utils.save_image(
        grid_image, 
        f"example_fake_noise.png"
    )

    epoch_t = torch.randint(1, T, size=(X.shape[0], 1), device=device)
    noise = torch.randn_like(X_0, device=device)
    noise_example = qt_sample(X_0, epoch_t, noise).to(device)  
    grid_image = torchvision.utils.make_grid(noise_example[:sample_size])
    torchvision.utils.save_image(
        grid_image, 
        f"example_real_noise.png"
    )
