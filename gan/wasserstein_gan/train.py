from optimization_playground_shared.models.SimpleGenerator import SimpleGenerator
from discriminator import Discriminator
import torch
import torchvision
from parameters import Parameters
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from plot import plot
from torch import nn
import numpy as np

device = torch.device('cuda')

def train(parameters: Parameters, generator, discriminator, dataloader):
    opt_g = torch.optim.RMSprop(
        generator.parameters(), 
        lr=parameters.LEARNING_RATE
    )
    opt_d = torch.optim.RMSprop(
        discriminator.parameters(), 
        lr=parameters.LEARNING_RATE
    )

    for current_epoch in range(1_000):
        for batch, (real, _) in enumerate(dataloader):
            real_images = real.to(device)
            noise = torch.normal(mean=0, std=1,  size=(real_images.shape[0], 100)).to(device)

            real_data_loss = torch.mean(
                discriminator(real_images)
            )
            fake_data_loss = torch.mean(
                discriminator(generator(noise).detach())
            )
            loss = -real_data_loss + fake_data_loss

            opt_d.zero_grad()
            loss.backward()
            opt_d.step()
            # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), (parameters.CLIPPING_PARAMETER) ** 2)
            for p in discriminator.parameters():
                p.data.clamp_(-parameters.CLIPPING_PARAMETER, parameters.CLIPPING_PARAMETER)

            if batch % parameters.ITERATIONS_CRITIC == 0:
                loss_generator = - torch.mean(discriminator(generator(noise)))
                opt_g.zero_grad()
                loss_generator.backward()
                opt_g.step()

            if batch % 128 == 0:
                print(f"Loss discriminator {loss.item()}, generator {loss_generator.item()}")

        with torch.no_grad():
            noise = torch.normal(mean=0, std=1,  size=(10, 100)).to(device)
            output = generator(noise)
            grid_image = torchvision.utils.make_grid(output, nrow=10)
            torchvision.utils.save_image(
                grid_image, f"imgs/img_{current_epoch}.png")
            print(f"saved -> imgs/img_{current_epoch}.png")
            plot()

if __name__ == '__main__':
    parameters = Parameters()
    train_ds = MNIST(
        "./", 
        train=True, 
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    )
    train_loader = DataLoader(
        train_ds, batch_size=parameters.BATCH_SIZE, shuffle=True)

    gan = train(
        parameters=parameters,
        discriminator=Discriminator().to(device),
        generator=SimpleGenerator(z=100, normalize=True).to(device),
        dataloader=train_loader,
    )
