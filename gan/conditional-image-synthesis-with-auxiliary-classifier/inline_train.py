# from https://arxiv.org/pdf/1406.2661.pdf

from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn.functional as F
from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def forward(generator, discriminator, x_real, y_real, is_generator_loss):
    y_real = F.one_hot(y_real, num_classes=10)

    one_hot_class_labels = torch.arange(0, 10).repeat(x_real.shape[0])[:x_real.shape[0]] #.reshape((x_real.shape[0], 10))

    x_fake = torch.normal(mean=0, std=1,  size=(x_real.shape[0], 100))
    y_fake = F.one_hot(one_hot_class_labels, num_classes=10)
        
    

    x_fake = generator(x_fake, y_fake)
    (discriminator_real_source ,discriminator_real_labels) = discriminator(x_real)
    (discriminator_fake_source, discriminator_fake_labels) = discriminator(x_fake)

    if is_generator_loss:
        discriminator_real_source.detach()
        discriminator_real_labels.detach()
        discriminator_fake_source.detach()
        discriminator_fake_labels.detach()
    else:
        x_fake.detach()

    l_s_real = F.binary_cross_entropy(
        discriminator_real_source.reshape((-1, 1)),
        torch.ones(x_real.shape[0], 1)
    )
    l_s_fake = F.binary_cross_entropy(
        discriminator_fake_source.reshape((-1, 1)),
        torch.zeros(x_fake.shape[0], 1)
    )

    l_s_loss = (l_s_real + l_s_fake)

    l_c_real = F.binary_cross_entropy(
        discriminator_real_labels,
        y_real.float()
    )
    l_c_fake = F.binary_cross_entropy(
        discriminator_fake_labels,
        y_fake.float()
    )

    l_c_loss = l_c_real + l_c_fake

    generator_loss = l_s_loss - l_c_loss
    discriminator_loss = l_s_loss + l_c_loss
    
    if is_generator_loss:
        return generator_loss
    return discriminator_loss

def train(generator, discriminator):
    lr = 1e-3
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True)
    for current_epoch in range(25):
        for batch, (x_real, y_real) in enumerate(train_loader):
            total_discriminator_loss = torch.tensor(0).float()
            for _ in range(16):
                discriminator_loss = forward(
                    generator,
                    discriminator,
                    x_real,
                    y_real,
                    is_generator_loss=False
                )

                discriminator.zero_grad()
                discriminator_loss.backward()
                total_discriminator_loss += discriminator_loss.item()
                opt_d.step()

            generator_loss = forward(
                generator,
                discriminator,
                x_real,
                y_real,
                is_generator_loss=True
            )
            generator.zero_grad()
            generator_loss.backward()
            opt_g.step()

            if  batch % 25 == 0:
                print(f"\tg_loss {generator_loss}, d_loss {total_discriminator_loss/10}")

        noise = torch.normal(mean=0, std=1,  size=(10, 100))
        one_hot_class_labels = torch.arange(0, 10)
        y_fake = F.one_hot(one_hot_class_labels, num_classes=10)

        output = generator(noise, y_fake)
        grid_image = torchvision.utils.make_grid(output)
        torchvision.utils.save_image(
            grid_image, f"imgs/img_{current_epoch}.png")
        print(f"saved -> imgs/img_{current_epoch}.png")

if __name__ == '__main__':
    discriminator = Discriminator()
    generator = Generator(z=100)
    gan = train(
        discriminator=discriminator,
        generator=generator
    )
