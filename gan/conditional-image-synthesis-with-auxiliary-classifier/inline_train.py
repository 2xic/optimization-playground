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


def forward(generator, discriminator, x_real, y_real, is_generator_loss, device):
    x_real = x_real.to(device)
    y_real = F.one_hot(y_real, num_classes=10).to(device)
    x_fake = torch.normal(mean=0, std=1,  size=(x_real.shape[0], 100)).to(device)

    random_labels = torch.argmax(torch.rand(size=(y_real.shape[0], 10)), dim=1)
    y_fake =  F.one_hot(random_labels, num_classes=10).to(device).float() 
     
    x_fake = generator(x_fake, y_fake)
    (discriminator_real_source, discriminator_real_labels) = discriminator(x_real)
    (discriminator_fake_source, discriminator_fake_labels) = discriminator(x_fake)

    if is_generator_loss:
        discriminator_real_source.detach()
        discriminator_real_labels.detach()
        discriminator_fake_source.detach()
        discriminator_fake_labels.detach()
    else:
        x_fake.detach()
    """
    Should learn to determine what is real, and not.

    Standard gan loss
    """
    loss = F.cross_entropy
    l_s_real = torch.nn.L1Loss()(
        discriminator_real_source,
        torch.ones((x_real.shape[0], 1)).float().to(device)
    )
    l_s_fake = torch.nn.L1Loss()(
        discriminator_fake_source,
        torch.zeros((x_fake.shape[0], 1)).float().to(device)
    )
    l_s_loss = (l_s_real + l_s_fake)

    """
    Should learn to determine correct class.

    Which is the part meant to guide.

    Question then is, how should it .

    oh, I think the idea is to reuse the same C for both generator and discriminator!

    But the discriminator should assume everything from the fake source to be wrong. Right?
    So it should be trained on a zero value. At least in the begging.
    """
    y_real = torch.argmax(y_real, dim=1).long()
    l_c_real = loss(
        discriminator_real_labels,
        y_real,
    )
#    y_fake_labels = y_real if is_generator_loss else 
    #y_fake_labels = torch.zeros(y_real.shape[0]).long().to(device) if not is_generator_loss else y_real
    l_c_fake = loss(
        discriminator_fake_labels,
        y_fake,
    )

    l_c_loss = l_c_real + l_c_fake

    generator_loss = -(l_s_loss - l_c_loss)
    discriminator_loss = (l_s_loss + l_c_loss)

    if is_generator_loss:
        return generator_loss
    return discriminator_loss

def train(generator, discriminator, device):
    lr_d = 0.0002 #1e-3
    lr_g = lr_d
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)

    train_ds = MNIST("./", train=True, download=True,
                     transform=transforms.ToTensor())
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True)
    
    forward_loss = forward if True else forward_nnl

    for current_epoch in range(25):
        for batch, (x_real, y_real) in enumerate(train_loader):
            total_discriminator_loss = torch.tensor(0).float()
            train_batch = 4
            for _ in range(train_batch):
                discriminator_loss = forward_loss(
                    generator,
                    discriminator,
                    x_real,
                    y_real,
                    is_generator_loss=False,
                    device=device
                )

                discriminator.zero_grad()
                discriminator_loss.backward()
                total_discriminator_loss += discriminator_loss.item()
                opt_d.step()

            generator_loss = forward_loss(
                generator,
                discriminator,
                x_real,
                y_real,
                is_generator_loss=True,
                device=device
            )
            generator.zero_grad()
            generator_loss.backward()
            opt_g.step()

            if batch % 25 == 0:
                print(
                    f"\tg_loss {generator_loss}, d_loss {total_discriminator_loss/train_batch}")
            
        noise = torch.normal(mean=0, std=1,  size=(10, 100)).to(device)
        one_hot_class_labels = torch.arange(0, 10).to(device)
        y_fake = F.one_hot(one_hot_class_labels, num_classes=10).to(device)
        output = generator(noise, y_fake)

        grid_image = torchvision.utils.make_grid(output)
        torchvision.utils.save_image(
            grid_image, f"imgs/img_{current_epoch}.png")
        print(f"saved -> imgs/img_{current_epoch}.png")

if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    discriminator = Discriminator().to(device)
    generator = Generator(z=100).to(device)
    gan = train(
        discriminator=discriminator,
        generator=generator,
        device=device
    )
