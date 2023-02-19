# from https://arxiv.org/pdf/1406.2661.pdf

from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import train_loader
from pytorch_lightning import LightningModule, Trainer
import torchvision

class GanModel:
    def __init__(self, generator, discriminator):
        self.current_epoch = 0
        self.generator = generator.to('cuda')
        self.discriminator = discriminator.to('cuda')

        lr = 0.0002

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def on_train_epoch_start(self):
        noise = torch.normal(mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1)).to('cuda')
        image = torchvision.utils.make_grid(self.generator(noise))
        torchvision.utils.save_image(image, f"imgs/img_{self.current_epoch}.png")

def forward(gan, train_loader):
    k = 1
    gan.on_train_epoch_start()
    for index, (X, _) in enumerate(train_loader):
        X = X.to('cuda')
        sum_d_loss = 0
#        loss = F.binary_cross_entropy
        loss = F.binary_cross_entropy
        for _ in range(k):
            gan.opt_d.zero_grad()

            real = loss(
                    gan.discriminator(X),
                    torch.ones(X.shape[0], 1).to('cuda')
            )
            real.backward()

            noise = torch.normal(mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1)).to('cuda')
            generator_output = None
            with torch.no_grad():
                generator_output = gan.generator(noise).detach()
            fake = loss(
                    gan.discriminator(generator_output),
                    torch.zeros(noise.shape[0], 1).to('cuda')
            )
            fake.backward()

            d_loss = (real + fake) / 2            
            gan.opt_d.step()
            sum_d_loss += d_loss.item()

        generated = gan.generator(noise)
        g_loss = loss(
            gan.discriminator(generated),
            torch.ones(noise.shape[0], 1).to('cuda')
        )
        gan.opt_g.zero_grad()
        g_loss.backward()
        gan.opt_g.step()

        if index % 20 == 0:
            print(f"Generator {g_loss.item()}")
            print(f"Discriminator {d_loss.item()}")
            print("")

if __name__ == '__main__':
    gan = GanModel(
        discriminator=Discriminator(),
        generator=Generator(z=100)
    )
    for epoch in range(75):
        gan.current_epoch = epoch
        forward(gan, train_loader)
        print(f"Epoch done {epoch}")
