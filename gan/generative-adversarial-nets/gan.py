# from https://arxiv.org/pdf/1406.2661.pdf

from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import train_loader
from pytorch_lightning import LightningModule, Trainer
import torchvision

class GanModel(pl.LightningModule):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def on_epoch_start(self):
        noise = torch.normal(mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1))
        image = torchvision.utils.make_grid(self.generator(noise))
        torchvision.utils.save_image(image, f"imgs/img_{self.current_epoch}.png")

    def training_step(self, batch, _batch_idx, optimizer_idx):
        noise = torch.normal(mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1))
        real, _ = batch

        if optimizer_idx == 0:
            loss = F.binary_cross_entropy(
                self.discriminator(self.generator(noise)),
                torch.ones(noise.shape[0], 1)
            )

        elif optimizer_idx == 1:
           real = F.binary_cross_entropy(
                self.discriminator(real),
                torch.ones(real.shape[0], 1)
           )
           fake = F.binary_cross_entropy(
                self.discriminator(self.generator(noise).detach()),
                torch.zeros(noise.shape[0], 1)
           )
           loss = (real + fake) / 2
        return loss

    def configure_optimizers(self):
        lr = 1e-4

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

if __name__ == '__main__':
    gan = GanModel(
        discriminator=Discriminator(),
        generator=Generator(z=100)
    )
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
    )

    trainer.fit(gan, train_loader)
