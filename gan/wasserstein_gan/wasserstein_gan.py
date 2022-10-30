from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torchvision
from parameters import Parameters
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

class WassersteinGan(pl.LightningModule):
    def __init__(self, parameters, generator, discriminator):
        super().__init__()
        self.model_parameters = parameters
        self.generator = generator
        self.discriminator = discriminator

    def on_epoch_start(self):
        noise = torch.normal(
            mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1))
        image = torchvision.utils.make_grid(self.generator(noise))
        torchvision.utils.save_image(
            image, f"imgs/img_{self.current_epoch}.png")

    def training_step(self, batch, batch_idx, optimizer_idx):
        noise = torch.normal(
            mean=0.5, std=torch.arange(1., 101.)).reshape((1, -1))
        real, _ = batch
        # generator
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.model_parameters.ITERATIONS_CRITIC == 0:
                loss = - self.discriminator(self.generator(noise)).sum() / real.shape[0]
            else:
                return torch.tensor([0.0], requires_grad=True)
        # discriminator
        elif optimizer_idx == 1:
            generated = self.generator(noise).detach()
            loss = (self.discriminator(real) - self.discriminator(generated)).sum() / real.shape[0]
        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.model_parameters.LEARNING_RATE)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.model_parameters.LEARNING_RATE)
        
        return [opt_g, opt_d], []

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # generator
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)
        # discriminator
        if optimizer_idx == 1:
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.model_parameters.CLIPPING_PARAMETER)
            if (batch_idx + 1) % self.model_parameters.ITERATIONS_CRITIC == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                optimizer_closure()




if __name__ == '__main__':
    parameters = Parameters()
    train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=parameters.BATCH_SIZE)


    gan = WassersteinGan(
        parameters=parameters,
        discriminator=Discriminator(),
        generator=Generator(z=100)
    )
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=100,
    )

    trainer.fit(gan, train_loader)
