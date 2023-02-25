from .GanLoss import GanLoss
import abc
import torch

class GanBceLoss(GanLoss):
    def __init__(self):
        self.loss = torch.nn.BCELoss()

    def generator(self, discriminator, _, __, generator, noise):
        batch_size = noise.shape[0]
        generated = generator(noise)
        g_loss = self.loss(
            discriminator(generated),
            torch.ones(batch_size, 1).to('cuda')
        )
        return g_loss

    def discriminator(self, discriminator, real, y, generator, noise):
        batch_size = real.shape[0]
        real = self.loss(
                discriminator(real),
                torch.ones(batch_size, 1).to('cuda')
        )
        fake = self.loss(
            discriminator(generator(noise).detach()),
            torch.zeros(batch_size, 1).to('cuda')
        )
        d_loss = (real + fake) / 2 
        return d_loss
