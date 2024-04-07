from .GanLoss import GanLoss
import torch


class StandardLoss(GanLoss):
    def __init__(self):
        self.loss = torch.nn.BCELoss()
        self.eps = 10e-4

    def generator(self, discriminator, real, generator, noise):
        g_loss = (
            torch.log(
                discriminator(generator(noise))
                + self.eps
            )
        ).mean()
        return g_loss

    def discriminator(self, discriminator, real, generator, noise):
        d_loss = (
            torch.log(
                discriminator(generator(noise).detach())
                + self.eps
            )
        ).mean() + (torch.log((1 -
                              discriminator(real)
                              + self.eps
                               )
                              )
                    ).mean()
        return d_loss
