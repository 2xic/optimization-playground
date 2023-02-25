from .GanLoss import GanLoss
import torch

class GanBceLabelLoss(GanLoss):
    def __init__(self):
        self.loss = torch.nn.BCELoss()

    def generator(self, discriminator, _, y, generator, noise):
        batch_size = noise.shape[0]
        generated = generator(noise, y)

        is_fake, labels = discriminator(generated)
        g_loss = self.loss(
            is_fake,
            torch.ones(batch_size).to('cuda')
        ) + torch.nn.CrossEntropyLoss()(
            labels,
            y.long()
        )
        return g_loss

    def discriminator(self, discriminator, real, y, generator, noise):
        batch_size = real.shape[0]

        real_x, real_y = discriminator(real)
        real = self.loss(
                real_x,
                torch.ones(batch_size).to('cuda')
        ) + torch.nn.CrossEntropyLoss()(
            real_y,
            y.long()
        )
        fake_image = generator(noise, y).detach()
        
        is_fake, is_fake_classes = discriminator(fake_image)
        fake = self.loss(
            is_fake,
            torch.zeros(batch_size).to('cuda')
        ) + torch.nn.CrossEntropyLoss()(
            is_fake_classes,
            torch.zeros(batch_size).to('cuda').long()
        )
        d_loss = (real + fake) / 2 
        return d_loss
