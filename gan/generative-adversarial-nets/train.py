# from https://arxiv.org/pdf/1406.2661.pdf

import torch
import torch.nn.functional as F
from dataset import train_loader
import torchvision
from optimization_playground_shared.models.SimpleDiscriminator import SimpleDiscriminator
from optimization_playground_shared.models.SimpleGenerator import SimpleGenerator
from plot import plot

device = torch.device('cuda')

class GanModel:
    def __init__(self, generator: SimpleGenerator, discriminator: SimpleDiscriminator):
        self.current_epoch = 0
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        lr = 0.0002
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def on_train_epoch_start(self):
        n = 9
        noise = self.get_noise(n=n)
        image = torchvision.utils.make_grid(self.generator(noise), nrow=n)
        torchvision.utils.save_image(image, f"imgs/img_{self.current_epoch}.png")

    def get_noise(self, n):
        noise = torch.normal(0, 1, size=(n, self.generator.z)).to(device)
        return noise

def forward(gan: GanModel, train_loader):
    k = 1
    gan.on_train_epoch_start()
    for index, (X, _) in enumerate(train_loader):
        X = X.to(device)
        sum_d_loss = 0
        loss = F.binary_cross_entropy
        batch_size = X.shape[0]
        for _ in range(k):

            real = loss(
                    gan.discriminator(X),
                    torch.ones(batch_size, 1).to(device)
            )
            generator_output = gan.generator(gan.get_noise(n=batch_size)).detach()
            fake = loss(
                    gan.discriminator(generator_output),
                    torch.zeros(batch_size, 1).to(device)
            )
            d_loss = (real + fake) / 2            

            gan.opt_d.zero_grad()
            d_loss.backward()
            gan.opt_d.step()
            sum_d_loss += d_loss.item()

        generated = gan.generator(gan.get_noise(n=batch_size))
        g_loss = loss(
            gan.discriminator(generated),
            torch.ones(batch_size, 1).to(device)
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
        discriminator=SimpleDiscriminator(),
        generator=SimpleGenerator(z=100)
    )
    for epoch in range(1_00):
        gan.current_epoch = epoch
        forward(gan, train_loader)
        print(f"Epoch done {epoch}")
        plot()
