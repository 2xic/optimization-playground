from generator import Generator
from discriminator import Discriminator
import torch
import torchvision
from parameters import Parameters
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def train(parameters: Parameters, generator, discriminator, dataloader):
    opt_g = torch.optim.RMSprop(
        generator.parameters(), lr=parameters.LEARNING_RATE)
    opt_d = torch.optim.RMSprop(
        discriminator.parameters(), lr=parameters.LEARNING_RATE)

    for current_epoch in range(25):
        for _batch in range(512):
            loss = torch.tensor(0).float()
            for index, (real, _) in enumerate(dataloader):
                noise = torch.normal(mean=0, std=1,  size=(real.shape[0], 100))
                real_data_loss = torch.nn.L1Loss()(
                    discriminator(real),
                    torch.ones(real.shape[0], 1)
                )
                fake_data_loss = torch.nn.L1Loss()(
                    discriminator(generator(noise).detach()),
                    torch.zeros(noise.shape[0], 1)
                )#.sum()/real.shape[0]
                loss = (real_data_loss + fake_data_loss) #/ 2
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), parameters.CLIPPING_PARAMETER)
                if index > parameters.ITERATIONS_CRITIC:
                    break
            discriminator.zero_grad()
            loss.backward()
            opt_d.step()

            loss_generator = - \
                discriminator(generator(noise)).sum() / real.shape[0]
            generator.zero_grad()
            loss_generator.backward()
            opt_g.step()

            if _batch % 128 == 0:
                print(
                    f"Loss discriminator {loss.item()}, generator {loss_generator.item()}")
        noise = torch.normal(mean=0, std=1,  size=(10, 100))
        output = generator(noise)
        grid_image = torchvision.utils.make_grid(output)
        torchvision.utils.save_image(
            grid_image, f"imgs/img_{current_epoch}.png")
        print(f"saved -> imgs/img_{current_epoch}.png")


if __name__ == '__main__':
    parameters = Parameters()
    train_ds = MNIST("./", train=True, download=True,
                     transform=transforms.ToTensor())
    train_loader = DataLoader(
        train_ds, batch_size=parameters.BATCH_SIZE, shuffle=True)

    gan = train(
        parameters=parameters,
        discriminator=Discriminator(),
        generator=Generator(z=100),
        dataloader=train_loader,
    )
