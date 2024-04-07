import torch
import torch.nn.functional as F
import torchvision
#from optimization_playground_shared.models.SimpleGenerator import SimpleGenerator
#from optimization_playground_shared.models.SimpleDiscriminator import SimpleDiscriminator
#from optimization_playground_shared.models.SimpleDeeperGenerator import SimpleDeeperGenerator
from optimization_playground_shared.models.SimpleLabelDiscriminator import SimpleLabelDiscriminator
from optimization_playground_shared.models.SimpleLabelGenerator import SimpleLabelGenerator
from gan_model import GanModel
import matplotlib.pyplot as plt
import parameters
import random
# from losses.GanBceLoss import GanBceLoss
# from losses.StandardLoss import StandardLoss
from losses.GanBceLabelLoss import GanBceLabelLoss

generator_loss = []
discriminator_loss = []

discriminator_real = []
discriminator_fake = []

def forward(gan: GanModel, train_loader):
    index_batches = 0
    batches = 0
    sum_g_loss = 0
    sum_d_loss = 0
    avg_discriminator_fake = 0
    avg_discriminator_real = 0

    loss = GanBceLabelLoss()
#    loss = StandardLoss()

    for index, (X, y) in enumerate(train_loader):
        batch_size = X.shape[0]
        X = X.to('cuda')
        X = torchvision.transforms.Resize(size=((parameters.IMG_SHAPE_X, parameters.IMG_SHAPE_Y)))(X)
        y = y.to('cuda').float()
        if parameters.APPLY_AUGMENTATIONS:
            transforms = torch.nn.Sequential(
                torchvision.transforms.RandomRotation(180),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur(9),
                ], p=0.3),
            )
            X = transforms(X)
        
        if parameters.NORMALIZE_INPUTS_WITH_TANH:
            min_val = torch.min(X)
            max_val = torch.max(X)
            X = 2 * ((X - min_val) / (max_val - min_val)) - 1
            assert torch.max(X) <= 1
            assert -1 <= torch.min(X) 
        
        batch_sum_d_loss = 0
        for _ in range(parameters.ITERATIONS_OF_DISCRIMINATOR_BATCH):
            gan.opt_d.zero_grad()

            noise = parameters.GET_NOISE_SAMPLE(batch_size)
            d_loss = loss.discriminator(
                gan.discriminator,
                X,
                y,
                gan.generator,
                noise,
            )     

            d_loss.backward()
            gan.opt_d.step()
            batch_sum_d_loss += d_loss.item()

            with torch.no_grad():
                avg_discriminator_fake += gan.discriminator(gan.generator(noise, y).detach())[0].mean().item()
                avg_discriminator_real += gan.discriminator(X)[0].mean().item()
                batches += 1
#                avg_discriminator_fake += gan.discriminator(gan.generator(noise).detach()).mean().item()
#                avg_discriminator_real += gan.discriminator(X).mean().item()

            if parameters.DEBUG and random.randint(0, 100) == 2:
                with torch.no_grad():
                    print("Real")
                    print(gan.discriminator(X)[:3, :])
                    print("Fake")
                    print(gan.discriminator(gan.generator(noise).detach())[:3])
                    print("Real vs fake img")
                    output_noise = gan.generator(noise)
                    print("Fake ")
                    print(output_noise[0][0][:1][:10])
                    real = X
                    print("Real ")
                    print(real[0][0][:1][:10])

        """
        Training of the generator
        """
        for _ in range(parameters.ITERATIONS_OF_GENERATOR_BATCH):
            noise = parameters.GET_NOISE_SAMPLE(batch_size)
            g_loss = loss.generator(
                gan.discriminator,
                X,
                y,
                gan.generator,
                noise,
            )
            gan.opt_g.zero_grad()
            g_loss.backward()
            gan.opt_g.step()

        if index % 250 == 0:
            print(f"Generator {g_loss.item()}")
            print(f"Discriminator {batch_sum_d_loss}")
            print("")
            sum_g_loss += g_loss.item()
            sum_d_loss += batch_sum_d_loss
            index_batches += 1
    discriminator_real.append(avg_discriminator_real / batches)
    discriminator_fake.append(avg_discriminator_fake / batches)
    generator_loss.append(sum_g_loss / index_batches)
    discriminator_loss.append(sum_d_loss / index_batches)

def plot_loss_discriminator(
    generator_loss,
    discriminator_loss,
    discriminator_fake,
    discriminator_real
):
    plt.title('Generator loss')
    plt.plot(generator_loss, label="Generator")
    plt.savefig('generator_loss.png')
    plt.clf()
    plt.title('Discriminator loss')
    plt.plot(discriminator_loss)
    plt.savefig('discriminator_loss.png')
    plt.clf()
    plt.title('Discriminator prediction (fake should go to 1)')
    plt.plot(discriminator_fake, label="Fake")
    plt.plot(discriminator_real, label="Real")
    plt.legend(loc="upper left")
    plt.savefig('discriminator_predictions.png')
    plt.clf()


if __name__ == '__main__':
    (train_loader, _) = parameters.DATALOADER()
    gan = GanModel(
        discriminator=SimpleLabelDiscriminator(input_shape=((1, parameters.IMG_SHAPE_X, parameters.IMG_SHAPE_Y))),
        generator=SimpleLabelGenerator(z=parameters.Z_SHAPE, input_shape=((1, parameters.IMG_SHAPE_X, parameters.IMG_SHAPE_Y)))
    )
    for epoch in range(parameters.EPOCHS):
        gan.current_epoch = epoch
        forward(gan, train_loader)
        print(f"Epoch done {epoch}")

        if epoch % 10 == 0:
            gan.plot()
    gan.plot_final()

    if parameters.PLOT_LOSS_AND_DISCRIMINATOR:
        plot_loss_discriminator(
             generator_loss,
            discriminator_loss,
            discriminator_fake,
            discriminator_real   
        )
