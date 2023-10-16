import torch
import torchvision
from optimization_playground_shared.models.SimpleConditionalDiscriminator import SimpleConditionalDiscriminator
from optimization_playground_shared.models.SimpleConditionalGenerator import SimpleConditionalGenerator
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.plot.Plot import Plot, Image
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import torch.nn.functional as F

metrics_tracker = Tracker("conditional_gan")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataloader, _ = get_dataloader(shuffle=True, batch_size=32)
ITERATIONS_CRITIC = 2
n_classes = 10


def epoch_scale(epochs):
    if epochs < 3:
        return 3
    if epochs < 5:
        return 2
    else:
        return 1

def forward_loss(generator, discriminator, x_real, y_real, is_generator_loss, epochs):
    x_real = x_real.to(device)
    y_real = y_real.to(device)
    y_real_one_hot =  F.one_hot(y_real, num_classes=10).to(device).float() 

    # loss on the real data
    (is_real_data_real, real_data_labels) = discriminator(x_real)
    real_data_loss = epoch_scale(epochs) * torch.nn.CrossEntropyLoss()(
        real_data_labels,
        y_real
    ) + torch.nn.CrossEntropyLoss()(
        is_real_data_real,
        torch.ones((is_real_data_real.shape[0]), device=device).long()
    )
    # loss on the fake data
    noise = torch.normal(mean=0, std=1,  size=(x_real.shape[0], 100), device=device)
    random_labels = torch.argmax(torch.rand(size=(y_real.shape[0], 10), device=device), dim=1)
    y_fake =  F.one_hot(random_labels, num_classes=10).to(device).float() 

    #target_zero_class = torch.zeros((noise.shape[0]), device=device).long()
    generator_images = generator(noise, y_fake)
    (is_real, labels) = discriminator(generator_images)
    fake_data_loss = epoch_scale(epochs) * torch.nn.CrossEntropyLoss()(
        labels,
        random_labels,
    ) +  torch.nn.CrossEntropyLoss()(
        is_real,
        torch.zeros((is_real.shape[0]), device=device).long()
    )
    loss = (real_data_loss + fake_data_loss)

    if is_generator_loss:
        is_real.detach()
        labels.detach()
        is_real_data_real.detach()
        real_data_labels.detach()
#        return - (loss )
        return -loss
    else:
        generator_images.detach()
        return loss

def train(discriminator, generator):
    opt_g = torch.optim.RMSprop(generator.parameters(), lr=0.0002)
    opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.0002)

    for current_epoch in range(205):
      #  sum_real_data_loss = torch.tensor(0, device=device).float()
      #  sum_fake_data_loss = torch.tensor(0, device=device).float()
        total_loss = torch.tensor(0, device=device).float()
        total_loss_generator = torch.tensor(0, device=device).float()

        for _, (real, classes) in enumerate(train_dataloader):
            """"
            Training the discriminator
            """
            for _ in range(5):
                loss = forward_loss(generator, discriminator, real, classes, is_generator_loss=False, epochs=current_epoch)
                discriminator.zero_grad()
                (loss).backward()
                opt_d.step()
                total_loss += loss.item()

            """
            Training the generator
            """
            loss_generator = forward_loss(generator, discriminator, real, classes, is_generator_loss=True, epochs=current_epoch)
            generator.zero_grad()
            loss_generator.backward()
            opt_g.step()
            total_loss_generator += loss_generator.item()

        print(f"Loss discriminator {total_loss.item()}, generator {total_loss_generator.item()}")
       # print(f"Loss generator real {sum_real_data_loss.item()}, fake {sum_fake_data_loss.item()}")
        print("")

        with torch.no_grad():
            noise = torch.normal(mean=0, std=1,  size=(10, 100), device=device)
            classes = torch.arange(0, 10, device=device)
            y_real_one_hot =  F.one_hot(classes, num_classes=10).to(device).float() 

            generator_images = generator(noise, y_real_one_hot)
            grid_image = torchvision.utils.make_grid(generator_images)
            inference = Plot().plot_image([
                Image(
                    image=grid_image.cpu().numpy(),
                    title='output'
                )
            ], f'inference.png')

            metrics_tracker.log(
                Metrics(
                    epoch=current_epoch,
                    loss={
                        "generative loss": total_loss_generator.item(),
                        "discriminator loss": total_loss.item(),
                    },
                    training_accuracy=None,
                    prediction=Prediction.image_prediction(
                        inference
                    )
                )
            )

if __name__ == '__main__':
    gan = train(
        discriminator=SimpleConditionalDiscriminator().to(device),
        generator=SimpleConditionalGenerator(z=100).to(device),
    )
