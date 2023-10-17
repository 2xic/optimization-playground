import torch
import torchvision
import optimization_playground_shared
from optimization_playground_shared.models.SimpleConditionalDiscriminator import SimpleConditionalDiscriminator
from optimization_playground_shared.models.SimpleConditionalGenerator import SimpleConditionalGenerator
from optimization_playground_shared.dataloaders.OrderedMnistData import get_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.plot.Plot import Plot, Image
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import torch.nn.functional as F

metrics_tracker = Tracker("conditional_gan").send_code_state([
    __file__,
    optimization_playground_shared.__file__
])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataloader = get_dataloader(
    batch_size=64,
    max_train_size=10_000,
)
ITERATIONS_CRITIC = 5
N_CLASSES = 10


def forward_loss(generator, discriminator, x_real, y_real, is_generator_loss, epochs):
    x_real = x_real.to(device).float()
    y_real = y_real.to(device).long()

    noise = torch.normal(mean=0, std=1,  size=(x_real.shape[0], 100), device=device)
    random_labels = torch.argmax(torch.rand(size=(y_real.shape[0], N_CLASSES), device=device), dim=1)
    y_fake =  F.one_hot(random_labels, num_classes=N_CLASSES).to(device).float() 
    x_fake_generated = generator(noise, y_fake)

    # loss on the real data
    (is_real_data_real, real_data_labels) = discriminator(x_real)
    # loss on the fake data
    (is_fake_data_real, fake_data_labels) = discriminator(x_fake_generated)

    # detach all the items
    if is_generator_loss:
        is_fake_data_real.detach()
        fake_data_labels.detach()
        is_real_data_real.detach()
        real_data_labels.detach()
    else:
        x_fake_generated.detach()

    l_is_real = torch.nn.L1Loss()(
        is_fake_data_real,
        torch.zeros((is_fake_data_real.shape[0], 1), device=device).float()
    ) +  torch.nn.L1Loss()(
        is_real_data_real,
        torch.ones((is_real_data_real.shape[0], 1), device=device).float()
    )
    l_class_error = torch.nn.CrossEntropyLoss()(
        fake_data_labels,
        random_labels,
    ) +  torch.nn.CrossEntropyLoss()(
        real_data_labels,
        y_real
    )

    loss = (l_is_real + l_class_error) 

    if is_generator_loss:
        return - loss
    else:
        return loss

def train(discriminator, generator):
    lr = 0.0002
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for current_epoch in range(205):
        total_loss = torch.tensor(0, device=device).float()
        total_loss_generator = torch.tensor(0, device=device).float()

        for _, (real, classes) in enumerate(train_dataloader):
            """"
            Training the discriminator
            """
            for _ in range(ITERATIONS_CRITIC):
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
        print("")

        with torch.no_grad():
            noise = torch.normal(mean=0, std=1,  size=(10, 100), device=device)
            classes = torch.arange(0, 10, device=device)
            y_real_one_hot =  F.one_hot(classes, num_classes=N_CLASSES).to(device).float() 

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
