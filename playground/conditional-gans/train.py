import torch
import torchvision
from optimization_playground_shared.models.SimpleConditionalDiscriminator import SimpleConditionalDiscriminator
from optimization_playground_shared.models.SimpleConditionalGenerator import SimpleConditionalGenerator
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.metrics_tracker.producer import Tracker
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
from optimization_playground_shared.plot.Plot import Plot, Image
from optimization_playground_shared.metrics_tracker.metrics import Prediction
import random

metrics_tracker = Tracker("conditional_gan")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataloader, _ = get_dataloader(shuffle=True)
lr = 1e-3
ITERATIONS_CRITIC = 2
n_classes = 10

def train(discriminator, generator):
    opt_g = torch.optim.RMSprop(
        generator.parameters(), lr=lr)
    opt_d = torch.optim.RMSprop(
        discriminator.parameters(), lr=lr)
    slow_start_until_epoch = 10
    for current_epoch in range(205):
        sum_real_data_loss = torch.tensor(0, device=device).float()
        sum_fake_data_loss = torch.tensor(0, device=device).float()
        loss = torch.tensor(0, device=device).float()
        for index, (real, classes) in enumerate(train_dataloader):
            real = real.to(device).float()
            classes = classes.to(device)
            noise = torch.normal(mean=0, std=1,  size=(real.shape[0], 100), device=device)
            (is_real, labels) = discriminator(real)
            real_data_loss = torch.nn.CrossEntropyLoss()(
                labels,
                classes
            )
            real_data_loss += torch.nn.L1Loss()(
                is_real,
                torch.ones(is_real.shape[0], device=device)
            )
            # we want the generator to try everything :D 
            #noise = torch.normal(mean=0, std=1,  size=(classes.shape[0], 100), device=device)
            target_zero_class = torch.zeros((noise.shape[0]), device=device).long()
            (is_real, labels) = discriminator(generator(noise, (classes / 10).reshape((-1, 1)).float()).detach())
            fake_data_loss = torch.nn.CrossEntropyLoss()(
                labels,
                target_zero_class,
            )
            fake_data_loss += torch.nn.L1Loss()(
                is_real,
                torch.zeros(is_real.shape[0], device=device)
            )
            sum_real_data_loss += real_data_loss
            sum_fake_data_loss += fake_data_loss
            loss += (real_data_loss + 0.5 * fake_data_loss) / 2
            if index > ITERATIONS_CRITIC:
                break
        discriminator.zero_grad()
        (loss ).backward()
        opt_d.step()

        #loss_generator = - discriminator(generator(noise)).sum() / real.shape[0]
        loss_generator = torch.tensor(0, device=device).float()
        n_generator =  1 if current_epoch < slow_start_until_epoch else 1
        for _ in range(n_generator):
            classes = torch.arange(0, 10, device=device).reshape((-1, 1)).float()
            noise = torch.normal(mean=0, std=1,  size=(classes.shape[0], 100), device=device)
            target_zero_class = torch.zeros((noise.shape[0]), device=device).long()
            generated = generator(noise, classes / 10)
            (is_real, labels) = discriminator(generated)
            loss_generator -= torch.nn.CrossEntropyLoss()(
                labels,
                classes.long().reshape((-1)),
            )
            loss_generator -= torch.nn.L1Loss()(
                is_real,
                torch.ones(is_real.shape[0], device=device)
            )
        generator.zero_grad()
        (loss_generator ).backward()
        opt_g.step()

        print(f"Loss discriminator {loss.item()}, generator {loss_generator.item()}")
        print(f"Loss generator real {sum_real_data_loss.item()}, fake {sum_fake_data_loss.item()}")
        print("")

        if current_epoch % 10:
            with torch.no_grad():
                noise = torch.normal(mean=0, std=1,  size=(10, 100), device=device)
                classes = torch.arange(0, 10, device=device).reshape((-1, 1)).float()
                output = generator(noise, classes)
                grid_image = torchvision.utils.make_grid(output)
            # print(grid_image.shape)
                inference = Plot().plot_image([
                    Image(
                        image=grid_image.cpu().numpy(),
                        title='output'
                    )
                ], f'inference.png')

                metrics_tracker.log(
                    Metrics(
                        epoch=current_epoch,
                        loss=None,
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
