import torch
import torchvision
from optimization_playground_shared.models.SimpleConditionalDiscriminator import SimpleConditionalDiscriminator
from optimization_playground_shared.models.SimpleConditionalGenerator import SimpleConditionalGenerator
from optimization_playground_shared.dataloaders.OrderedMnistData import get_dataloader
import torch.nn.functional as F
from optimization_playground_shared.parameters_search.ParameterSearchWithFeedback import ParameterSearchWithFeedback, StateFile, Parameter
from optimization_playground_shared.process_pools.ProcessPool import ProcessPool
from dataclasses import dataclass

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
N_CLASSES = 10


def forward_loss(generator, discriminator, x_real, y_real, is_generator_loss, z_size):
    x_real = x_real.to(device).float()
    y_real = y_real.to(device).long()

    noise = torch.normal(mean=0, std=1,  size=(x_real.shape[0], z_size), device=device)
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

def train(lock, parameter_search: ParameterSearchWithFeedback):
    parameters = parameter_search.parameters()

    discriminator = SimpleConditionalDiscriminator().to(device)
    generator = SimpleConditionalGenerator(z=parameters["z_dimension_size"]).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=parameters["lr_generator"])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=parameters["lr_discriminator"])
    train_dataloader = get_dataloader(
        batch_size=parameters["batch_size"],
        max_train_size=10_000,
    )
    ITERATIONS_CRITIC = parameters["n_iterator_critic"]
    z_size = parameters["z_dimension_size"]

    for current_epoch in range(50):
        total_loss_discriminator = torch.tensor(0, device=device).float()
        total_loss_generator = torch.tensor(0, device=device).float()

        for _, (real, classes) in enumerate(train_dataloader):
            """"
            Training the discriminator
            """
            for _ in range(ITERATIONS_CRITIC):
                loss = forward_loss(generator, discriminator, real, classes, is_generator_loss=False, z_size=z_size)
                discriminator.zero_grad()
                (loss).backward()
                opt_d.step()
                total_loss_discriminator += loss.item()

            """
            Training the generator
            """
            loss_generator = forward_loss(generator, discriminator, real, classes, is_generator_loss=True, z_size=z_size)
            generator.zero_grad()
            loss_generator.backward()
            opt_g.step()
            total_loss_generator += loss_generator.item()

        lock.acquire()
        try:
            print(f"Loss discriminator {total_loss_discriminator.item()}, generator {total_loss_generator.item()}")
            print("")
        finally:
            lock.release()

        with torch.no_grad():
            noise = torch.normal(mean=0, std=1,  size=(10, z_size), device=device)
            classes = torch.arange(0, 10, device=device)
            y_real_one_hot =  F.one_hot(classes, num_classes=N_CLASSES).to(device).float() 

            generator_images = generator(noise, y_real_one_hot)
            grid_image = torchvision.utils.make_grid(generator_images)

            state_file = StateFile()
            state_file.add_float("generator_loss", total_loss_generator.item())
            state_file.add_float("discriminator_loss", total_loss_discriminator.item())
            state_file.add_image("generated_image", grid_image)
            parameter_search.store_state(
                current_epoch,
                state_file
            )

if __name__ == '__main__':
    parameter_search = ParameterSearchWithFeedback([
        Parameter("lr_generator", 1e-5, 1e-2),
        Parameter("lr_discriminator", 1e-5, 1e-2),
        Parameter("z_dimension_size", 10, 100, cast=int),
        Parameter("n_iterator_critic", 1, 10, cast=int),
        Parameter("batch_size", 8, 64, cast=int),
        # Categorical also ? Sigmoid vs Relu etc.
    ])
    search_space = parameter_search.all()
    print("Search space size: {}".format(len(search_space)))

    pool = ProcessPool(max_workers=30)
    for item in pool.execute(train, search_space):
        pass
