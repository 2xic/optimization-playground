import torch
import torchvision
from parameters import model_sample, qt_sample

def plot(X_0, model, device, sample_size, T, shape=((1, 28, 28)), epoch=0):
    """
    Goes from noise to image
    P(X_(t - 1) | X_t ) ->

    X_t = noise
    X_0 = Image
    """
    shape = (sample_size, ) + shape
    with torch.no_grad():
        noise = torch.randn(shape, device=device)
        for time in range(T - 1, 0, -1):
            time_tensor = torch.full((sample_size, 1), time, device=device)
            output = model(
                noise,
                time_tensor,
            )
            noise = model_sample(
                output,
                noise,
                time,
                shape
            )

        grid_image = torchvision.utils.make_grid(noise)
        torchvision.utils.save_image(
            grid_image,
            f"example_generated_raw_epoch_{epoch}.png"
        )

        noise -= torch.min(noise)
        noise /= torch.max(noise)
        noise = noise.cpu()

        grid_image = torchvision.utils.make_grid(noise)
        torchvision.utils.save_image(
            grid_image,
            f"example_generated_normalized_epoch_{epoch}.png"
        )

        noise_example = torch.rand_like(torch.zeros((sample_size, 1, 28, 28)), device=device)
        grid_image = torchvision.utils.make_grid(noise_example)

        torchvision.utils.save_image(
            grid_image,
            f"example_fake_noise.png"
        )

        epoch_t = torch.randint(1, T, size=(X_0.shape[0], 1), device=device)
        noise = torch.randn_like(X_0, device=device)
        noise_example = qt_sample(X_0, epoch_t, noise).to(device)
        grid_image = torchvision.utils.make_grid(noise_example[:sample_size])
        torchvision.utils.save_image(
            grid_image,
            f"example_real_noise.png"
        )

    print("saved :)")
