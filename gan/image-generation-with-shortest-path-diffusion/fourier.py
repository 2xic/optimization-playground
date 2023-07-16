import torch
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
import torch
from optimization_playground_shared.plot.Plot import Plot, Image

(train, _) = get_dataloader()

class Fourier:
    def __init__(self) -> None:
        pass

    def noise(self, tensor):
        pixel_noise = torch.randn_like(tensor)
        fourier_noise = torch.fft.fft2(pixel_noise, dim=(-2, -1), norm='ortho')
        return pixel_noise, fourier_noise
    

if __name__ == "__main__":
    (images, _) = next(iter(train))
    image = images[0]

    (pixel_noise, fourier_noise) = Fourier().noise(image)

    Plot().plot_image([
        Image(
            image=pixel_noise,
            title="Pixel noise"
        ), 
        Image(
            image=fourier_noise.real,
            title="Pixel noise in fourier real space"
        )
    ], 'noise.png')
