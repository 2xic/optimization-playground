import torch
import torchvision
import os
import parameters

class GanModel:
    def __init__(self, generator, discriminator):
        self.current_epoch = 0
        self.generator = generator.to('cuda')
        self.discriminator = discriminator.to('cuda')

        lr = parameters.LEARNING_RATE

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def plot(self):
        if parameters.PLOT_EVERY_EPOCH:
            image = self._predict()
            os.makedirs("imgs", exist_ok=True)
            torchvision.utils.save_image(image, f"imgs/img_{self.current_epoch}.png")

    def plot_final(self):
        print(f"imgs/{parameters.PLOT_FINAL_OUTPUT_NAME}.png")
        image = self._predict()
        os.makedirs("imgs", exist_ok=True)
        torchvision.utils.save_image(image, f"imgs/{parameters.PLOT_FINAL_OUTPUT_NAME}.png")

    def _predict(self):
        noise = parameters.GET_NOISE_SAMPLE(4)
        y = torch.arange(0, 4).reshape((-1, 1)).to('cuda').float()
        output = self.generator(noise, y)
        image = torchvision.utils.make_grid(output)
        return image
