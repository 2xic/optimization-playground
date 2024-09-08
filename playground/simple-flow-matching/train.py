"""
Want to try a naive implementation with ideas from:

https://tommyc.xyz/posts/simple-diffusion
https://transferlab.ai/pills/2024/flow-matching/
https://arxiv.org/pdf/2210.02747
"""

import torch
from tqdm import tqdm
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
import torch.nn as nn
import math
import torch.optim as optim
import random
from optimization_playground_shared.plot.Plot import Plot, Figure, Image
from optimization_playground_shared.models.SimpleLabelDiscriminator import SimpleLabelDiscriminator

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

T = 10
STEP_SIZe = 1 / T

class SimpleFlowModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super().__init__()
        self.input_shape = input_shape
        self.label = nn.Sequential(
          # label + n
          nn.Linear(10 + 1, 8),
          nn.ReLU(),
          nn.Linear(8, 2),
          nn.ReLU()
        )
        z_shape = 256
        self.out = nn.Sequential(
          nn.Linear(math.prod(input_shape) + 2, 512),
          nn.Dropout(p=0.01),
          nn.ReLU(),
          nn.Linear(512, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Linear(512, z_shape),
          nn.Tanh(),
        )
        self.mean = nn.Linear(z_shape, math.prod(input_shape))
        self.var = nn.Linear(z_shape, math.prod(input_shape))
 
    def sample_image(self, x, shape):
        x = self.out(x) 
        mean = self.mean(x)
        log_var = self.var(x)
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z.reshape(shape)

    def forward(self, x, label_time):
        x_label_time = self.label(label_time)
        combined = torch.concat((
            x,
            x_label_time
        ), dim=1)
        x = self.sample_image(combined,  shape=((x.shape[0], ) + self.input_shape))
        return x

    def make_noisy_image(self, image, noise, step):
        return (image  * step / T) + noise * (1 - step / T)
    
    def simple_forward(self, x, y, index):
        raw_index = index.reshape((-1, 1)) / T
        combined = self.combined(y, raw_index)
        delta = self.forward(x.reshape((x.shape[0], -1)), combined)
        return delta

    def train(self, x, y, discriminator):

        step_index_raw = torch.randint(0, T - 1, size=(y.shape[0], 1), device=device)
        step_index = step_index_raw.view(-1, 1, 1, 1).expand_as(x)

        original_noise = torch.randn(x.shape, device=device)
        x_n = self.make_noisy_image(x, original_noise, step_index)
        delta = self.simple_forward(x_n, y, step_index_raw)

        next_x_n = x_n + delta * STEP_SIZe
        # Compare
        next_time_step = (step_index + 1)
        next_time_step[next_time_step > T] = T
        target = self.make_noisy_image(x, original_noise, next_time_step)
        error = torch.nn.functional.mse_loss(
            next_x_n * 100,
            target * 100
        )
        # Use discriminator to verify if the model seems reasonable
        # if random.randint(0, 10) == 2:
        (_, labels) = discriminator(self.sample(y))
        error += torch.nn.functional.cross_entropy(
            labels,
            y
        )

        if random.randint(0, 100) == 10:
            plot = Plot()
            plot.plot_image(
                images=[
                    Image(
                        image=x_n[:10],
                        title="x_n"
                    ),
                    Image(
                        image=delta[:10],
                        title="Model"
                    ),
                    Image(
                        image=target[:10],
                        title="x_n + 1"
                    ),
                ],
                name='training.png'
            )

        return error
    
    def combined(self, y, time_step):
        combined = torch.concat((
            torch.nn.functional.one_hot(y, num_classes=10), 
            time_step
        ), dim=1)
        return combined

    def sample(self, y):
        n_sample = y.shape[0]
        x_n = torch.randn((n_sample, )+ self.input_shape, device=device)
 
        for index in range(1, T):
            step_index = torch.full((n_sample, 1), index, device=device).float()
            delta = self.simple_forward(x_n, y, step_index)
            x_n += delta  * STEP_SIZe

        return x_n / torch.max(x_n)

def train():
    progress = tqdm(range(2_000), desc='Training flow')
    train, _ = get_dataloader(
        subset=2048,
        batch_size=128,
        shuffle=True
    )
    discriminator = SimpleLabelDiscriminator().to(device)
    discriminator_optim = optim.Adam(discriminator.parameters())
    for epoch in tqdm(range(5)):
        for _, (X, y) in enumerate(train):
            X = X.to(device)
            y = y.to(device)
            (_, y_predicted) = discriminator(X)
            discriminator_optim.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                y_predicted,
                y
            )
            loss.backward()
            discriminator_optim.step()

    model = SimpleFlowModel().to(device)
    adam = optim.Adam(model.parameters()) 
    training_loss = []
    for epoch in progress:
        sum_loss = 0
        for _, (X, y) in enumerate(train):
            X = X.to(device)
            y = y.to(device)
            adam.zero_grad()
            loss = model.train(X, y, discriminator)
            loss.backward()
            adam.step()
            sum_loss += loss

        if epoch % 10 == 0:
            generated = model.sample(y)
            plot = Plot()
            plot.plot_image(
                images=[
                    Image(
                        image=generated[:10],
                        title="generated"
                    ),
                    Image(
                        image=X[:10],
                        title="truth"
                    ),
                ],
                name='inference.png'
            )            
        progress.set_description(f'Loss {sum_loss.item()}')
        training_loss.append(sum_loss.item())

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "Loss": training_loss,
                },
                title="Training loss",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots={
                    "Loss": training_loss[-100:],
                },
                title="Training loss (n - 100)",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
        ],
        name='loss.png'
    )  

if __name__ == "__main__":
    train()
