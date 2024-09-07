"""
Idea from https://tommyc.xyz/posts/simple-diffusion
"""

import torch
from tqdm import tqdm
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from torchvision.utils import save_image
import torch.nn as nn
import math
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

T = 10

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
        self.out = nn.Sequential(
          nn.Linear(math.prod(input_shape) + 2, 512),
          nn.Dropout(p=0.01),
          nn.ReLU(),
          nn.Linear(512, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Linear(512, math.prod(input_shape)),
          nn.Tanh(),
        )

    def forward(self, x, label_time):
        x_label_time = self.label(label_time)
        combined = torch.concat((
            x,
            x_label_time
        ), dim=1)
        x = self.out(combined) 
        x = x.reshape((x.shape[0], ) + self.input_shape)
        return x

    def train(self, x, y):
        step_size = 1 / T
 
       
        step_index_raw = torch.randint(0, T - 1, size=(y.shape[0], 1), device=device)
        step_index = step_index_raw.view(-1, 1, 1, 1).expand_as(x)

        x_n = torch.randn(x.shape, device=device)
        x_n = (x  * step_index / T) + x_n * (1 - step_index / T)

        combined = torch.concat((
            torch.nn.functional.one_hot(y, num_classes=10).to(device=device), 
            step_index_raw / T
        ), dim=1)
        delta = self.forward(x_n.reshape((x.shape[0], -1)), combined)
        x_n += delta * step_size

        error = torch.nn.functional.mse_loss(
            x_n,
            # Image noise lowers over time and image gets more sharp
            (x  * (step_index + 1) / T) + (torch.randn(x.shape, device=device) * (step_index + 1) / T)
        )
        error.backward()
        return error

    def sample(self, y):
        steps = torch.linspace(0,1, T, device=device)
        step_size = 1 / T
        n_sample = y.shape[0]
        x_n = torch.randn((n_sample, )+ self.input_shape, device=device)

        for step in range(T):
            step = torch.zeros((y.shape[0], 1), device=device).fill_(steps[step])
            combined = torch.concat((
                torch.nn.functional.one_hot(y, num_classes=10), 
                step
            ), dim=1)
            delta = self.forward(x_n.reshape((n_sample, -1)), combined)
            x_n += delta * step_size
        # todo : should really try to trick a discriminator or something
        return x_n / torch.max(x_n)

def train():
    progress = tqdm(range(1_000), desc='Training flow')
    train, _ = get_dataloader(
        subset=128,
        batch_size=128
    )
    model = SimpleFlowModel().to(device)
    adam = optim.Adam(model.parameters(), lr=0.0002)
  
    for epoch in progress:
        sum_loss = 0
        for _, (X, y) in enumerate(train):
            X = X.to(device)
            y = y.to(device)
            adam.zero_grad()
            loss = model.train(X, y)
            adam.step()
            sum_loss += loss

        if epoch % 10 == 0:
            generated = model.sample(y)
            save_image(X[:4], 'truth.png')
            save_image(generated[:4], 'flow.png')
        progress.set_description(f'Loss {sum_loss.item()}')

if __name__ == "__main__":
    train()
