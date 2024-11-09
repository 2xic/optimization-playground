import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import Net
from tqdm import tqdm
import os

from torch.utils.data import DataLoader
from dataloader import Cifar10Dataloader

def random_direction(model):
    direction = []
    for param in model.parameters():
        direction.append(torch.randn_like(param))
    return direction

def perturb_model(model, base_params, direction, alpha, beta):
    for i, param in enumerate(model.parameters()):
        param.data = base_params[i] + alpha * direction[0][i] + beta * direction[1][i]

def run(model: Net, data_loader: DataLoader):
    name = "loss_skip_connections.png" if model.skip_connections else "loss_no_skip_connections.png"
    if os.path.isfile(name):
        print(f"{name} exists, skipping")
        return 
    alphas = np.linspace(-1, 1, 50)
    betas = np.linspace(-1, 1, 50)
    loss_surface = np.zeros((len(alphas), len(betas)))

    base_params = [param.clone() for param in model.parameters()]
    direction = [random_direction(model) for _ in range(2)]

    loss_fn = nn.NLLLoss()
    index = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturb_model(model, base_params, direction, alpha, beta)
            
            # Compute loss over a batch or dataset
            total_loss = 0.0
            for inputs, targets in tqdm(data_loader):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
            loss_surface[i, j] = total_loss / len(data_loader)
            index += 1
            print(index)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    alphas, betas = np.meshgrid(alphas, betas)
    ax.plot_surface(alphas, betas, loss_surface, cmap='viridis')
    ax.set_zlabel("Loss")
    plt.savefig(name)

if __name__ == "__main__":
    for i in [False, True]:
        model = Net(skip_connections=i)

        train_loader = DataLoader(Cifar10Dataloader(),
                                batch_size=64,
                                shuffle=True, 
                                num_workers=8)
        run(model, train_loader)
