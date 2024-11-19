import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import Net
import os
from copy import deepcopy
import time

from torch.utils.data import DataLoader
from dataloader import Cifar10Dataloader
from concurrent.futures import ThreadPoolExecutor

def random_direction(model):
    direction = []
    for param in model.parameters():
        direction.append(torch.randn_like(param))
    return direction

def perturb_model(model, base_params, direction, alpha, beta):
    for i, param in enumerate(model.parameters()):
        param.data = base_params[i] + alpha * direction[0][i] + beta * direction[1][i]

def get_loss(train_loader, device, model, loss_fn, point, loss_surface):
#    print(device)
    total_loss = 0.0
    model.to(device)
    for inputs, targets in (train_loader):
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, targets.to(device))
        total_loss += loss.item()
    return point, total_loss / len(train_loader)

CURRENT_ITERATIONS = 0

def run(model: Net):
    global CURRENT_ITERATIONS
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
    # Total size
    TOTAL_ITERATIONS = (alphas.size * betas.size) * 2
    start = time.time()

    def iteration_done(future):
        global CURRENT_ITERATIONS
        point, loss_norm = future.result()
        (i, j) = point
        loss_surface[i, j] = loss_norm

        if CURRENT_ITERATIONS % torch.cuda.device_count() == 0 and torch.device:
            print(CURRENT_ITERATIONS, TOTAL_ITERATIONS, CURRENT_ITERATIONS / float(TOTAL_ITERATIONS) * 100)
            runtime = (time.time() - start)
            # what is left
            if CURRENT_ITERATIONS > 0:
                delta = CURRENT_ITERATIONS / float(TOTAL_ITERATIONS)
                total_time = runtime / delta 
                
                print("Timer (sec): ", total_time - runtime)
                print("Timer (hours): ", (total_time - runtime) / 3600)
        CURRENT_ITERATIONS += 1

    train_loaders = {
        i: DataLoader(
            Cifar10Dataloader(),
            batch_size=512,
            shuffle=True, 
            num_workers=2,
        )
        for i in range(torch.cuda.device_count())
    }
    counter = 0
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count() * 8) as executor:
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                perturb_model(model, base_params, direction, alpha, beta)
                
                # Compute loss over a batch or dataset
                device_counter = counter % torch.cuda.device_count()
                device = torch.device(f'cuda:{(device_counter)}')
                future = executor.submit(get_loss, train_loaders[device_counter], device, deepcopy(model), loss_fn, (i, j), loss_surface)
                future.add_done_callback(iteration_done)
                counter += 1

        while executor._work_queue.qsize() > 0:
            print(f"{CURRENT_ITERATIONS / TOTAL_ITERATIONS * 100} % ")
            time.sleep(0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    alphas, betas = np.meshgrid(alphas, betas)
    ax.plot_surface(alphas, betas, loss_surface, cmap='viridis')
    ax.set_zlabel("Loss")
    plt.savefig(name)

if __name__ == "__main__":
    for i in [False, True]:
        model = Net(skip_connections=i)
        run(model)
