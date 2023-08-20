"""
Core model layers
"""
import torch.nn as nn
from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from plot import plot_input
import numpy as np
import torch
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torchvision.transforms as T

class LayerCoordinates:
    def __init__(self, layer: nn.Linear) -> None:
        out_dim, in_dim = layer.weight.shape        
        """
        Based off the reference implementation
        https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb
        """
        in_dim_fold = int(in_dim)
        self.out_dim_fold = int(out_dim)
        in_dim_sqrt = int(np.sqrt(in_dim_fold))
        self.out_dim_sqrt = int(np.sqrt(self.out_dim_fold))
        x = np.linspace(1/(2*in_dim_sqrt), 1-1/(2*in_dim_sqrt), num=in_dim_sqrt)
        X, Y = np.meshgrid(x, x)
        self.in_coordinates = torch.tensor(np.transpose(np.array([X.reshape(-1,), Y.reshape(-1,)])), dtype=torch.float)
        self.out_coordinates = None # set by the model construction

    def _get_coordinates(self, is_final_layer):
        """
        Based off the reference implementation
        https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb
        """
        if is_final_layer:
            thetas = np.linspace(1/(2*self.out_dim_fold)*2*np.pi, (1-1/(2*self.out_dim_fold))*2*np.pi, num=self.out_dim_fold)
            return 0.5+torch.tensor(np.transpose(np.array([np.cos(thetas), np.sin(thetas)]))/4, dtype=torch.float)
        else:
            x = np.linspace(1/(2*self.out_dim_sqrt), 1-1/(2*self.out_dim_sqrt), num=self.out_dim_sqrt)
            X, Y = np.meshgrid(x, x)
            return torch.tensor(np.transpose(np.array([X.reshape(-1,), Y.reshape(-1,)])), dtype=torch.float)


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, output_dimension) -> None:
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.linear_layer = nn.Linear(input_dimension, output_dimension)
        self.coordinates = LayerCoordinates(self.linear_layer)

    def forward(self, X):
        return self.linear_layer(X)

class Model(nn.Module):
    def __init__(self, shape) -> None:
        super(Model, self).__init__()
        self.layer_modules = []
        for index, i in enumerate(shape[:-1]):
            is_last_layer = (index + 2) == len(shape)
            linear = LinearLayer(i, shape[index + 1])
            linear.coordinates.out_coordinates = linear.coordinates._get_coordinates(is_last_layer)
            self.layer_modules.append(linear)

        model_layers = []
        for index, i in enumerate(self.layer_modules):
            model_layers.append(i)
            if (index + 1) == len(self.layer_modules):
                model_layers.append(nn.LogSoftmax(dim=1))
            else:
                model_layers.append(nn.ReLU())

        self.model = nn.Sequential(*model_layers)
        print(self.model)

    def forward(self, X):
        return self.model(X)

def train():
    train, _ = get_dataloader(
        transforms=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))])
    )
    image = next(iter(train))[0][0].reshape((28, 28))

    model = Model([784, 100, 100, 10])
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    plot_input(model, image, "untrained")

    trainer = TrainingLoop(
        model,
        optimizer,
    )
    for _ in range(10):
        (loss, acc) = trainer.train(train)
        print(f"loss: {loss}, accuracy: {acc}")
    plot_input(model, image, "trained")

if __name__ == "__main__":
    train()
