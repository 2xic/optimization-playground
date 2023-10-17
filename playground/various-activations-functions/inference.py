from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import get_output_shape
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from optimization_playground_shared.plot.Plot import Plot, Figure
from collections import defaultdict

class CustomActivation(nn.Module):
    def __init__(self, activation) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, input: torch.Tensor):
        return input

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class BasicConvModel(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.training_decimals = 32
        input_shape=(1, 28, 28)
        num_classes=10
        self.conv1 = nn.Conv2d(input_shape[0], 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.output_shape = ((
            get_output_shape(input_shape,
            [self.conv1, self.pool, self.conv2, self.pool])
        ))
        self.out = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            self.pool,
            Flatten(),
            nn.Linear(16 * self.output_shape[0] * self.output_shape[1], 256),
            nn.Linear(256, 128),
            CustomActivation(activation),
            nn.Linear(128, 63),
            CustomActivation(activation),
            nn.Dropout(p=0.01),
            nn.Linear(63, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.out(x)
        return x
    
def train_and_test():
    train, test = get_dataloader(subset=10_000)

    plots_loss = defaultdict(list)
    plots_accuracy = defaultdict(list)
    for activation in [F.sigmoid, F.leaky_relu, F.tanh]:
        model = BasicConvModel(activation)
        optimizer = optim.Adam(model.parameters())
        iterator = TrainingLoop(model, optimizer)

        name = activation.__name__
        for epoch in range(1_00):
            print(f"Epoch: {epoch}")
            (loss, acc) = iterator.train(train)
            print(f"loss: {loss}, acc: {acc}")

            (loss, acc) = iterator.eval_with_loss(test)
            plots_loss[name].append(loss)
            plots_accuracy[name].append(acc)

    plot = Plot()
    plot.plot_figures(
        figures=[
            Figure(
                plots=plots_loss,
                title="Loss for testing",
                x_axes_text="Epochs",
                y_axes_text="Loss",
            ),
            Figure(
                plots=plots_accuracy,
                title="Accuracy on testing",
                x_axes_text="Epochs",
                y_axes_text="Accuracy (%)",
            )
        ],
        name='training.png'
    )

if __name__ == "__main__":
    train_and_test()
