"""
The effect of decimals places during training and inference for simple MNIST playground
"""

from optimization_playground_shared.dataloaders.Mnist import get_dataloader
from optimization_playground_shared.models.BasicConvModel import get_output_shape
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F

class DecimalConverter(nn.Module):
    def __init__(self, decimals) -> None:
        super().__init__()
        self.decimals = decimals

    def forward(self, input: torch.Tensor):
        return input.round(decimals=self.decimals)

class BasicConvModel(nn.Module):
    def __init__(self, decimals):
        super().__init__()
        self.training_decimals = 32
        self.decimals = decimals
        input_shape=(1, 28, 28)
        num_classes=10
        self.conv1 = nn.Conv2d(input_shape[0], 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.output_shape = ((
            get_output_shape(input_shape,
            [self.conv1, self.pool, self.conv2, self.pool])
        ))
        self.fc1 = nn.Linear(16 * self.output_shape[0] * self.output_shape[1], 256)
        self.out = nn.Sequential(
            #DecimalConverter(self.decimals),
            nn.Linear(256, 128),
            #DecimalConverter(self.decimals),
            nn.Sigmoid(),
            nn.Linear(128, 63),
            #DecimalConverter(self.decimals),
            nn.Sigmoid(),
            nn.Dropout(p=0.01),
            nn.Linear(63, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x.round(decimals=self.decimals))))
        x = torch.flatten(x.round(decimals=self.decimals), 1)
        x = F.dropout(x, p=0.1)
        x = self.out(x.round(decimals=self.decimals))
        return x
    
def train_and_test_with_same_decimals():
    x_decimal_places = []
    y_accuracy = []
    train, test = get_dataloader(max_train_size=10_000)

    model = BasicConvModel(32)
    optimizer = optim.Adam(model.parameters())
    iterator = TrainingLoop(model, optimizer)

    for epoch in range(32):
        print(f"Epoch: {epoch}")
        (loss, acc) = iterator.train(train)
        print(f"loss: {loss}, acc: {acc}")

    # inference is okay, but it does not working training with this for some reason
    # oh because it is not differential https://discuss.pytorch.org/t/torch-round-gradient/28628/2
    # https://discuss.pytorch.org/t/torch-round-gradient/28628/4
    for num_decimal_places in [2, 4, 8, 32]:
        model.decimals = num_decimal_places
        model.eval()
        accuracy = iterator.eval(test)
        print(f"Got {accuracy}% accuracy on test")
        x_decimal_places.append(num_decimal_places)
        y_accuracy.append(accuracy.item())

    plt.clf()
    plt.plot(x_decimal_places, y_accuracy)
    plt.xlabel(f'Decimals places')
    plt.ylabel('Accuracy on test dataset %')
    plt.savefig(f'decimals_testing_accuracy.png')
    plt.clf()

if __name__ == "__main__":
    train_and_test_with_same_decimals()