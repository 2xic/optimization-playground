from optimization_playground_shared.dataloaders.Cifar10 import get_dataloader as get_cifar_dataloader
from optimization_playground_shared.dataloaders.Mnist import get_dataloader as get_mnist_dataloader
from optimization_playground_shared.models.BasicConvModel import BasicConvModel
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
import torch.optim as optim
import torch.nn as nn
import torch

EPOCHS = 5

def train_cifar_10():
    train, _ = get_cifar_dataloader()
    model = BasicConvModel(input_shape=(3, 32, 32))
    optimizer = optim.Adam(model.parameters())
    iterator = TrainingLoop(model, optimizer)

    for _ in range(EPOCHS):
        (loss, acc) = iterator.train(train)
        print(f"\tLoss {loss}, Acc: {acc}")
    return model

def train_mnist():
    train, _ = get_mnist_dataloader()
    model = BasicConvModel(input_shape=(1, 28, 28))
    optimizer = optim.Adam(model.parameters())
    iterator = TrainingLoop(model, optimizer)

    for _ in range(EPOCHS):
        (loss, acc) = iterator.train(train)
        print(f"\tLoss {loss}, Acc: {acc}")
    return model

class MOE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gated_unit = nn.Sequential(
            nn.Linear(1, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        ) # gated based on channel lol
        # disable the training of these models
        self.mnist_model = train_mnist().requires_grad_(False)
        self.cifar_model = train_cifar_10().requires_grad_(False)

    def _get_cifar_image(self, x):
        if x.shape[1] != 3:
            z = torch.zeros((x.shape[0], 3, 32, 32))
            z[:, 0, :28, :28] = x[:, 0, :, :]
            z[:, 1, :28, :28] = x[:, 0, :, :]
            z[:, 2, :28, :28] = x[:, 0, :, :]
            return z
        else:
            return x
    
    def forward(self, x):
        weights = self.gated_unit(torch.tensor([[x.shape[1]]]).float())
        # -> Wait there is no way for us to weight based on this lol
        mnist_score = self.mnist_model(x[:, :1, :28, :28])
        cifar_model = self.cifar_model(self._get_cifar_image(x))

        mnist_score = weights * mnist_score # if weight is zero, this is zero, if weight is one this is one
        cifar_score = (1 - weights) * cifar_model # if weight is zero this is one, if weight is one this is zero

        return mnist_score + cifar_score, weights

def train_model():
    model = MOE()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_cifar, _ = get_cifar_dataloader()
    train_mnist, _ = get_mnist_dataloader()

    for epoch in range(1_0):
        iter_train_cifar = iter(train_cifar)
        iter_train_mnist = iter(train_mnist)
        while True:
            X_cifar, X_cifar_labels = next(iter_train_cifar, [None, None])
            X_mnist, X_mnist_labels = next(iter_train_mnist, [None, None])

            if X_cifar is None or X_mnist is None:
                break

            cifar_score, cifar_weights = model(X_cifar)
            mnist_score, mnist_weights = model(X_mnist)

            error = (torch.nn.CrossEntropyLoss()(
                cifar_score, X_cifar_labels
            ) + torch.nn.CrossEntropyLoss()(
                mnist_score, X_mnist_labels
            )) / 2
            accuracy = (torch.sum((
                X_cifar_labels == torch.argmax(cifar_score, dim=1)
            )) / X_cifar_labels.shape[0] + torch.sum((
                X_mnist_labels == torch.argmax(mnist_score, dim=1)
            )) / X_mnist_labels.shape[0]) / 2 * 100
            optimizer.zero_grad()
            #print((error, accuracy))
            print("Error: {error}, Accuracy: {acc}, Epoch: {epoch}".format(
                error=error.item(),
                acc=accuracy.item(),
                epoch=epoch
            ))
            print("\tcifar (should -> 0)" , cifar_weights.item())
            print("\tmnist (should -> 1)" , mnist_weights.item())
            error.backward()
            optimizer.step()
        

if __name__ == "__main__":
    train_model()

