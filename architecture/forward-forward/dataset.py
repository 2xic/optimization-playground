import torch
from torch.autograd.variable import Variable
import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

def get_simple_dataset():
    positive = torch.tensor([
                [
                    0, 0, 0
                ],
                [
                    0, 1, 1
                ],
                [
                    1, 0, 1
                ],
                [
                    1, 1, 1
                ]
            ], dtype=torch.float, requires_grad=True).float()
    negative = torch.tensor([
                [
                    0, 0, 1
                ],
                [
                    0, 1, 0
                ],
                [
                    1, 0, 0
                ],
                [
                    1, 1, 0
                ]
            ], dtype=torch.float, requires_grad=True).float()
    return (
        Variable(positive),
        Variable(negative)
    )

def get_mnist_dataloader():
    dataset = torchvision.datasets.MNIST(root="./", download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    test_dataset = torchvision.datasets.MNIST(root="./", download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return dataloader, test_dataloader

