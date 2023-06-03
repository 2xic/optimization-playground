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
    dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root="./", download=True, transform=dataset_transforms)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(
        root="./", download=True, train=False, transform=dataset_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    return dataloader, test_dataloader
