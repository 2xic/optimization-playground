from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import torch

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Cifar10Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=(not test),
                                                          download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Cifar100Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR100(root='./data', train=(not test),
                                                          download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class SimClrCifar100Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR100(root='./data', train=(not test),
                                                          download=True, transform=transform)

        transforms_seq_1 = torch.nn.Sequential(
            torchvision.transforms.GaussianBlur(3),
            torchvision.transforms.ColorJitter(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        transforms_seq_2 = torch.nn.Sequential(
            torchvision.transforms.RandomRotation(180),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.transformations = [
            torchvision.transforms.RandomRotation(180),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.RandomRotation(270),
            torchvision.transforms.ColorJitter(),
            transforms_seq_1,
            transforms_seq_2,
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, _ = self.dataset[idx]
        
        transformation = self.transformations[random.randint(0, len(self.transformations) - 1)]

        return transformation(X), transformation(X)

