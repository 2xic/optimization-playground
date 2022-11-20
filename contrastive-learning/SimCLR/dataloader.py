from torch.utils.data import Dataset
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
        transforms_seq_3 = torch.nn.Sequential(
            torchvision.transforms.GaussianBlur(9),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        transforms_seq_4 = torch.nn.Sequential(
            transforms.CenterCrop(25),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            torchvision.transforms.Resize(size=(32, 32)),
        )
        transforms_seq_5 = torch.nn.Sequential(
            transforms.CenterCrop(10),
            torchvision.transforms.Resize(size=(32, 32)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        transforms_seq_6 = torch.nn.Sequential(
            torchvision.transforms.ColorJitter(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        transforms_seq_7 = torch.nn.Sequential(
            torchvision.transforms.Grayscale(3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

        self.transformations = [
            transforms_seq_1,
            transforms_seq_2,
            transforms_seq_3,
            transforms_seq_4,
            transforms_seq_5,
            transforms_seq_6,
            transforms_seq_7,
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, _ = self.dataset[idx]
        transformation_index = random.randint(0, len(self.transformations) - 1)
        transformation = self.transformations[transformation_index]
        transformation_2 = self.transformations[
            (transformation_index + 1) % len(self.transformations)
        ]

        t_x = transformation(X)
        t_y  = transformation_2(X)

        return t_x, t_y

