from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
from collections import defaultdict

class RestrictedMnist(Dataset):
    def __init__(self, entries_per_class):
        self.entries_per_class = entries_per_class
        self.train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
        self.entries_class = defaultdict(list)
        for (X, y) in self.train_ds:
            self.entries_class[y].append(X)

    def __len__(self):
        return self.entries_per_class * 10

    def __getitem__(self, idx):
        item_class = idx % 10
        n = idx // self.entries_per_class

        X = self.entries_class[item_class][n]
        return (X, item_class)

def get_dataloader(entries_per_class, batch_size=64):
    train_ds = RestrictedMnist(entries_per_class)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = MNIST("./", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return (
        train_loader,
        test_loader
    )
