from collections import defaultdict
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random

class Cifar10Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=(not test),
                                                    download=True)
        # cifar 10 has 6000 images per class, so we train on 1/6
        self.labels_per_class = 1_000
        if not test:
            self.filtered_dataset = self.filter()
        self.validation = None
        self.test = test

    def filter(self):
        dataset = self.dataset
        class_distribution = defaultdict(int)
        labeled = []
        val_index = int(len(dataset) * 0.75)

        for index in range(val_index):
            (X, y) = dataset[index]
            if class_distribution[y] < self.labels_per_class:
                class_distribution[y] += 1
                labeled.append((X, y))

        self.dataset = labeled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return torchvision.transforms.ToTensor()(X), y

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

def get_dataloader(batch_size=64, shuffle=False, sampler=None):
    train_ds = Cifar10Dataloader(test=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, train_ds))

    test_ds = Cifar10Dataloader(test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, test_ds))

    return (
        train_loader,
        test_loader
    )
