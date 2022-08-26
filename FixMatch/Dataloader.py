from collections import defaultdict
from typing import DefaultDict
from torch.utils.data import Dataset
import torchvision
import random

class Cifar10Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=(not test),
                                                    download=True)
        self.filtered_dataset = self.filter()

    def filter(self):
        dataset = self.dataset
        max_class_count = 10
        class_distribution = defaultdict(int)
        labeled = []
        unlabeled = []

        for (X, y) in dataset:
            if class_distribution[y] > max_class_count:
                if random.randint(0, 5) < 3:
                    unlabeled.append(X)
            else:
                class_distribution[y] += 1
                labeled.append((X, y))
        self.dataset = labeled
        self.unlabeled = unlabeled


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        unlabeled_x = self.unlabeled[idx]
        return torchvision.transforms.ToTensor()(X), y, torchvision.transforms.ToTensor()(unlabeled_x),

