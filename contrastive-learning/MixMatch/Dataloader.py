from collections import defaultdict
from typing import DefaultDict
from torch.utils.data import Dataset
import torchvision
import random
import torch

class Cifar10Dataloader(Dataset):
    def __init__(self, test=False):
        self.dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=(not test),
                                                    download=True)  
        if not test:
            self.filtered_dataset = self.filter()
        self.validation = None
        self.test = test

    def filter(self):
        dataset = self.dataset
        # cifar 10 has 6000 images per class, so we train on 1/6
        labels_per_class = 1_000
        class_distribution = defaultdict(int)
        total_class_distribution = defaultdict(int)
        labeled = []
        unlabeled = []

        val_index = int(len(dataset) * 0.75)

        for index in range(val_index):
            (X, y) = dataset[index]
            if class_distribution[y] > labels_per_class:
                if random.randint(0, 5) < 3:
                    unlabeled.append(X)
            else:
                class_distribution[y] += 1
                labeled.append((X, y))
            total_class_distribution[y] += 1
      #  print(total_class_distribution)

        self.dataset = labeled
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        if self.test:
            return torchvision.transforms.ToTensor()(X), y,
        
        y_hot = torch.zeros(10)
        y_hot[y] = 1

        unlabeled_x = self.unlabeled[random.randint(0, len(self.unlabeled)-1)]
        return torchvision.transforms.ToTensor()(X), y_hot, torchvision.transforms.ToTensor()(unlabeled_x),
