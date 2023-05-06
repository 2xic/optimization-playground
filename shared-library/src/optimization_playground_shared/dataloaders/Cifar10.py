from collections import defaultdict
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import torch

class Cifar10Dataloader(Dataset):
    def __init__(self, test=False, generated_labels=None, transforms=None):
        self.transforms = transforms
        self.dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=(not test),
                                                    download=True)
        # Cifar 10 has 6000 images per class, so we train on 1/6 of the dataset
        self.labels_per_class = 1_000
        self.generated_labels = generated_labels
        if not test:
            self.filtered_dataset = self.filter()
        self.validation = None
        self.test = test

    def filter(self):
        dataset = self.dataset
        class_distribution = defaultdict(int)
        self.labeled = []
        self.unlabeled = []
        val_index = int(len(dataset) * 0.75)

        for index in range(val_index):
            (X, y) = dataset[index]
            if class_distribution[y] < self.labels_per_class:
                class_distribution[y] += 1
                self.labeled.append((X, y))
            else:
                self.unlabeled.append((X, None))

        if self.generated_labels is None:
            self.dataset = self.labeled
        else:
            self.dataset = self.labeled + random.sample(self.unlabeled, 500)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        if self.transforms is None:    
            X = torchvision.transforms.ToTensor()(X)
        else:
            X = self.transforms(X)
        generated = False
        if self.generated_labels and y is None:
            y = self.generated_labels(X)[0]
            generated = True
        elif self.generated_labels:
            y_hot = torch.zeros(10)
            y_hot[y] = 1
            y = y_hot
        #print(y.shape)

        if self.generated_labels:
            return X, y, generated
        return X, y

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

def get_dataloader(batch_size=64, shuffle=False, num_workers=0, sampler=None, generated_labels=None, transforms=None):
    train_ds = Cifar10Dataloader(test=False, transforms=transforms, generated_labels=generated_labels)
    train_loader = DataLoader(train_ds, pin_memory=True, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, train_ds))

    test_ds = Cifar10Dataloader(test=True, transforms=transforms,)
    test_loader = DataLoader(test_ds, pin_memory=True, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, test_ds))

    return (
        train_loader,
        test_loader
    )
