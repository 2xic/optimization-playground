from collections import defaultdict
from torch.utils.data import Dataset
import torchvision

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
        labeled = []
        val_index = int(len(dataset) * 0.75)

        for index in range(val_index):
            (X, y) = dataset[index]
            if class_distribution[y] < labels_per_class:
                class_distribution[y] += 1
                labeled.append((X, y))

        self.dataset = labeled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        return torchvision.transforms.ToTensor()(X), y
