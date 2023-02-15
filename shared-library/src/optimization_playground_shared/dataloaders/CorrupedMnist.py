from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random

class CorruptedMnistDataloader(Dataset):
    def __init__(self, corruption_rate, max_train_size):
        self.corruption_rate = corruption_rate
        self.train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
        assert 0 <= corruption_rate and corruption_rate <= 1
        self.len = max_train_size

    def __len__(self):
        return self.len # len(self.train_ds)

    def __getitem__(self, idx):
        X, y = self.train_ds[idx]
   #     print((idx, self.len, self.corruption_rate * self.len))
#        batch_corrup = int(y.shape[0]*self.corruption_rate)
        if idx < self.corruption_rate * self.len:
            old_y = y
            y =  torch.randint(0, 9, (1,)).item()#[0]
         #   print((y, old_y))
        return (X, y)

def get_dataloader(corruption_rate, max_train_size=1_000, batch_size=64):
    train_ds = CorruptedMnistDataloader(corruption_rate, max_train_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = MNIST("./", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return (
        train_loader,
        test_loader
    )
