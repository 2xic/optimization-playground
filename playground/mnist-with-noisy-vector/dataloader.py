from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch

import torchvision.transforms as T
transform = T.ToPILImage()

class MnistWitNoise(Dataset):
    def __init__(self, train):
        self.train_ds = MNIST("./", train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.train_ds)

    def __getitem__(self, idx):
        X, y = self.train_ds[idx]
        noise_vector = torch.rand(X.shape)
        combined = torch.concat((X, noise_vector), dim=2)
        return (combined, y)

def get_dataloader(batch_size=64):
    train_ds = MnistWitNoise(train=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    img = transform(train_ds[0][0])
    img.save('example.png')


    test_ds = MnistWitNoise(train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return (
        train_loader,
        test_loader
    )
