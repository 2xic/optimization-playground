from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils.data import Dataset

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

class CorruptedMnistDataloader(Dataset):
    def __init__(self,  max_train_size, train, transforms):
        self.train_ds = MNIST("./", train=train, download=True, transform=transforms)
        self.len = min(max_train_size, len(self.train_ds))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        X, y = self.train_ds[idx]
        return (X, y)
    
def get_dataloader(batch_size=64, overfit=False, subset=None, shuffle=False, transforms=transforms.ToTensor(), sampler=None, max_train_size=float('inf')):
    train_ds = CorruptedMnistDataloader(train=True, transforms=transforms, max_train_size=max_train_size)
    test_ds = CorruptedMnistDataloader(train=False, transforms=transforms, max_train_size=max_train_size)

    if subset is not None:
        train_ds = torch.utils.data.Subset(train_ds, list(range(0, subset)))
        #test_ds = torch.utils.data.Subset(test_ds, list(range(0, subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, train_ds))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, test_ds))

    if overfit:
        idx = train_ds.targets == 8
        train_ds.targets = train_ds.targets[idx]
        train_ds.data = train_ds.data[idx]

    return (
        train_loader,
        test_loader
    )
