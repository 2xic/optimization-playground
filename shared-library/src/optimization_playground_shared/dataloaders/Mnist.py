from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

def get_dataloader(batch_size=64, overfit=False, subset=None, shuffle=False, transforms=transforms.ToTensor(), sampler=None):
    train_ds = MNIST("./", train=True, download=True, transform=transforms)
    test_ds = MNIST("./", train=False, download=True, transform=transforms)

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
