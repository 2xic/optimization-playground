from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch.utils.data import Dataset

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

    
def get_dataloader(batch_size=64, overfit=False, subset=None, shuffle=False, transforms=transforms.ToTensor(), sampler=None):
    (train, test, _) = get_dataloader_validation(
        batch_size=batch_size,
        overfit=overfit,
        subset=subset,
        shuffle=shuffle,
        transforms=transforms,
        sampler=sampler,
    )
    return (
        train,
        test
    )

def get_dataloader_validation(batch_size=64, overfit=False, subset=None, shuffle=False, transforms=transforms.ToTensor(), sampler=None):
    train_ds = MNIST("./", train=True, download=True, transform=transforms)
    test_ds = MNIST("./", train=False, download=True, transform=transforms)
    validation_ds = None

    if subset is not None:
        validation_end_offset = int(subset * 0.1)
        validation_ds = torch.utils.data.Subset(train_ds, list(range(0, validation_end_offset)))
        train_ds = torch.utils.data.Subset(train_ds, list(range(validation_end_offset, subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, train_ds))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, test_ds))
    validation_loader = None
    if validation_ds is not None:
        validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, validation_ds))

    if overfit:
        idx = train_ds.targets == 8
        train_ds.targets = train_ds.targets[idx]
        train_ds.data = train_ds.data[idx]

    return (
        train_loader,
        test_loader,
        validation_loader,
    )
