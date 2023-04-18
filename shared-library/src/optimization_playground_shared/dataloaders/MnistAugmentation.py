from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

def get_dataloader(batch_size=64, overfit=False, subset=None):
    train_ds = MNIST("./", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.GaussianBlur(9),
        ]),
        transforms.RandomApply([
            transforms.RandomRotation(180),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip(p=0.33),
        ], p=0.3),
        transforms.RandomErasing(
            p=0.3,
            ratio=(1, 1),
            scale=(0.03, 0.03)
        ),
        transforms.RandomApply([
            transforms.RandomVerticalFlip(p=0.33),
        ], p=0.3),
        transforms.Normalize(
            (0.1307,
             ), (0.3081,)
        )
    ]))

    test_ds = MNIST("./", train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ]))

    if subset is not None:
        train_ds = torch.utils.data.Subset(train_ds, list(range(0, subset)))
        #test_ds = torch.utils.data.Subset(test_ds, list(range(0, subset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    if overfit:
        idx = train_ds.targets == 8
        train_ds.targets = train_ds.targets[idx]
        train_ds.data = train_ds.data[idx]

    return (
        train_loader,
        test_loader
    )
