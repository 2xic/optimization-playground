import torchvision
from torch.utils.data import DataLoader
import os
import pickle
from typing import Optional, Callable, Any, Tuple
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

"""
Copy form the torchvision.datasets.CIFAR10
"""
class CIFAR10Resnet(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)

        data = []
        for i in range(self.data.shape[0]):
            img = (self.data[i])
            data.append(img)
#            data.append(Compose([Resize((224, 224))])(img))
        self.data = data


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(self.data[index])
        #return Compose([Resize((224 // 4, 224 // 4)), ToTensor()])(img), target
        return Compose([ToTensor()])(img), target

def _call_if_func(func, dataset):
    if func is not None:
        return func(dataset)
    return func

def get_dataloader(batch_size=64, shuffle=False, num_workers=0, sampler=None, transforms=None, collate_fn=None):
    download = True #not torchvision.datasets.CIFAR10(root=".")._check_integrity()

    train_ds = CIFAR10Resnet(root=".", download=download, train=True, transform=transforms)
    train_loader = DataLoader(train_ds, pin_memory=True, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, train_ds), collate_fn=collate_fn)

    test_ds = CIFAR10Resnet(root=".", download=download, train=False, transform=transforms,)
    test_loader = DataLoader(test_ds, pin_memory=True, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, sampler=_call_if_func(sampler, test_ds), collate_fn=collate_fn)
    
    return (
        train_loader,
        test_loader,
        train_ds,
        test_ds,
    )
