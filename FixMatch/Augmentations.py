import torchvision
import random

class Augmentations:
    def __init__(self) -> None:
        pass

    def get_strong_augmentation(self, X):
        return torchvision.transforms.RandomErasing(p=1,
                                                    ratio=(1, 1),
                                                    scale=(0.03, 0.03))(X)

    def get_weak_augmentation(self, X):
        if random.randint(0, 1):
            return torchvision.transforms.RandomHorizontalFlip(p=0.5)(X)
        else:
            return torchvision.transforms.RandomVerticalFlip(p=0.5)(X)
