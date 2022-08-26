import torchvision

class Augmentations:
    def __init__(self) -> None:
        pass

    def get_strong_augmentation(self, X):
        # cutout
        return torchvision.transforms.RandomErasing(p=1)(X)

    def get_weak_augmentation(self, X):
        # flip
        return torchvision.transforms.RandomVerticalFlip(p=1)(X)

