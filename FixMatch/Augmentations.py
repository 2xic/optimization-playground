import torchvision


class Augmentations:
    def __init__(self) -> None:
        pass

    def get_strong_augmentation(self, X):
        return torchvision.transforms.RandomErasing(p=1,
                                                    ratio=(1, 1),
                                                    scale=(0.01, 0.01))(X)

    def get_weak_augmentation(self, X):
        return torchvision.transforms.RandomVerticalFlip(p=0.5)(X)
