import torchvision

augmentations = [
    torchvision.transforms.RandomHorizontalFlip(p=1),
    torchvision.transforms.RandomVerticalFlip(p=1),
    torchvision.transforms.RandomGrayscale(p=1)
]
