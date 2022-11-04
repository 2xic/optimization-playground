import torchvision
import torch

augmentations = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(p=0.33),
    torchvision.transforms.RandomVerticalFlip(p=0.33),
    torchvision.transforms.RandomGrayscale(p=0.33)
)
