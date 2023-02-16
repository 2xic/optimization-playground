import os
from torchvision.io import read_image
import json

from torch.utils.data import Dataset

class LatexDataloader(Dataset):
    def __init__(self):
        data = {}
        with open("dataset.json", "r") as file:
            data = json.loads(file.read())
        self.images, self.values = zip(*data.items())

    def __len__(self):
        return len(self.images) - 1

    def __getitem__(self, idx):
        image = read_image(self.images[idx])
        label = self.values[idx]
        return image, label
