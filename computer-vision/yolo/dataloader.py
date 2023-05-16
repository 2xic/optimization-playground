from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from coco2yolo import Coco2Yolo

import torchvision.transforms as T

class YoloDataset(Dataset):
    def __init__(self, 
            dataset: Coco2Yolo, 
            constants,
        ):
        self.constants = constants
        self.dataset = dataset
        self.image_keys = dataset.get_list()

    def __len__(self):
        return len(self.image_keys)
    
    def get_raw_index(self, idx):
        name = self.image_keys[idx]
        results = self.dataset.load(
            name
        )
        return results

    def __getitem__(self, idx):
        results = self.get_raw_index(idx)
        labels = results["yolo_bounding_boxes"]
        labels += [
            [0, 0, 0, 0]
        ] * (self.constants.BOUNDING_BOX_COUNT - len(labels))
     #   print(labels)
#        exit(0)
        return (
            results["image"],
            torch.tensor(labels),
          #  results
        )
    