import torchvision
from dataset import generate_diff_image, box_size, apply_segmentation_callback
import torch
import os 
import glob

class DynamicDiffs:
    def __init__(self, path) -> None:
        self.x = torchvision.io.read_image(path)

class CustomCallbackImage:
    def __init__(self, x, y, callback) -> None:
        self.x = torchvision.io.read_image(x)
        self.y = torchvision.io.read_image(y)
        self.callback = callback

class DatasetEntries:
    def __init__(self) -> None:
        self.datasets = []

    def generate_dynamic_diffs(self, path):
        self.datasets.append(DynamicDiffs(path))
        return self
    
    def add_same_image_with_no_delta(self, path):
        path = os.path.join(path, "*")
        print("Exploring ", path)
        files = glob.glob(path)
        x = files[0]
        def same_image_delta(_x, _y):
            segmentation = torch.zeros((1, box_size, box_size))
            return segmentation, False
        
        for y in files[1:]:
            self.datasets.append(CustomCallbackImage(x, y, same_image_delta))
        return self

    
    def add_same_image_with_with_delta(self, path):
        path = os.path.join(path, "*")
        print("Exploring ", path)
        files = glob.glob(path)
        x = files[0]
        def same_image_delta(x, y):
            segmentation = torch.zeros((1, box_size, box_size))
            delta = torch.abs(x- y)

            for i in range(delta.shape[0]):
                segmentation[0, :, :] = segmentation[0, :, :] + delta[i, :, :]
            segmentation[segmentation > 1] = 1 
            return segmentation, False if torch.sum(torch.abs(x - y) > 200) > 1_00 else True
         
        for y in files[1:]:
            self.datasets.append(CustomCallbackImage(x, y, same_image_delta))
        return self

    def iterates(self):
        dataloaders = []
        print(len(self.datasets))
        for i in self.datasets:
            if isinstance(i, DynamicDiffs):
                dataloaders.append(generate_diff_image(i.x))
            elif isinstance(i, CustomCallbackImage):
                dataloaders.append(apply_segmentation_callback(i.x, i.y, i.callback))
            else:
                raise Exception("Unknown image type broski")
        return dataloaders
    
