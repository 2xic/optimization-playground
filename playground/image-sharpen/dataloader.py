from torch.utils.data import Dataset
from coco_image_loader import CocoImageLoader
import torchvision
import torch
from config import IMAGE_SIZE

class Dataset(Dataset):
    def __init__(self, 
            dataset: CocoImageLoader, 
            max_items=None
        ):
        self.dataset = dataset
        self.max_items = max_items
        self.size = len(self.dataset.files)

    def __len__(self):
        if self.max_items is not None:
            return min(self.size, self.max_items)
        return self.size
    
    def get_raw_index(self, idx):
        return self.dataset.load(self.dataset.files[idx])

    def __getitem__(self, idx):
        results = self.get_raw_index(idx)
        width, height = results["original_size"]
        scale = 2
        new_width = 0
        new_height = 0
        while new_height < IMAGE_SIZE:
            new_width = IMAGE_SIZE * scale
            new_height = int(new_width * (height / width))
            scale += 1
        image = results["image"]
        if image.shape[0] == 1:
            image = torch.concat([
                image,
                image,
                image
            ], dim=0)
        noise_image = torchvision.transforms.Compose([
            torchvision.transforms.Resize((new_height, new_width)),
            torchvision.transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            torchvision.transforms.Grayscale(num_output_channels=1)
        ])
        blurred_image = torchvision.transforms.Compose([
            torchvision.transforms.GaussianBlur(kernel_size=3),
        ])
        noise = noise_image(image)
        blurred = blurred_image(noise)
        #assert noise.shape[0] == 3, noise.shape
        #assert blurred.shape[0] == 3, blurred.shape
        #return blurred, noise
        return noise, blurred
