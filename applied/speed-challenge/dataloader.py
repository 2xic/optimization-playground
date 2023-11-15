import glob
import os
from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np
import cv2

class CarDriveDataset(Dataset):
    def __init__(self, train=True, fraction_of_dataset=1) -> None:
        super().__init__()

        self.fraction_of_dataset = fraction_of_dataset
        self.X, self.y = self.load_dataset(train)

    def load_dataset(self, train):
        speed = None
        with open("dataset/train.txt", "r") as file:
            speed = file.read().split("\n")
        images = glob.glob("dataset/images/*png")
        lookup_files = {}
        for i in images:
            lookup_files[os.path.basename(i)] = i
        X = []
        y = []
        for i in range(len(images)):
            X.append(lookup_files[f"frame_{i + 1}.png"])
            y.append(float(speed[i]))

        split_index = int(len(X) * 0.8)
        if train:
            return X[:split_index], y[:split_index]
        else:
            return X[split_index:], y[split_index:]

    def __len__(self):
        return int(len(self.X) * self.fraction_of_dataset)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        x = torchvision.io.read_image(x)
        y = y

        return x, torch.tensor(y)

class CarDriveDatasetWithDeltaFrame(CarDriveDataset):
    def __init__(self, train=True, fraction_of_dataset=1) -> None:
        super().__init__(train, fraction_of_dataset)

    def __len__(self):
        return super().__len__() - 1

    def __getitem__(self, idx):
        # delta frame
        #x = torchvision.io.read_image(self.X[idx])
        #x -= torchvision.io.read_image(self.X[idx + 1])

        x = self.get_optical_flow(
            cv2.imread(self.X[idx]),
            cv2.imread(self.X[idx + 1])
        )

        y = self.y[idx]
        y -= self.y[idx + 1]

        return x.float(), torch.tensor(y).float()
    
    def get_optical_flow(self, a, b):
        # pip install opencv-python
        background_subtraction = cv2.createBackgroundSubtractorMOG2()
        mask = background_subtraction.apply(a)
        mask = background_subtraction.apply(b)

        img = np.zeros_like(a)
        img[:,:,0] = mask
        img[:,:,1] = mask
        img[:,:,2] = mask

        return torch.from_numpy(img).permute(2, 0, 1)

if __name__ == "__main__":
    X, y = CarDriveDatasetWithDeltaFrame()[0]
    print((X, X.shape, y))
    torchvision.utils.save_image(X, 'dataset/example.png')
