import glob
import os
from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np
import cv2

class CarDriveDataset(Dataset):
    def __init__(self, train=True, fraction_of_dataset=1, transformers=None) -> None:
        super().__init__()

        self.fraction_of_dataset = fraction_of_dataset
        self.X, self.y = self.load_dataset(train)
        self.transformers = transformers

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

        split_index = int(len(X) * 0.95)
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

        if self.transformers is not None:
            x = self.transformers(x)

        return x, torch.tensor([y]).float()

methods = ["optical_flow", "background_subtraction", "plain"]

class CarDriveDatasetWithDeltaFrame(CarDriveDataset):
    def __init__(self, train=True, fraction_of_dataset=1, method="optical_flow") -> None:
        super().__init__(train, fraction_of_dataset)
        self.method = method
        self.cache = True # Note that running it without a cache slows things down a lot
        self.train = train
        assert self.method in methods

    def __len__(self):
        return super().__len__() - 1

    def __getitem__(self, idx):
        x = self.load_cache(idx, self.method)

        if x is None:
            if self.method == "optical_flow":
                x = self.get_optical_flow(
                    cv2.imread(self.X[idx]),
                    cv2.imread(self.X[idx + 1])
                )
            elif self.method == "background_subtraction":
                x = self.get_background_subtraction(
                    cv2.imread(self.X[idx]),
                    cv2.imread(self.X[idx + 1])
                )
            elif self.method == "plain":
                x = cv2.imread(self.X[idx])
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x = np.asarray(x)
                x = torch.from_numpy(x).permute(2, 0, 1) 
                assert x.shape[0] == 3/ x.max()
            else:
                method = self.method
                raise Exception(f"Unknown method {method}")

        # delta of the speed, maybe it is so low number that the model gets confused ? 
        self.save_cache(idx, self.method, x)
        if torch.max(x) > 1:
            x = x / 255.0
        y = self.y[idx] - self.y[idx + 1]

        # make sure things are normalized
        assert torch.max(x) <= 1, torch.max(x)
        assert torch.min(x) <= 0

        return x.float(), torch.tensor([y]).float()

    def load_cache(self, idx, method):
        if method != "plain":
            cache_path = self.get_cache_path(idx, method)
            if os.path.isfile(cache_path):
                #return ##torchvision.io.read_image(cache_path)
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    # fallback to None - sometimes the data is bad for some reason :( 
                    pass
        return None

    def save_cache(self, idx, method, image):
        if method != "plain":
            if self.cache:
                cache_path = self.get_cache_path(idx, method)
                try:
                    #torchvision.utils.save_image(image, cache_path)
                    torch.save(image, cache_path)
                except Exception as e:
                    # fallback to None - sometimes the data is bad for some reason :( 
                    pass

    def get_cache_path(self, idx, method):
        os.makedirs(".cache", exist_ok=True)
        training = "training" if self.train else "test"
        return f".cache/{idx}_{method}_{training}.png"

    def get_optical_flow(self, a, b):
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        # https://docs.opencv.org/3.4/dc/d6b/group__video__track.html
        flow = cv2.calcOpticalFlowFarneback(
            a_gray, 
            b_gray, 
            None, 
            pyr_scale = 0.5, 
            levels = 8, 
            winsize = 11, 
            iterations = 16, 
            poly_n = 5, 
            poly_sigma = 1.1, 
            flags = 0
        )
        img = np.zeros_like(a)
        img[:,:, :2] = flow

        return torch.from_numpy(img).permute(2, 0, 1)

    def get_background_subtraction(self, a, b):
        background_subtraction = cv2.createBackgroundSubtractorMOG2()
        mask = background_subtraction.apply(a)
        mask = background_subtraction.apply(b)

        img = np.zeros_like(a)
        img[:,:,0] = mask
        img[:,:,1] = mask
        img[:,:,2] = mask

        return torch.from_numpy(img).permute(2, 0, 1)

if __name__ == "__main__":
    for i in methods:
        X, y = CarDriveDatasetWithDeltaFrame(
            method=i
        )[0]
        print((X, X.shape, y))
        torchvision.utils.save_image(X, f"dataset/example_{i}.png")
