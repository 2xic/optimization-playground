import glob
import os
from torch.utils.data import Dataset
from torchvision import datasets
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch

root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'dataset'
)
good_files = glob.glob(os.path.join(root, "1/*.png"))
bad_files = glob.glob(os.path.join(root, "0/*.png"))
count = min(len(good_files), len(bad_files))
split = .75

training_count = int(count * split)
testining_count = count - int(count * split)
X = good_files[:training_count] + bad_files[:training_count]
y = [1, ]* training_count + [0,] * training_count

X_test = good_files[training_count:training_count+testining_count] + bad_files[training_count:training_count+testining_count]
y_test = [1, ]* testining_count + [0,] * testining_count

class DatasetLoader(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = read_image(self.X[idx])
        label = self.y[idx]
        return image.float(), torch.tensor(label).long()

train_dataloader = DataLoader(DatasetLoader(X, y), batch_size=64, shuffle=True)
test_dataloader = DataLoader(DatasetLoader(X_test, y_test), batch_size=64, shuffle=True)
