from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())

idx = train_ds.targets == 8
train_ds.targets = train_ds.targets[idx]
train_ds.data = train_ds.data[idx]

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

