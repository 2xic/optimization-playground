from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
