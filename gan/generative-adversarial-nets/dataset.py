from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

