from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

train_ds = MNIST("./", train=True, download=True, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
))

# idx = train_ds.targets == 8
# train_ds.targets = train_ds.targets[idx]
# train_ds.data = train_ds.data[idx]

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

