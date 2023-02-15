from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

def get_dataloader(batch_size=64):
    train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
    test_ds = MNIST("./", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return (
        train_loader,
        test_loader
    )
