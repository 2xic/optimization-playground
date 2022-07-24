from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


"""
Will be extended by the SimCLR dataloader
"""
class Cifar10Dataloader(Dataset):
    def __init__(self, test=False):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=(not test),
                                                          download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
