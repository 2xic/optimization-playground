from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Cifar10Dataloader(Dataset):
    def __init__(self, test=False, max_files=float('inf')):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=(not test),
                                                          download=True, transform=transform)
        self.max_files = max_files

    def __len__(self):
        return min(len(self.dataset), self.max_files)

    def __getitem__(self, idx):
        return self.dataset[idx]
