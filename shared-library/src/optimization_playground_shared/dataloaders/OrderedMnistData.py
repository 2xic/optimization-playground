from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

class OrderedMnistData(Dataset):
    def __init__(self, max_train_size):
        self.train_ds = MNIST("./", train=True, download=True, transform=transforms.ToTensor())
        self.len = max_train_size
        self.items = {}
        for (index, y) in enumerate(self.train_ds.targets):
            y = int(y)
            if y not in self.items:
                self.items[y] = [index]
            else:
                self.items[y].append(index)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        labels = idx % 9
        entries = self.items[labels]
        index = entries[idx % len(entries)]
        X = self.train_ds.data[index]
        return (X.reshape((1, 28, 28)), float(labels))

def get_dataloader(max_train_size=1_000, batch_size=64):
    train_ds = OrderedMnistData(max_train_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    return train_loader
