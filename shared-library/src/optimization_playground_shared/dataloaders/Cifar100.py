from collections import defaultdict
from torch.utils.data import Dataset
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Cifar100Dataloader(Dataset):
    def __init__(self, test=False, remove_classes=[], get_classes=[]):
        self.dataset = torchvision.datasets.CIFAR100(root='./data',
                                                    train=(not test),
                                                    download=True)        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        remove_idx = {}
        get_idx = None
        if len(remove_classes):
            for i in remove_classes:
                remove_idx[self.dataset.class_to_idx[i]] = True
        if len(get_classes):
            get_idx = {}
            for i in get_classes:
                get_idx[self.dataset.class_to_idx[i]] = True

        self.X, self.Y = self._filter(remove_idx, get_idx)

    def _filter(self, remove_idx, get_idx):
        X = []
        Y = []
        for index in range(len(self.dataset)):
            (x, y) = self.dataset[index]
            if y in remove_idx:
                pass
            elif get_idx is None:
                X.append(x)
                Y.append(y)
            elif y in get_idx:
            #    print(y, get_idx)
                X.append(x)
                Y.append(y)
        return X, Y 

    def __getitem__(self, idx):
        return self.transforms(self.X[idx]), self.Y[idx]

    def __len__(self):
        return len(self.X)

def get_dataloader(batch_size=64, shuffle=False, remove_classes=[], get_classes=[]):
    train_ds = Cifar100Dataloader(test=False, remove_classes=remove_classes, get_classes=get_classes)
    train_loader = DataLoader(train_ds, pin_memory=True, batch_size=batch_size, shuffle=shuffle)

    test_ds = Cifar100Dataloader(test=True,remove_classes=remove_classes, get_classes=get_classes)
    test_loader = DataLoader(test_ds, pin_memory=True, batch_size=batch_size, shuffle=shuffle)

    return (
        train_loader,
        test_loader
    )
