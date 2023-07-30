from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class _RawTensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            x = self.X[idx]
            return x
        else:
            x = self.X[idx]
            y = self.y[idx]
            return (x, y)

def get_dataloader(X, y, batch_size=32):
    loader = DataLoader(_RawTensorDataset(X, y), batch_size=batch_size)
    return loader
