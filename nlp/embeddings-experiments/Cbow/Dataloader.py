from torch.utils.data import Dataset
from .CreateDataset import CreateDataset
import torch
import torch.functional as F

class CbowDataset(Dataset):
    def __init__(self, vocab, dataset):
        self.X, self.y = CreateDataset(vocab).process(dataset)
        self.vocab = vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X_tensor = torch.zeros((1, 4))
        X_tensor[:] = self.vocab.PADDING_IDX
        for index, i in enumerate(X):
            X_tensor[0][index] = i

        y = torch.zeros((1))
        y[0] = self.y[idx]

        return X_tensor.long(), y.float()
