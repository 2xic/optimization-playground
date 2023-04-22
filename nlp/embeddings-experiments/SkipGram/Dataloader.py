from torch.utils.data import Dataset, DataLoader
from .CreateDataset import CreateDataset
import torch
import torch.functional as F

class SkipGramDataset(Dataset):
    def __init__(self, vocab, dataset):
        self.X, self.y = CreateDataset(vocab).process(dataset)
        self.vocab = vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.zeros((1))
        x[0] = self.X[idx]
      #  x[0] = self.X[idx]

        y = self.y[idx]
        y_tensor = torch.zeros(( self.vocab.size))
 #       y_tensor[0] = y
        for index, i in enumerate(y):
            y_tensor[i] = 1
#        y_tensor = torch.exp(y_tensor) / y_tensor.exp().sum()
        return x.long(), y_tensor.float()
