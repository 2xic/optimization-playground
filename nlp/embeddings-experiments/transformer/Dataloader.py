from torch.utils.data import Dataset
from .CreateDataset import CreateDataset
import torch
import torch.functional as F

class TransformerDataset(Dataset):
    def __init__(self, vocab, dataset):
        self.X = CreateDataset(vocab).process(dataset)
        self.vocab = vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.zeros((32))
        document = self.X[idx]
        for index, i in enumerate(document):
            x[index] = i
        for i in range(len(document), 32):
            x[i] = self.vocab.PADDING_IDX
        return x
