from torch.utils.data import Dataset
from .CreateDataset import CreateDataset
import torch
import torch.functional as F
import random

class TransformerDataset(Dataset):
    def __init__(self, vocab, dataset):
        self.X = CreateDataset(vocab).process(dataset)
        self.vocab = vocab
        self.SEQUENCE_SIZE = 4

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.zeros((self.SEQUENCE_SIZE)).fill_(self.vocab.PADDING_IDX)
        document = self.X[idx]
        masked = []
        mask_index = random.randint(0, len(document))
        for index, i in enumerate(document[:self.SEQUENCE_SIZE]):
            if mask_index == index:
                x[index] = self.vocab.MASK_IDX
                masked.append(i)
            else:
                x[index] = i
        
        y = torch.zeros((self.SEQUENCE_SIZE)).fill_(self.vocab.PADDING_IDX)
        for index, i in enumerate(document[:self.SEQUENCE_SIZE]):
            y[index] = i
        return x, y
