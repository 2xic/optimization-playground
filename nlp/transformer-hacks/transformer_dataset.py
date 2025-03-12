import glob 
from torch.utils.data import DataLoader, Dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
from model import Config
import os 
import torch
import random
from typing import Union

class SimpleTextEncoder:
    def __init__(self):
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.is_locked = False

    def add_word(self, word):
        if word not in self.vocab_idx:
            idx = len(self.vocab_idx)
            self.vocab_idx[word] = idx
            self.idx_vocab[idx] = word
        return self.vocab_idx[word]
    
    def decode_idx(self, word_idx: Union[torch.Tensor, int]):
        word_idx = word_idx.item() if isinstance(word_idx, torch.Tensor) else word_idx
        return self.idx_vocab[word_idx]

class TextDataset(Dataset):
    def __init__(self, documents, config: Config):
        self.documents = documents
        self.encoder = SimpleTextEncoder()
        self.X, self.y = [], []

        for doc in documents:
            space = config.sequence_length * 2 + 1
            for index, vocab in enumerate(doc[:-space]):
                self.encoder.add_word(vocab)
                n_tokens_forward = index + config.sequence_length + 1
                self.X.append([
                    self.encoder.add_word(v)
                    for v in doc[index:n_tokens_forward]
                ])
                self.y.append([
                    self.encoder.add_word(v)
                    for v in doc[n_tokens_forward:n_tokens_forward + config.sequence_length + 1]
                ])

    @property
    def vocab_size(self):
        return len(self.encoder.idx_vocab)

    @classmethod
    def from_folder(cls, folder, config):
        documents = []
        for i in glob.glob(folder):
            if os.path.isfile(i):
                with open(i, "rb") as file:
                    documents.append(splitter(file.read()))
        print(documents)
        assert len(documents) > 0
        return TextDataset(documents, config)

    @classmethod
    def from_file(cls, txt, config):
        documents = []
        with open(txt, "rb") as file:
            documents.append(splitter(file.read()))
        assert len(documents) > 0
        return TextDataset(documents, config)
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)

    def iter(self, batch_size=4):
        return DataLoader(self, batch_size=batch_size)

    def sample(self, n):
        return list(map(lambda idx: [torch.tensor(self.X[idx]), torch.tensor(self.y[idx])], random.sample(range(len(self.X)), k=n)))
