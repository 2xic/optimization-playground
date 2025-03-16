import glob
from torch.utils.data import DataLoader, Dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
import os
import torch
import random
from typing import Union
from abc import ABC, abstractmethod


class SimpleTextEncoder:
    def __init__(self):
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.is_locked = False
        self.padding_index = self.add_word("<PADDING>")

    def add_word(self, word):
        if word not in self.vocab_idx:
            idx = len(self.vocab_idx)
            self.vocab_idx[word] = idx
            self.idx_vocab[idx] = word
        return self.vocab_idx[word]

    def decode_idx(self, word_idx: Union[torch.Tensor, int]):
        word_idx = word_idx.item() if isinstance(word_idx, torch.Tensor) else word_idx
        return self.idx_vocab[word_idx]


class TransformerDataset(ABC):
    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def sequence_size(self):
        pass

    @abstractmethod
    def decode(self, X):
        pass

    @property
    def padding_index(self):
        pass

    def __getitem__(self, index):
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)

    def iter(self, batch_size=4):
        return DataLoader(self, batch_size=batch_size)

    def sample(self, n):
        return list(
            map(
                lambda idx: [torch.tensor(self.X[idx]), torch.tensor(self.y[idx])],
                random.sample(range(len(self.X)), k=n),
            )
        )


class XorDataset(TransformerDataset):
    def __init__(self):
        self._X = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        self._y = torch.tensor(
            [
                [self.padding_index, self.padding_index, 0],
                [self.padding_index, self.padding_index, 1],
                [self.padding_index, self.padding_index, 1],
                [self.padding_index, self.padding_index, 0],
            ]
        )
        self._vocab_size = 3

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def sequence_size(self):
        return 3

    @property
    def padding_index(self):
        return 2

    def decode(self, X):
        if isinstance(X, torch.Tensor):
            return str(X.item())
        return str(X)


class TransformerTextDataset(TransformerDataset, Dataset):
    def __init__(self, X, y, encoder, sequence_size):
        super().__init__()
        self._X = X
        self._y = y
        self.encoder: SimpleTextEncoder = encoder
        self._vocab_size = len(self.encoder.idx_vocab)
        self._sequence_size = sequence_size

    @property
    def sequence_size(self):
        return self._sequence_size

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def vocab_size(self):
        return self._vocab_size

    @classmethod
    def from_folder(cls, folder, sequence_length):
        documents = []
        for i in glob.glob(folder):
            if os.path.isfile(i):
                with open(i, "rb") as file:
                    documents.append(splitter(file.read()))
        print(documents)
        assert len(documents) > 0
        return TransformerTextDataset._from_documents(documents, sequence_length)

    @classmethod
    def from_file(cls, txt, sequence_length):
        documents = []
        with open(txt, "rb") as file:
            documents.append(splitter(file.read()))
        assert len(documents) > 0
        return TransformerTextDataset._from_documents(documents, sequence_length)

    @classmethod
    def _from_documents(self, documents, sequence_length):
        documents = documents
        encoder = SimpleTextEncoder()
        X, y = [], []

        for doc in documents:
            space = sequence_length * 2
            for index, vocab in enumerate(doc[:-space]):
                encoder.add_word(vocab)
                n_tokens_forward = index + sequence_length
                X.append([encoder.add_word(v) for v in doc[index:n_tokens_forward]])
                y.append(
                    [
                        encoder.add_word(v)
                        for v in doc[
                            n_tokens_forward : n_tokens_forward + sequence_length + 1
                        ]
                    ]
                )
        return TransformerTextDataset(X, y, encoder, sequence_length)

    def decode(self, word_idx):
        return self.encoder.decode_idx(word_idx)

    @property
    def padding_index(self):
        return self.encoder.padding_index
