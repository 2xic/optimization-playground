import os
import torch
import random
import glob
from torch.utils.data import DataLoader, Dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
from abc import ABC, abstractmethod
from dataset_tokenizer import SimpleTextEncoder

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
        if isinstance(self.X[index], torch.Tensor):
            return self.X[index], self.y[index]
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)

    def iter(self, batch_size=4):
        return DataLoader(self, batch_size=batch_size)

    def sample(self, n):
        return list(
            map(
                lambda idx: [*self.__getitem__(idx)],
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
    def from_file(cls, tokenizer, txt, sequence_length):
        documents = []
        with open(txt, "rb") as file:
            documents.append(file.read())
        assert len(documents) > 0
        return TransformerTextDataset._from_documents(tokenizer, documents, sequence_length)

    @classmethod
    def _from_documents(self, tokenizer: SimpleTextEncoder, documents, sequence_length):
        X, y = [], []
        tokenizer.is_locked = True

        for doc in documents:
            doc = tokenizer.split(doc)
            space = sequence_length * 2
            for index, _ in enumerate(doc[:-space]):
                n_tokens_forward = index + sequence_length
                X.append([tokenizer.add_word(v) for v in doc[index:n_tokens_forward]])
                y.append(
                    [
                        tokenizer.add_word(v)
                        for v in doc[
                            n_tokens_forward : n_tokens_forward + sequence_length 
                        ]
                    ]
                )
        return TransformerTextDataset(X, y, tokenizer, sequence_length)

    def decode(self, word_idx):
        return self.encoder.decode_idx(word_idx)

    @property
    def padding_index(self):
        return self.encoder.padding_index
