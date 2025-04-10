import os
import torch
import random
import glob
from torch.utils.data import DataLoader, Dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
from abc import ABC, abstractmethod
from dataset_tokenizer import SimpleTextEncoder
import pickle
from multiprocessing import Process, Queue, Value
import time 
import aiofiles
import asyncio
import numpy as np
from tqdm import tqdm
from itertools import chain
from functools import lru_cache

MAX_WORKERS = 16
BATCH_SIZE = 2 << 12

async def read_file(filename):
    async with aiofiles.open(filename, mode='r') as file:
        contents = await file.read()
        return contents

async def gather_tasks(tasks):
    item = await asyncio.gather(
        *tasks
    )
    return item


class TransformerDatasetBase(ABC):
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

    @abstractmethod
    def __len__(self):
        pass

    def iter(self, batch_size=4):
        return DataLoader(self, batch_size=batch_size, num_workers=4)

    def sample(self, n):
        return list(
            map(
                lambda idx: [*self.__getitem__(idx)],
                random.sample(range(self.__len__()), k=n),
            )
        )

class TransformerDataset(ABC):
    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    def __getitem__(self, index):
        if isinstance(self.X[index], torch.Tensor):
            return self.X[index], self.y[index]
        return torch.tensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.X)


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


class PartialMemoryTensor:
    def __init__(self, name):
        self.counter = 0
        self.name = name
        self.sizes = 0
        self.sizes = []

        # WIP:
        self.X = []
        self.y = []
        self.size = 0

    def add(self, X, y):
        self.X.append(X)
        self.y.append(y)

        self.size += 1

        if self.size > BATCH_SIZE:
            self.flush()

    def flush(self):
        path = self.get_path(self.counter)
        self.sizes.append([self.counter, self.size])
        with open(path, "wb") as file:
            pickle.dump({
                "X": self.X,
                "y": self.y,
            }, file)
        path = self.get_path(None)
        with open(path, "wb") as file:
            pickle.dump(self.sizes, file)
        self.y = []
        self.X = []
        self.size = 0
        self.counter += 1

    def load(self, batch_id):
        path = self.get_path(batch_id)
        with open(path, "rb") as file:
            data = pickle.load(file)
            return data["X"], data["y"]

    def ids(self):
        with open(self.get_path(None), "rb") as file:
            data = pickle.load(file)
            return data

    def get_path(self, batch_id):
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.join(
            dir_path,
            "tensors",
        )
        name = self.name
        os.makedirs(dir_path, exist_ok=True)
        if batch_id is None:
            return os.path.join(dir_path, name + f".pkl")
        return os.path.join(dir_path, name + f"_{batch_id}.pkl")

class TransformerTextDatasetLazy(Dataset, TransformerDatasetBase):
    def __init__(self, partial_memory_tensor_name, tokenizer):
        super().__init__()
        self.memory = PartialMemoryTensor(partial_memory_tensor_name)
        self.ids = self.memory.ids()
        self.rows = 0
        for (id, size) in self.ids:
            self.rows += size
        self.lookup = {}
        self.row_index = {}
        running_counter = 0
        for (id, size) in self.ids:
            for v in range(0, size):
                self.lookup[running_counter] = id
                self.row_index[running_counter] = v
                running_counter += 1
        prev = -1
        for i in (sorted(self.lookup.keys())):
            assert (i - prev) == 1, f"{i} != {prev}"
            prev = i
        # TODO: 
        self.tokenizer = tokenizer  

    @property
    def padding_index(self):
        return 1

    @property
    def vocab_size(self):
        return len(self.tokenizer.vocab_idx)

    @property
    def sequence_size(self):
        return 256

    def decode(self, X):
        return self.tokenizer.decode(X)
    
    @lru_cache(2048)
    def load_file(self, id):
        return self.memory.load(id)

    def __getitem__(self, index):
        X, y = self.load_file(self.lookup[index])
        X, y = X[self.row_index[index]], y[self.row_index[index]]
        X, y = torch.tensor(X), torch.tensor(y)
        assert X.size().numel() == self.sequence_size, y.size()
        assert y.size().numel() == self.sequence_size, y.size()
        return X, y

    def __len__(self):
        return self.rows

    def iter(self, batch_size=4):
        return DataLoader(self, batch_size=batch_size, num_workers=1, shuffle=False)


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
        assert len(documents) > 0
        return TransformerTextDataset._from_documents(documents, sequence_length)

    @classmethod
    def from_file(cls, tokenizer, txt, sequence_length):
        tokenizer.is_locked = True
        documents = []
        with open(txt, "rb") as file:
            documents.append(file.read())
        assert len(documents) > 0
        return TransformerTextDataset._from_documents(tokenizer, documents, sequence_length)
    
    @classmethod
    def from_iterator_single(cls, name, tokenizer, iterator, sequence_length):
        tokenizer.is_locked = True
        batch_sizer = PartialMemoryTensor(name)
        def add_batch(doc):
            encoded = list(chain.from_iterable(tokenizer.encode(v) for v in doc))
            view = encoded
            chunked_encoded = [view[i:i+sequence_length + 1] for i in range(0, len(view), sequence_length)]
            for encoded in chunked_encoded:
                if not sequence_length < len(encoded):
                    encoded += [1, ] * (sequence_length - len(encoded) + 1)
                X = encoded[:-1]
                y = encoded[1:]

                assert len(X) == sequence_length, len(X)
                assert len(y) == sequence_length

                batch_sizer.add(
                    X,
                    y,
                )

        promises = []
        for path in tqdm(iterator):
            if len(promises) > 64:
                items = asyncio.run(gather_tasks(promises))
                for doc in items:
                    add_batch(doc)
                promises = []
            else:
                promises.append(read_file(path))     
        items = asyncio.run(gather_tasks(promises))
        
        for doc in items:
            add_batch(doc)

        batch_sizer.flush()
        return TransformerTextDatasetLazy(
            name,
            tokenizer,
        )
    
    @classmethod
    def get_path(self, name):
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.join(
            dir_path,
            "dataset",
        )
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, name + ".pkl")

    def save(self, name):
        path = self.get_path(name)
        with open(path, "wb") as file:
            pickle.dump({
                "X": self.X,
                "y": self.y,
                "sequence_length": self._sequence_size,
            }, file)
    
    @classmethod
    def load(self, name, tokenizer):
        path = self.get_path(name)
        if os.path.isfile(path):
            with open(path, "rb") as file:
                data = pickle.load(file)
                return TransformerTextDataset(
                    data["X"],
                    data["y"],
                    tokenizer,
                    data["sequence_length"],
                )
        return None

    @classmethod
    def _from_documents(self, tokenizer: SimpleTextEncoder, documents, sequence_length):
        X, y = [], []
        tokenizer.is_locked = True

        for doc in documents:
            doc = tokenizer.split(doc)
            space = sequence_length * 2
            for index, _ in enumerate(doc[:-space]):
                n_tokens_forward = index + sequence_length
                X.append([tokenizer.encode(v) for v in doc[index:n_tokens_forward]])
                y.append(
                    [
                        tokenizer.encode(v)
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
