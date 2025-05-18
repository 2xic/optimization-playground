import os
import torch
import random
import glob
from torch.utils.data import DataLoader, Dataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
from abc import ABC, abstractmethod
from .dataset_tokenizer import SimpleTextEncoder
import pickle
import aiofiles
import asyncio
from tqdm import tqdm
from itertools import chain
from typing import List
from functools import lru_cache
from copy import deepcopy

MAX_WORKERS = 16
BATCH_SIZE = 2 << 12


async def read_file(filename):
    async with aiofiles.open(filename, mode="r") as file:
        contents = await file.read()
        return contents


async def gather_tasks(tasks):
    item = await asyncio.gather(*tasks)
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
    def decode_tokens(self, X: List[int]):
        pass

    @property
    def padding_index(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def iter(self, batch_size=4, workers=4):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=(workers > 0),
        )

    def sample(self, n):
        return list(
            map(
                lambda idx: [*self.__getitem__(idx)],
                random.sample(range(self.__len__()), k=min(n, self.__len__())),
            )
        )


class TransformerDataset(TransformerDatasetBase):
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
    def __init__(self, sequence_size=3):
        self._sequence_size = sequence_size
        self._vocab_size = 3
        self._padding_idx = 2
        self._X = self._create_padded_vector(
            torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        )
        self._y = self._create_padded_vector(
            torch.tensor(
                [
                    [self.padding_index, self.padding_index, 0],
                    [self.padding_index, self.padding_index, 1],
                    [self.padding_index, self.padding_index, 1],
                    [self.padding_index, self.padding_index, 0],
                ]
            )
        )
        assert self._X.shape[-1] == sequence_size
        assert self._y.shape[-1] == sequence_size

    def _create_padded_vector(self, input):
        tensor = (
            torch.zeros((input.shape[0], self.sequence_size))
            .fill_(self.padding_index)
            .long()
        )
        tensor[:, -input.shape[1] :] = input
        return tensor

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
        return self._sequence_size

    @property
    def padding_index(self):
        return self._padding_idx

    def decode_tokens(self, X: List[int]):
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
            pickle.dump(
                {
                    "X": self.X,
                    "y": self.y,
                },
                file,
            )
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
        dir_path = self.get_dir_path(self.name)
        os.makedirs(dir_path, exist_ok=True)
        if batch_id is None:
            return os.path.join(dir_path, "metadata.pkl")
        return os.path.join(dir_path, f"{batch_id}.pkl")

    @classmethod
    def get_dir_path(self, name):
        return os.path.join(
            os.path.dirname(__file__),
            "tensors",
            name,
        )

    @classmethod
    def does_exists(self, name):
        return os.path.isdir(PartialMemoryTensor.get_dir_path(name))


class TransformerTextDatasetLazy(Dataset, TransformerDatasetBase):
    def __init__(self, partial_memory_tensor_name, tokenizer):
        super().__init__()
        self.memory = PartialMemoryTensor(partial_memory_tensor_name)
        self.ids = self.memory.ids()
        self.rows = 0
        for id, size in self.ids:
            self.rows += size
        self.lookup = {}
        self.row_index = {}
        running_counter = 0
        for id, size in self.ids:
            for v in range(0, size):
                self.lookup[running_counter] = id
                self.row_index[running_counter] = v
                running_counter += 1
        prev = -1
        for i in sorted(self.lookup.keys()):
            assert (i - prev) == 1, f"{i} != {prev}"
            prev = i
        self.tokenizer = tokenizer
        self.max_size = float("inf")
        self._sequence_size = 256

    @classmethod
    def load(self, name, tokenizer):
        if PartialMemoryTensor.does_exists(name):
            return TransformerTextDatasetLazy(
                name,
                tokenizer,
            )
        return None

    @property
    def padding_index(self):
        return 1

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def sequence_size(self):
        return self._sequence_size

    def decode_tokens(self, X: List[int]):
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
        size = min(self.rows, self.max_size)
        return size

    def iter(self, batch_size=4, workers=2):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            persistent_workers=(workers > 0),
        )


class TransformerTextDataset(TransformerDataset, Dataset):
    def __init__(self, X, y, encoder, sequence_size):
        super().__init__()
        self._X = X
        self._y = y
        self.encoder: SimpleTextEncoder = encoder
        self._vocab_size = self.encoder.vocab_size
        self._sequence_size = sequence_size
        self._len = len(self.X)

    def __len__(self):
        return self._len

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
        return TransformerTextDataset.from_documents(documents, sequence_length)

    @classmethod
    def from_file(cls, tokenizer, txt, sequence_length):
        tokenizer.is_locked = True
        documents = []
        with open(txt, "rb") as file:
            documents.append(file.read())
        assert len(documents) > 0
        return TransformerTextDataset.from_documents(
            tokenizer, documents, sequence_length
        )

    @classmethod
    def from_iterator_single(
        cls, name, tokenizer, iterator, sequence_length
    ) -> TransformerTextDatasetLazy:
        tokenizer.is_locked = True
        batch_sizer = PartialMemoryTensor(name)

        def add_batch(doc):
            encoded = list(chain.from_iterable(tokenizer.encode(v) for v in doc))
            chunked_encoded = [
                encoded[i : i + sequence_length + 1]
                for i in range(0, len(encoded), sequence_length)
            ]
            for encoded in chunked_encoded:
                # TODO, maybe this is the issue?
                if not sequence_length < len(encoded):
                    encoded += [
                        1,
                    ] * (sequence_length - len(encoded) + 1)
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
            pickle.dump(
                {
                    "X": self.X,
                    "y": self.y,
                    "sequence_length": self._sequence_size,
                },
                file,
            )

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
    def from_documents(self, tokenizer: SimpleTextEncoder, documents, sequence_length):
        X, y = [], []
        tokenizer.is_locked = True

        def read_sequence(arr, index):
            return arr[index : index + sequence_length]

        for doc in documents:
            tokens = tokenizer.encode_document(doc)
            for index, _ in enumerate(tokens):
                next_tokens_index = index + 1
                current_tokens = read_sequence(tokens, index)
                next_tokens = read_sequence(tokens, next_tokens_index)
                if len(next_tokens) != sequence_length:
                    continue
                X.append(current_tokens)
                y.append(next_tokens)
        return TransformerTextDataset(X, y, tokenizer, sequence_length)

    def decode_tokens(self, tokens: List[int]):
        return self.encoder.decode(tokens)

    @property
    def padding_index(self):
        return self.encoder.padding_index


class BertTextDataset(TransformerDataset):
    def __init__(self, X, y, encoder, sequence_size):
        super().__init__()

    def from_iterator_single(self):
        raise Exception("?")

    @classmethod
    def from_documents(self, tokenizer: SimpleTextEncoder, documents, sequence_length):
        X, y = [], []
        tokenizer.is_locked = True

        def read_sequence(arr, index):
            return arr[index : index + sequence_length]

        for doc in documents:
            tokens = tokenizer.encode_document(doc)
            for index, _ in enumerate(tokens):
                current_tokens = read_sequence(tokens, index)
                copy_next_tokens = deepcopy(current_tokens)
                if len(current_tokens) != sequence_length:
                    continue
                n = random.sample(
                    list(range(len(current_tokens))), k=int(sequence_length * 0.15)
                )
                for i in n:
                    current_tokens[i] = tokenizer.masked_index
                assert None not in current_tokens
                assert None not in copy_next_tokens
                X.append(current_tokens)
                y.append(copy_next_tokens)
        return TransformerTextDataset(X, y, tokenizer, sequence_length)

    def decode_tokens(self, tokens: List[int]):
        return self.encoder.decode(tokens)

    @property
    def padding_index(self):
        return self.encoder.padding_index
