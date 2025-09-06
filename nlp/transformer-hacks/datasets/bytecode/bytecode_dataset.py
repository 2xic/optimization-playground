import glob
from utils.dataset_tokenizer import (
    HuggingFaceTokenizerWrapper,
)
from utils.transformer_dataset import TransformerTextDataset, TransformerTextDatasetLazy
import asyncio
import torch
from datasets.dataset import BaseDataset
from collections import defaultdict
from tqdm import tqdm
import random


class BytecodeDataset(BaseDataset):
    def __init__(self, name="bytecode", kind="next_token"):
        # 1. Tokenize
        # 2. Create dataset of a given sequence size
        # 3. Train on it
        self._name = name
        self.base = "/home/brage/bigdrive/evm-contract-data/evm-opcodes-data/**/*.txt"
        self.kind = kind
        self._dataset = None
        self._tokenizer = None

    def create_tokenizer(self, name):
        (new_tokenizer, cached) = HuggingFaceTokenizerWrapper.load_cache(name)
        if not cached:
            new_tokenizer.train_tokenizer(self.get_file_content_tokenizer())
            new_tokenizer.save_cache()
        return (new_tokenizer, cached)

    def create_dataset(self, sequence_size=512, recreate=False):
        name = self.get_dataset_name(sequence_size)
        # We now have the dataset and can try to train the model on it.
        (new_tokenizer, cached) = self.create_tokenizer(self._name)
        text_dataset = TransformerTextDatasetLazy.load(name, new_tokenizer)
        # text_dataset = None
        if text_dataset is None or not cached or recreate:
            print("Starting dataset generation .. ", text_dataset)
            text_dataset = asyncio.run(
                TransformerTextDataset.from_iterator_single(
                    name,
                    new_tokenizer,
                    self.get_file_path(),
                    sequence_length=sequence_size,
                )
            )
        text_dataset._sequence_size = sequence_size
        # Just decrease the max size of the model, to force it to train
        # text_dataset.max_size = 1
        # return new_tokenizer, text_dataset
        self._dataset = text_dataset
        self._tokenizer = new_tokenizer
        return self

    @property
    def dataset(self):
        if self._dataset is None:
            raise Exception("Dataset is not yet created")
        return self._dataset

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise Exception("tokenizer is not yet created")
        return self._tokenizer

    def get_dataset_name(self, sequence_size):
        if self.kind != "next_token":
            return "masked_" + self._name + "_" + str(sequence_size)
        else:
            return self._name + "_" + str(sequence_size)

    def get_file_content_tokenizer(self):
        files = self.get_files()
        for index in range(0, len(files), 1_00):
            batch = files[index : index + 1_00]
            dataset = []
            for i in batch:
                with open(i, "r") as file:
                    dataset.append(file.read())
            yield dataset

    def get_file_content_tokenized(self, sequence_size):
        (tokenizer, cached) = self.create_tokenizer(self._name)
        assert cached
        files = self.get_files()
        for path in files:
            with open(path, "r") as file:
                tokens = tokenizer.encode(file.read())
                for i in range(0, len(tokens), sequence_size):
                    t = torch.tensor(tokens[i : i + sequence_size])
                    if t.shape[-1] != sequence_size:
                        continue
                    yield t

    def get_file_path(self):
        for path in self.get_files():
            yield path

    def get_files(self):
        return glob.glob(
            self.base,
            recursive=True,
        )

    def get_token_distribution(self, k):
        (tokenizer, cached) = self.create_tokenizer(self._name)
        assert cached

        distributions = defaultdict(int)
        files = self.get_files()
        files = random.sample(files, k=min(10_000, len(files)))
        for i in tqdm(files):
            with open(i, "r") as file:
                tokens = tokenizer.encode(file.read())
                for v in tokens:
                    distributions[v] += 1
        keys = sorted(distributions.keys(), key=lambda x: distributions[x])
        keys = [tokenizer.decode_idx(i) for i in keys]
        keys = list(filter(lambda x: len(x) > 4, keys))
        top_k = keys[-k:]
        lower_k = [key for key in keys[:k] if key not in top_k]
        return top_k, lower_k


class BytecodeDatasetTiny(BytecodeDataset):
    def __init__(self, name="bytecode_tiny", kind="next_token"):
        super().__init__(name, kind)

    def get_files(self):
        return sorted(
            glob.glob(
                self.base,
                recursive=True,
            )
        )[:1_000]


class BytecodeDatasetMedium(BytecodeDataset):
    def __init__(self, name="bytecode_medium", kind="next_token"):
        super().__init__(name, kind)

    def get_files(self):
        return sorted(
            glob.glob(
                self.base,
                recursive=True,
            )
        )[:25_000]


class BytecodeDatasetBig(BytecodeDataset):
    def __init__(self, name="bytecode_big", kind="next_token"):
        super().__init__(name, kind)


if __name__ == "__main__":
    #    BytecodeDataset().create_dataset(sequence_size=512, recreate=True)
    #    BytecodeDataset().create_dataset(sequence_size=32, recreate=True)
    BytecodeDatasetTiny().create_dataset(sequence_size=512)
