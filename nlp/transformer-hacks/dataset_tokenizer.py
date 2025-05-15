"""
When working on larger dataset, you might want to be able to preprocess the dataset for the tokenizer etc.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
import torch
from typing import Union, Dict, Any
from abc import ABC, abstractmethod
from optimization_playground_shared.nlp.SimpleVocab import splitter
import json
import os
from tqdm import tqdm
from tokenizers import Tokenizer as HfTokenizer
import os
from tokenizers import ByteLevelBPETokenizer


class Tokenizer(ABC):
    name: str

    @property
    @abstractmethod
    def encode(self, word):
        pass

    @property
    @abstractmethod
    def decode_idx(self, word_idx):
        pass

    def split(self, doc):
        return splitter(doc)

    def save_cache(self):
        tokenizer_path = Tokenizer.get_tokenizer_path(self.name)
        with open(tokenizer_path, "w") as file:
            json.dump(self.get_cache_fields(), file)

    @classmethod
    def load_cache(_cls, name) -> Dict[str, Any]:
        tokenizer_path = Tokenizer.get_tokenizer_path(name)
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "r") as file:
                return json.load(file)
        return None

    @classmethod
    def get_tokenizer_path(_cls, name):
        tokenizer_path = os.path.join(
            os.path.dirname(__file__), "tokenizer", name + ".json"
        )
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        return tokenizer_path

    @abstractmethod
    def get_cache_fields(self) -> Dict[str, any]:
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    def build_from_files(self, iterator) -> "Tokenizer":
        for i in tqdm(iterator, desc="Building up files"):
            with open(i, "r") as file:
                tokens = list(set(splitter(file.read())))
            for i in tokens:
                self.encode(i)
        return self


class SimpleTextEncoder(Tokenizer):
    def __init__(self, name):
        self.name = name
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.is_locked = False
        self.padding_index = self.encode_idx("<PADDING>")
        self.unknown_index = self.encode_idx("<UNKNOWN>")

    def get_cache_fields(self):
        return {
            "vocab_idx": self.vocab_idx,
            "idx_vocab": self.idx_vocab,
            "is_locked": self.is_locked,
            "padding_index": self.padding_index,
            "unknown_index": self.unknown_index,
        }

    def build_from_iterator(self, iterator):
        for content in tqdm(iterator):
            # with open(file, "r") as file:
            tokens = list(set(splitter(content)))
            for i in tokens:
                self.encode(i)
        return self

    @classmethod
    def load_cache(cls, name):
        cache_fields = Tokenizer.load_cache(name)
        encoder = SimpleTextEncoder(name)
        if cache_fields is not None:
            encoder.vocab_idx = cache_fields["vocab_idx"]
            encoder.idx_vocab = cache_fields["idx_vocab"]
            # TODO: THis shouldn't really be needed ?
            encoder.idx_vocab = dict(
                map(lambda kv: (int(kv[0]), kv[1]), encoder.idx_vocab.items())
            )
            encoder.is_locked = cache_fields["is_locked"]
            encoder.padding_index = cache_fields["padding_index"]
            encoder.unknown_index = cache_fields["unknown_index"]
            return encoder, True
        return encoder, False

    def encode_document(self, document):
        print(document)
        return self.encode(document)

    def encode(self, doc):
        return [self.encode_idx(i) for i in self.split(doc)]

    def encode_idx(self, word):
        if self.is_locked and word not in self.vocab_idx:
            return self.padding_index
        elif word not in self.vocab_idx:
            idx = len(self.vocab_idx)
            self.vocab_idx[word] = idx
            self.idx_vocab[idx] = word
            assert type(word) == str, f"Word == {word}, Idx == {idx}"
        return self.vocab_idx[word]

    def decode(self, tokens):
        decoded_tokens = []
        for tok in tokens:
            decoded_tokens.append(self.decode_idx(tok))
        return "".join(decoded_tokens)

    def decode_idx(self, word_idx: Union[torch.Tensor, int]):
        word_idx = word_idx.item() if isinstance(word_idx, torch.Tensor) else word_idx
        return self.idx_vocab.get(word_idx, self.idx_vocab[self.unknown_index])

    @property
    def vocab_size(self):
        return len(self.vocab_idx)


"""
Wrappers around existing tokenizers
"""


class HuggingFaceTokenizerWrapper(Tokenizer):
    def __init__(self, name, vocab_size=50265):
        # self.bpe = HfWordPiece() # Unigram() # BPE()
        # self.tokenizer = HfTokenizer(self.bpe)
        self.tokenizer = ByteLevelBPETokenizer()
        # self.tokenizer = SentencePieceBPETokenizer()
        self.tokenizer.add_special_tokens(["<PADDING>"])
        self.padding_index = self.tokenizer.encode("<PADDING>").ids[0]
        assert self.decode_idx(self.padding_index) == "<PADDING>"
        self.is_locked = False
        self.name = name
        self._preferred_vocab_size = vocab_size

    def train_tokenizer(self, documents):
        assert type(documents) != str
        # self.tokenizer.train_from_iterator(documents)#, trainer=BpeTrainer())
        self.tokenizer.train_from_iterator(
            documents,
            vocab_size=self._preferred_vocab_size,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ],
        )

    def encode(self, doc):
        return self.tokenizer.encode(doc).ids

    def encode_idx(self, doc):
        return self.tokenizer.encode(doc).ids[0]

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def decode_idx(self, idx):
        return self.tokenizer.decode([idx], skip_special_tokens=False)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_cache_fields(self):
        raise Exception("not implemented")

    def save_cache(self):
        tokenizer_path = Tokenizer.get_tokenizer_path(self.name)
        self.tokenizer.save(tokenizer_path)

    def encode_document(self, document):
        return self.tokenizer.encode(document).ids

    def split(self, document):
        raise Exception("This method should not be used by the hf")

    @classmethod
    def load_cache(cls, name):
        tokenizer_path = cls.get_tokenizer_path(name)
        print(tokenizer_path)
        hf = HuggingFaceTokenizerWrapper(name)
        if os.path.isfile(tokenizer_path):
            #            hf.tokenizer = HfTokenizer()
            hf.tokenizer = HfTokenizer.from_file(tokenizer_path)
            return hf, True
        return hf, False
