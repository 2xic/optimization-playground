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
import collections
from collections import defaultdict
from tqdm import tqdm
import time 
from functools import lru_cache
from tokenizers import Tokenizer as HfTokenizer
from tokenizers.models import BPE
import os 

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
    def load_cache(_cls, name) -> Dict[str, Any] :
        tokenizer_path = Tokenizer.get_tokenizer_path(name)
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "r") as file:
                return json.load(file)
        return None
    
    @classmethod
    def get_tokenizer_path(_cls, name):
        tokenizer_path = os.path.join(
            os.path.dirname(__file__),
            "tokenizer",
            name + ".json"
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

    def build_from_files(self, iterator) -> 'Tokenizer':
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
        for file in tqdm(iterator):
            with open(file, "r") as file:
                tokens = list(set(splitter(file.read())))
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
            encoder.idx_vocab = dict(map(lambda kv: (int(kv[0]), kv[1]), encoder.idx_vocab.items()))
            encoder.is_locked = cache_fields["is_locked"]
            encoder.padding_index = cache_fields["padding_index"]
            encoder.unknown_index = cache_fields["unknown_index"]
            return encoder, True
        return encoder, False

    def encode_document(self, document):
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

class WordPieceBuilder:
    def __init__(self, name):
        self.char_vocab = collections.Counter()
        
        special_tokens = ["<UNKNOWN>", "<PADDING>"]
        self.final_vocab = {token: 1 for token in special_tokens}
        self.subwords = defaultdict(int)
        self.name = name

    def build_from_iterator(self, iterator) -> 'WordPieceBuilder':
        for i in tqdm(iterator, desc="Building up files"):
            with open(i, "r") as file:
                self.add_document(file.read())
        return self

    def add_document(self, document):
        assert type(document) == str
        for word in splitter(document):
            for char in word:
                self.char_vocab[char] += 1
            self.subwords[" ".join(list(word))] += 1
        return self
        
    def build(self, vocab_size) -> 'WordPiece':
        for char, count in self.char_vocab.items():
            self.final_vocab[char] = count

        path = self.get_cache_path(self.name)
        with open(path, "w") as file:
            json.dump({
                "final_vocab": self.final_vocab,
                "subwords": self.subwords,
            }, file)
        
        return WordPiece.train_wordpiece(self.name, self.final_vocab, self.subwords, vocab_size)
    
    def build_from_cache(self, vocab_size):
        path = self.get_cache_path(self.name)
        with open(path, "r") as file:
            obj = json.load(file)
        
        return WordPiece.train_wordpiece(self.name, obj["final_vocab"], obj["subwords"], vocab_size)

    def get_cache_path(self, name):
        tokenizer_path = os.path.join(
            os.path.dirname(__file__),
            "wordpiece",
            name + ".json"
        )
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        return tokenizer_path


class RegularPrinter:
    def __init__(self, interval_seconds=60):
        self.last_print = time.time()
        self.interval_seconds = interval_seconds

    def print(self, *x):
        if self.interval_seconds < time.time() - self.last_print:
            print(*x)

class WordPiece(Tokenizer):
    def __init__(self, name, final_vocab, subwords):
        self.name = name
        self.subwords = subwords

        # Build the index.
        self.idx_vocab = {}
        self.vocab_idx = {}
        for index, vocab in enumerate(final_vocab.keys()):
            self.idx_vocab[index] = vocab
            self.vocab_idx[vocab] = index

    def get_cache_fields(self):
        return {
            "vocab_idx": self.vocab_idx,
            "idx_vocab": self.idx_vocab,
            "subwords": self.subwords,
        }
    
    @classmethod
    def load_cache(cls, name):
        cache_fields = Tokenizer.load_cache(name)
        if cache_fields is not None:
            vocab_idx = cache_fields["vocab_idx"]
            sub_words = cache_fields["subwords"]
            return cls(name, vocab_idx, sub_words), True
        else:
            raise Exception("Not yet created")

    @classmethod
    def train_wordpiece(self, name, final_vocab, subwords, vocab_size):
        start = time.time()
        progress = RegularPrinter()
        iteration = 0
        while len(final_vocab) < vocab_size and (time.time() - start) < 3600:
            progress.print(f"Delta {time.time() - start}, iteration {iteration}")
            pair_counts = collections.Counter()
            for word_subwords, count in subwords.items():
                word_subwords = word_subwords.split(" ")
                for i in range(len(word_subwords) - 1):
                    pair = (word_subwords[i], word_subwords[i + 1])
                    pair_counts[pair] += count

            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)            
            new_subwords = defaultdict(int)
            merged = 0
            for word_subwords, count in subwords.items():
                word_subwords = word_subwords.split(" ")

                merged_word = []
                i = 0
                while i < len(word_subwords):
                    if i < len(word_subwords) - 1 and (word_subwords[i], word_subwords[i + 1]) == best_pair:
                        merged_word.append(word_subwords[i] + word_subwords[i + 1]) 
                        merged += 1
                        i += 2
                    else:
                        merged_word.append(word_subwords[i])
                        i += 1
                new_subwords[" ".join(merged_word)] += count
            
            subwords = new_subwords
            merged_token = "".join(best_pair)
            final_vocab[merged_token] = pair_counts[best_pair]
            iteration += 1
        return WordPiece(name, final_vocab, subwords)

    def encode_document(self, document):
        tokens = []
        for words in self.split(document):
            for piece in self.encode(words):
                tokens.append(piece)
        return tokens

    def decode_idx(self, idx):
        return self.decode([idx])

    @lru_cache(maxsize=2 << 15)
    def encode(self, word):
        tokens = list(word)
        i = 0
        while i < len(tokens) - 1:
            combined = tokens[i] + tokens[i + 1]
            if combined in self.vocab_idx:
                tokens[i] = combined
                tokens.pop(i + 1)
            else:
                i += 1
        return list(map(lambda x: self.vocab_idx[x], tokens))
    
    def decode(self, tokens):
        return "".join(list(map(lambda x: self.idx_vocab[x], tokens)))

    @property
    def vocab_size(self):
        return len(self.vocab_idx)


"""
Wrappers around existing tokenizers
"""
class HuggingFaceTokenizerWrapper(Tokenizer):
    def __init__(self, name):
        self.bpe = BPE()
        self.tokenizer = HfTokenizer(self.bpe)
        self.tokenizer.add_special_tokens(["<PADDING>"])
        self.padding_index = self.tokenizer.encode("<PADDING>").ids[0]
        assert self.decode_idx(self.padding_index) == "<PADDING>"
        self.is_locked = False
        self.name = name

    def encode_document(self, documents):
        assert type(documents) != str
        self.tokenizer.train_from_iterator(documents)

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

    @classmethod
    def load_cache(cls, name):
        tokenizer_path = Tokenizer.get_tokenizer_path(name)
        print(tokenizer_path)
        hf = HuggingFaceTokenizerWrapper(name)
        if os.path.isfile(tokenizer_path):
            hf.tokenizer = hf.tokenizer.from_file(tokenizer_path)
            return hf, True
        return hf, False
