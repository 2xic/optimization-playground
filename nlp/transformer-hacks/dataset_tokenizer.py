"""
When working on larger dataset, you might want to be able to preprocess the dataset for the tokenizer etc.
"""
import torch
from typing import Union, Optional, Dict, Any
from abc import ABC, abstractmethod
from optimization_playground_shared.nlp.SimpleVocab import splitter
import json
import os
from collections import defaultdict

class Tokenizer(ABC):
    name: str
    
    @property
    @abstractmethod
    def add_word(self, word):
        pass

    @property
    @abstractmethod
    def decode_idx(self, word_idx):
        pass

    def split(self, doc):
        return splitter(doc)

    def save_cache(self):
        tokenizer_path = self.get_tokenizer_path()
        with open(tokenizer_path, "w") as file:
            json.dump(self.get_cache_fields(), file)

    def load_cache(self) -> Dict[str, Any] :
        tokenizer_path = self.get_tokenizer_path()
        if os.path.isfile(tokenizer_path):
            with open(tokenizer_path, "r") as file:
                return json.load(file)
    
    def get_tokenizer_path(self):
        tokenizer_path = os.path.join(
            os.path.dirname(__file__),
            "tokenizer",
            self.name + ".json"
        )
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        return tokenizer_path

    @abstractmethod
    def get_cache_fields(self) -> Dict[str, any]:
        pass

    def build_from_files(self, iterator) -> 'Tokenizer':
        for i in iterator:
            with open(i, "r") as file:
                tokens = list(set(splitter(file.read())))
            for i in tokens:
                self.add_word(i)
        return self

class SimpleTextEncoder(Tokenizer):
    def __init__(self, name):
        self.name = name
        self.vocab_idx = {}
        self.idx_vocab = {}
        self.is_locked = False
        self.padding_index = self.add_word("<PADDING>")

    def get_cache_fields(self):
        return {
            "vocab_idx": self.vocab_idx,
            "idx_vocab": self.idx_vocab,
            "is_locked": self.is_locked,
            "padding_index": self.padding_index,
        }
    
    def load_cache(self):
        cache_fields = super().load_cache()
        self.vocab_idx = cache_fields["vocab_idx"]
        self.idx_vocab = cache_fields["idx_vocab"]
        self.is_locked = cache_fields["is_locked"]
        self.padding_index = cache_fields["padding_index"]
        return self
    
    def add_word(self, word):
        if word not in self.vocab_idx:
            idx = len(self.vocab_idx)
            self.vocab_idx[word] = idx
            self.idx_vocab[idx] = word
            assert type(word) == str, f"Word == {word}, Idx == {idx}"
        return self.vocab_idx[word]

    def decode_idx(self, word_idx: Union[torch.Tensor, int]):
        word_idx = word_idx.item() if isinstance(word_idx, torch.Tensor) else word_idx
        return self.idx_vocab[word_idx]

# Rewritten version of the Bpe in wordpeice
class BpeTokenizer(Tokenizer):
    def __init__(self):
        # Stats
        self.word_index = {}
        self.index_word = {}
        self.word_frequency = {}
        self.tokens_index = {}
        self.merged_pairs = []
        # System tokens
        # Any tokens after this can be changed
        self.system_tokens = []
        self.padding_index = self.register_system_tokens("<PADDING>")
        self.is_readonly = True

    def register_system_tokens(self, token):
        idx = len(self.system_tokens)
        self.system_tokens.append(token)
        return idx

    def get_cache_fields(self):
        return {
            "word_index": self.word_index,
            "index_word": self.index_word,
            "word_frequency": self.word_frequency,
            "tokens_index": self.tokens_index,
            "system_tokens": self.system_tokens,
            "padding_index": self.padding_index,
        }
    
    def add_document(self, document):
        for i in self.split(document):
            self.add_word(i)

    def encode(self, document):
        assert type(document) == str
        output = []
        for token in self.split(document):
            for word in self._encode(token):
                output.append(self.tokens_index[word])
        return output

    def _encode(self, word):
        # Should only contain system tokens
        if word in self.word_index:
            return self.word_index[word]
        assert type(word) == str
        # Add end-of-word token
        word = list(word) + ['</w>']
        for pair in self.merged_pairs:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == pair:
                    word[i] = word[i] + word[i + 1]
                    del word[i + 1]
                else:
                    i += 1
        return word[:-1]

    def decode(self, tokens):
        assert type(tokens) == list
        output = []
        padding_token = self.padding_index
        for token in tokens:
            if padding_token == token:
                continue
            print(token, self.index_word)
            output.append(self.index_word[token])
        return "".join(output)
    
    def decode_idx(self, token):
        return self.decode([token])

    def merge(self, n=10):
        for i in range(n):
            self._merge()

    def _merge(self):
        pairs = self._get_stats()
        if len(pairs) == 0:
            return False
        pair = max(
            pairs,
            key=lambda x: pairs[x]
        )
        self.merged_pairs.append(pair)
        for index in self.index_word:
            word: str = self.index_word[index]
            joined = "".join(pair)
            splitted = " ".join(pair)
            output = word.replace(splitted, joined)
            # Pair got jointed need to update
            if word != output:
                self.update(
                    word,
                    output,
                    pair
                )
        # This should update it. 
        for index, value in enumerate(set(self.system_tokens + list(self.tokens_index.keys()))):
            self.tokens_index[value] = index
            self.index_word[index] = value
        assert max(self.tokens_index.values()) <= len(self.tokens_index)
        return True

    def _get_stats(self):
        """
        We get the frequency of word pairs in the total vocab
        """
        pairs = defaultdict(int)
        for word in self.word_index:
            frequency = self.word_frequency[word]
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency
        return pairs

    """
    Indexing logic
    """
    def add_word(self, word):
        word = self._tokenize_words(word)
        if word in self.word_frequency:
            self.word_frequency[word] += 1
        else:
            idx = len(self.word_index)
            self.word_index[word] = idx
            self.index_word[idx] = word
            self.word_frequency[word] = 1
        
        for symbol in word:
            if not symbol in self.tokens_index:
                self.tokens_index[symbol] = None

    def update(self, from_word, to_word, pair):
    #    print(self.word_index[to_word])
        self.word_index[to_word] = self.word_index[from_word]
        self.index_word[self.word_index[to_word]] = to_word
        self.word_frequency[to_word] = self.word_frequency[from_word]
        del self.word_index[from_word]
        del self.word_frequency[from_word]
        # Update the mapping token if not already updated
        if pair[0] in self.tokens_index:
            self.tokens_index[pair[0] + pair[1]] = None

    def _tokenize_words(self, word: str):
        return " ".join(list(word))
