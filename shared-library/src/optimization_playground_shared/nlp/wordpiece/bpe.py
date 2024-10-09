"""
Partially based off algorithm in "Neural Machine Translation of Rare Words with Subword Units" ( https://arxiv.org/pdf/1508.07909.pdf )
"""
from collections import defaultdict
from typing import List
import tqdm 
from ..SimpleVocab import splitter

class VocabIndex:
    def __init__(self) -> None:
        self.system_tokens = { }
        self.padding_idx = self._add_system_token("<PADDING>")
        self._system_tokens_length = len(self.system_tokens)
        self.word_index = {}
        self.index_word = {}
        self.word_frequency = {}

        # Stores the actual full token mapping index. Needs to be recalculated later on.
        self.tokens_index = {}
        self.index_tokens = {}

        self.is_readonly = False

    def _add_system_token(self, word):
        self.system_tokens[word] = len(self.system_tokens)
        return self.system_tokens[word]

    def add_sentence(self, sentence):
        for i in splitter(sentence):
            self.add_word(i)

    def add_word(self, word):
        assert not self.is_readonly
        word = self._tokenize_words(word)
        if not word in self.word_index:
            idx = self.size
            self.word_index[word] = idx
            self.index_word[idx] = word
            self.word_frequency[word] = 1
        else:
            self.word_frequency[word] += 1
        # add the word, but with no index.
        for symbol in word:
            if symbol not in self.tokens_index:
                self.tokens_index[symbol] = None

    def update(self, from_word, to_word, pair):
        # print(f"{from_word} -> {to_word}")
        self.word_index[to_word] = self.word_index[from_word]
        self.index_word[self.word_index[to_word]] = to_word
        self.word_frequency[to_word] = self.word_frequency[from_word]
        del self.word_index[from_word]
        del self.word_frequency[from_word]
        # Update the mapping token if not already updated
        if pair[0] in self.tokens_index:
            # del self.tokens_index[pair[0]]
            # adds a new token pair.
            self.tokens_index[pair[0] + pair[1]] = None
            # Delete the old token as we have optimized

    def get_token_by_index(self, idx):
        if idx in self.system_tokens:
            return self.system_tokens[idx]
        return self.tokens_index[idx]

    @property
    def size(self):
        return len(self.word_index) + self._system_tokens_length

    def _tokenize_words(self, word: str):
        return " ".join(list(word))

class BPE:
    def __init__(self, show_progress=True) -> None:
        self.index = VocabIndex()
        self.progress = tqdm.tqdm if show_progress else list
        self.merged_pairs = []
    
    # Allows usage of custom tokenizer
    def add_tokens(self, input: List[str]):
        for i in input:
            self.index.add_sentence(i)
        return self

    # Uses the built in tokenizer
    def add_vocab(self, input: str):
        self.index.add_sentence(input)
        return self

    def add_word(self, word: str, count: int):
        for _ in range(count):
            self.index.add_sentence(word)
        return self

    def merge(self, n=10):
        for _ in self.progress(range(n)):
            if not self.run_merge_step():
                break
        return self

    def merge_until(self, token_size=10_000):
        last_token_count = float('inf')
        counter = 0
        while token_size < last_token_count and (counter < 10):
            if not self.run_merge_step() or len(self.index.tokens_index) == last_token_count:
                counter += 1
            else:
                counter = 0
                last_token_count = len(self.index.tokens_index)
        return self

    def run_merge_step(self):
        pairs = self._get_stats()
        if len(pairs) == 0:
            return False
        pair = max(
            pairs,
            key=lambda x: pairs[x]
        )
        self.merged_pairs.append(pair)
        for index in self.index.index_word:
            word = self.index.index_word[index]
            joined = "".join(pair)
            splitted = " ".join(pair)
            output = word.replace(splitted, joined)
            # Pair got jointed need to update
            if word != output:
                self.index.update(
                    word,
                    output,
                    pair
                )
        
        for index, value in enumerate(list(self.index.system_tokens.keys()) + list(self.index.tokens_index.keys())):
            self.index.tokens_index[value] = index
            self.index.index_tokens[index] = value
        assert max(self.index.tokens_index.values()) <= self.size
        return True

    def _get_stats(self):
        """
        We get the frequency of word pairs in the total vocab
        """
        pairs = defaultdict(int)
        for word in self.index.word_index:
            frequency = self.index.word_frequency[word]
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency
        return pairs
    
    def encode(self, sentence):
        assert type(sentence) == str
        output = []
        for sentence in splitter(sentence):
            for word in self._encode(sentence):
                output.append(self.index.get_token_by_index(word))
        return output
    
    def _encode(self, word):
        assert type(word) == str
        word = list(word) + ['</w>']  # Add end-of-word token
        for pair in self.merged_pairs:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == pair:
                    word[i] = word[i] + word[i + 1]  # Merge
                    del word[i + 1]  # Remove the merged part
                else:
                    i += 1
        return word[:-1]
    
    def decode(self, tokens):
        assert type(tokens) == list
        output = []
        padding_token = self.index.padding_idx
        for token in tokens:
            if padding_token == token:
                continue
            output.append(self.index.index_tokens[token])
        return "".join(output)  
    
    def lock(self):
        self.index.is_readonly = True
    
    @property
    def size(self):
        return len(self.index.tokens_index) + 1

if __name__ == "__main__":
    # example word form the paper
    results = BPE(
        show_progress=False,
    ).add_word(
        "low",
        5
    ).add_word(
        "lowest",
        2
    ).add_word(
        "newer",
        6
    ).add_word(
        "wider",
        3
    ).add_tokens(
        "I love bagels"
    ).merge_until(
        token_size=1
    )
    print(results.encode(
        "lowest"
    ))
