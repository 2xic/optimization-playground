"""
Partially based off algorithm in "Neural Machine Translation of Rare Words with Subword Units" ( https://arxiv.org/pdf/1508.07909.pdf )
"""
from collections import defaultdict

class VocabIndex:
    def __init__(self) -> None:
        self.word_index = {}
        self.index_word = {}
        self.word_frequency = {}

    def add_sentence(self, sentence):
        for i in sentence.split(" "):
            if len(i):
                self.add_word(i)


    def add_word(self, word):
        word = self._normalize(word)
        if not word in self.word_index:
            idx =  len(self.word_index)
            self.word_index[word] = idx
            self.index_word[idx] = word
            self.word_frequency[word] = 1
        else:
            self.word_frequency[word] += 1

    def update(self, from_word, to_word):
        #print(from_word)
        #print(to_word)
        #print(self.word_frequency[from_word])
        self.word_index[to_word] = self.word_index[from_word]
        self.index_word[self.word_index[to_word]] = to_word
        self.word_frequency[to_word] = self.word_frequency[from_word]
        del self.word_index[from_word]
        del self.word_frequency[from_word]

    def _normalize(self, word: str):
        return " ".join(list(word.lower()))

class BPE:
    def __init__(self) -> None:
        self.index = VocabIndex()
        self.merges = 10

    def add_vocab(self, input: str):
        self.index.add_sentence(input)
        return self

    def add_word(self, word: str, count: int):
        for _ in range(count):
            self.index.add_sentence(word)
        return self

    def merge(self):
        for _ in range(self.merges):
            pairs = self._get_stats()
            pair = max(
                pairs,
                key=lambda x: pairs[x]
            )
            
            for index in self.index.index_word:
                word = self.index.index_word[index]
                output = word
                joined = "".join(pair)
                splitted = " ".join(pair)
                output = output.replace(splitted, joined)
                if word != output:
                    self.index.update(
                        word,
                        output
                    )    
            print(self.index.word_index)

        return self.index.word_index

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

if __name__ == "__main__":
    # example word form the paper
    results = BPE().add_word(
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
    ).merge()


