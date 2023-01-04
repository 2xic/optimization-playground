import torch

class Vocab:
    def __init__(self) -> None:
        self.word_idx = {}
        self.idx_word = {}
        self.is_locked = False
        self.unknown_idx = self._encode("<unknown>")
        self.end_idx = self._encode("<end>")

    def encode(self, sentence) -> torch.Tensor:
        return torch.tensor([[
            self._encode(i) for i in sentence.split(" ")
        ]])

    def decode(self, word):
        return self.idx_word[word]

    def _encode(self, word):
        word = word.lower()
        if word in self.word_idx:
            return self.word_idx[word]
        elif not self.is_locked:
            idx = len(self.word_idx)
            self.word_idx[word] = idx
            self.idx_word[idx] = word
            return idx
        else:
            return self.unknown_idx

    def lock(self):
        self.is_locked = True
        return len(self.word_idx) + 1
