import torch

class Vocab:
    def __init__(self) -> None:
        self.word_idx = {}
        self.idx_word = {}
        self.is_locked = False
        self.unknown_idx = self._encode("<unknown>")
        self.end_idx = self._encode("<end>")
        self.padding_idx = self._encode("<padding>")

    def encode(self, sentence) -> torch.Tensor:
        if type(sentence) == list:
            items = []
            max_length = 0
            for i in sentence:
                words = []
                for i in i.split(" "):
                    words.append(self._encode(i))
                items.append(words)
                max_length = max(len(words), max_length)

            tensor_item = torch.zeros((len(sentence), max_length)).fill_(self.padding_idx)
            for index, i in enumerate(items):
                tensor_item[index, :len(i)] = torch.tensor(i)
            return tensor_item.long()
        else:
            return torch.tensor([[
                self._encode(i) for i in sentence.split(" ")
            ]])

    def encode_file(self, file):
        with open(file, "r") as file:
            return self.encode(
                file.read().replace("\n", " ").split(".")
            )

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
