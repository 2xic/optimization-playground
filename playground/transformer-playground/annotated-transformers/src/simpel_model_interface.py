from .make_models import make_model
from .helpers import subsequent_mask
import torch

class SimpleModelInterface:
    def __init__(
        self,
        layers=2
    ) -> None:
        self.is_freezed = False
        self.layers = layers

        self.word_idx = {}
        self.idx_word = {}
        self.UNKNOWN_IDX = self._add_tokens("<UNKNOWN>")

    def build_model(self, documents):
        for i in documents:
            for j in i.split(" "):
                self._add_tokens(j)
        self._freeze()

        self.model = make_model(
            len(self.word_idx),
            len(self.word_idx),
            self.layers
        )
    
    def get_training(self, documents):
        for document in documents:
            doc = document.split(" ")
            src = torch.LongTensor([[
                self.word_idx.get(i, self.UNKNOWN_IDX) for i in doc
            ]])
            for i in range(0, len(doc) - 2):
                yield src[:, i:i+2], src[:, i+2:i+4]

    def forward(self, document, steps=30):
        src = torch.LongTensor([[
            self.word_idx.get(i, self.UNKNOWN_IDX) for i in document.split(" ")
        ]])
        src_mask = torch.ones((1, ) + src.shape)
        memory = self.model.encode(src, src_mask)
        ys = torch.zeros(1, 1).type_as(src)
        words = []
        for i in range(steps):
            out = self.model.decode(
                memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
            )

            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
            words.append(
                self.idx_word[next_word.item()]
            )
        return words

    def _add_tokens(self, word):
        if self.is_freezed:
            raise Exception("Vocab is freezed")

        if word not in self.word_idx:
            size = len(self.word_idx)
            self.word_idx[word] = size
            self.idx_word[size] = word
        return self.word_idx[word]

    def _freeze(self):
        self.is_freezed = True

