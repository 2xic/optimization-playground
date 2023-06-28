

class Vocab:
    def __init__(self) -> None:
        self.idx_2_word = {}
        self.word_2_idx = {}
        self.word_usage = {}
        self.UNKNOWN_IDX = self._add("<UNKNOWN>")
        self.PADDING_IDX = self._add("<PADDING>")
        self.MASK_IDX = self._add("<MASK>")
        self.MINIMUM_WORD_USAGE = -1
        self.SPECIAL_TOKENS = len(self.idx_2_word)

    @property
    def size(self):
        return len(self.idx_2_word)
    
    def fit(self, list_of_tokens):
        for i in list_of_tokens:
            if i not in self.word_usage:
                self.word_usage[i] = 1
            else:
                self.word_usage[i] += 1

    def add(self, list_of_tokens):
        if type(list_of_tokens) != list:
            raise Exception("Expected list")
        for i in list_of_tokens:
            if self.word_usage[i] < self.MINIMUM_WORD_USAGE:
                continue
            if i in self.word_2_idx:
                continue
            self._add(i)
        return self
    
    def _add(self, token):
        if token not in self.word_2_idx:
            idx = len(self.idx_2_word)

            self.word_2_idx[token] = idx
            self.idx_2_word[idx] = token
        return self.word_2_idx[token]
    
    def get(self, list_of_idx):
        if type(list_of_idx) != list:
            raise Exception("Expected list")
        return [
            self.word_2_idx.get(i, self.UNKNOWN_IDX)
            for i in list_of_idx
        ]

    def get_words(self, list_of_idx):
        if type(list_of_idx) != list:
            raise Exception("Expected list")
        return [
            self.idx_2_word.get(i, self.UNKNOWN_IDX)
            for i in list_of_idx
        ]
