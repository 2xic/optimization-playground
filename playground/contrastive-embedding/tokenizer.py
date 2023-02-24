import random

class Tokenizer:
    def __init__(self):
        self.tokens_idx = {}
        self.idx_tokens = {}
        self.size = 64
        self.fill_token_idx = self._encode_token("<FILL>")
        self.unknown_token_idx = self._encode_token("<UNKOWN>")

    @property
    def vocab(self):
        return len(self.tokens_idx)

    def encode(self, sentence):
        assert type(sentence) == str
        output = []
        for i in sentence.lower().split(" "):
            output.append(self._encode_token(i))
        length = len(output)
        output += [self.fill_token_idx, ] * (self.size - len(output))
        return output, length

    def encode_fuzz(self, sentence):
        (standard, length) = self.encode(sentence)
        for i in range(len(standard)):
            if random.randint(0, 5) == 2:
                standard[i] = self.unknown_token_idx
        return standard

    def _encode_token(self, token):
        if not token in self.tokens_idx:
            idx = len(self.tokens_idx)
            self.tokens_idx[token] = idx
            self.idx_tokens[idx] = token
        return self.tokens_idx[token]
