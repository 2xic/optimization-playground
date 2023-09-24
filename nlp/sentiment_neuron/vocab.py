import torch

class Vocab:
    def __init__(self):
        self.is_locked = False
        self.idx_token = {}
        self.token_idx = {}
        self.PADDING_IDX = self.add_token("<PADDING>")
        self.UNKNOWN_IDX = self.add_token("<UNKNOWN>")

    def lock(self):
        self.is_locked = True
        return self

    def add_token(self, i):
        if not i in self.token_idx and not self.is_locked:
            idx = len(self.idx_token)
            self.idx_token[idx] = i
            self.token_idx[i] = idx
        return self.token_idx[i]
    
    def add_sentence(self, sentence):
        x = []
        for i in sentence:
            i = i.lower()
            x.append(self.add_token(i))
        return torch.tensor(x)
    
    def process_dataset(self, sentences):
        for sentence in sentences:
            for i in sentence:
                self.add_token(i)
        return self
    
    def get_dataset(self, sentences, device=None):
        x = []
        y = []
        max_size = 0
        for sentence in sentences:
            vector_x = []
            vector_y = []
            for i in range(1, len(sentence)):
                vector_x.append(self.add_token(sentence[i - 1]))
                vector_y.append(self.add_token(sentence[i]))
            max_size = max(max_size, len(vector_y))
            x.append(vector_x)
            y.append(vector_y)
        x_torch = torch.zeros((len(sentences), max_size), device=device, dtype=torch.long).fill_(self.PADDING_IDX)
        y_torch = torch.zeros((len(sentences), max_size), device=device, dtype=torch.long).fill_(self.PADDING_IDX)
        for index, (i, j) in enumerate(zip(x, y)):
            x_torch[index, :len(i)] = torch.tensor(i)
            y_torch[index, :len(j)] = torch.tensor(j)
        return x_torch, y_torch

    def get_encoded(self, sentence, device=None):
        assert type(sentence) == str
        x_torch = torch.zeros((1, len(sentence)), device=device, dtype=torch.long).fill_(self.PADDING_IDX)

        for index, i in enumerate(sentence):
            x_torch[0][index] = self.add_token(i)
        return x_torch
        
    def get_vocab_size(self):
        return len(self.idx_token)
    
    def decode(self, indexes):
        output = []
        for i in indexes:
            output.append(self.idx_token[i])
        return "".join(output)
