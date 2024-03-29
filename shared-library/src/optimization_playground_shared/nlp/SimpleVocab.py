import torch

class Vocab:
    def __init__(self) -> None:
        self.index_vocab = {}
        self.vocab_index = {}
        self.locked = False
        self.PADDING_IDX = self.add("<PADDING>")

    def add(self, word):
        if not word in self.vocab_index and not self.locked:
            index = len(self.index_vocab)
            self.vocab_index[word] = index
            self.index_vocab[index] = word
            return index
        elif word in self.vocab_index:
            return self.vocab_index[word]
        return self.PADDING_IDX

class SimpleVocab:
    def __init__(self) -> None:
        self.vocab = Vocab()

    def encode(self, sentence):
        for i in sentence.split(" "):
            self.vocab.add(i)

    def get_tensor(self, sentence, sequence_length):
        if sequence_length == -1:
            output = []
            for index, i in enumerate(sentence.split(" ")[:sequence_length]):
                output.append(self.vocab.add(i))
            return torch.tensor([output])

        torch_tensor = torch.zeros((1, sequence_length)).fill_(self.vocab.PADDING_IDX).long()
        if sentence is not None:
            for index, i in enumerate(sentence.split(" ")[:sequence_length]):
                torch_tensor[0][index] = self.vocab.add(i)
        return torch_tensor
            
    def lock(self):
        self.vocab.locked = True
        return self
    
    @property
    def size(self):
        return len(self.vocab.index_vocab)
    