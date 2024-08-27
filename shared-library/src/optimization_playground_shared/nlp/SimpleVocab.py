import torch
import string

def splitter(sentence):
    tokenizer = []
    token = ""
    sentence = sentence.replace("\r\n", "\n")
    is_white_space = False
    for i in sentence:
        if is_white_space and i not in string.whitespace:
            if len(token) > 0:
                tokenizer.append(token)
                token = ""
            is_white_space = False
        
        if i in string.punctuation:
            if len(token) > 0:
                tokenizer.append(token)
            tokenizer.append(i)
            token = ""
        elif i in string.whitespace and not is_white_space:
            if len(token) > 0:
                tokenizer.append(token)
                token = ""
            token += i
            is_white_space = True
        else:
            token += i
    if len(token) > 0:
        tokenizer.append(token)
    return tokenizer            

class Vocab:
    def __init__(self) -> None:
        self.index_vocab = {}
        self.vocab_index = {}
        self.locked = False
        self.PADDING_IDX = self.add("<PADDING>")

    def get(self, word):
        out = self.vocab_index.get(word, self.PADDING_IDX)
        return out
    
    def add(self, word):
        if not self.locked and not word in self.vocab_index:
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
        X = []
        for i in splitter(sentence):
            X.append(self.vocab.add(i))
        return X
    
    def decode(self, tokens):
        return "".join(self.decoded_tokens(tokens))

    def decoded_tokens(self, tokens):
        X = []
        for i in tokens:
            X.append(self.vocab.index_vocab[i])
        return X

    def get_tensor(self, sentence, sequence_length):
        if sequence_length == -1:
            output = []
            for index, i in enumerate(splitter(sentence)[:sequence_length]):
                output.append(self.vocab.add(i))
            return torch.tensor([output])

        torch_tensor = torch.zeros((1, sequence_length)).fill_(self.vocab.PADDING_IDX).long()
        if sentence is not None:
            for index, i in enumerate(splitter(sentence)[:sequence_length]):
                torch_tensor[0][index] = self.vocab.add(i)
        return torch_tensor

    def get_tensor_from_tokens(self, tokens, sequence_length):
        torch_tensor = torch.zeros((1, sequence_length)).fill_(self.vocab.PADDING_IDX).long()
        c = torch.tensor(tokens)        
        torch_tensor[0, :len(tokens)] = c[:]
        return torch_tensor
            
    def lock(self):
        self.vocab.locked = True
        return self
    
    @property
    def size(self):
        return len(self.vocab.index_vocab)
