import torch

class Tokenizer:
    def __init__(self):
        self.idx_words = {}
        self.word_idx = {}

    def encode_document_tensor(self, document, sequence_length):
        #print(document)
        X = torch.zeros((1, sequence_length), dtype=torch.long)
        for index, i in enumerate(self._get_Tokens(document)[:sequence_length]):
            #print((i, self._encode_token(i)))
            X[0][index] = self._encode_token(i)
        #print(X)
        return X
    
    def encode_documents(self, documents):
        for document in documents:
            self.encode_document(document)
        return self
    
    def encode_document(self, document):
        tokens = []
        for token in self._get_Tokens(document):
            tokens.append(self._encode_token(token))
        return tokens
    
    def decode(self, tensor):
        words = []
        for i in tensor:
            words.append(self.idx_words[i.item()])
        return " ".join(words)

    def _encode_token(self, token):
        if token not in self.word_idx:
            idx = len(self.idx_words)
            self.idx_words[idx] = token
            self.word_idx[token] = idx
        return self.word_idx[token]

    def _get_Tokens(self, document):
        return document.split(" ")
    
