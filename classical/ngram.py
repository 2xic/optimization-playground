"""
https://web.stanford.edu/~jurafsky/slp3/3.pdf
"""
import numpy as np

class Dataloader:
    def __init__(self, window_size) -> None:
        self.window_size = window_size
        self.token_separator = lambda x: list(filter(lambda x: len(x) > 0, x.split(" ")))

    def load_documents(self, documents):
        for i in documents:
            yield self.load_document(i)

    def load_document(self, document):
        tokens = ()
        previous_token = ("<s>", )
        document = self.token_separator(document)
        for i in range(0, len(document)):
            next_token = tuple(document[i:i + self.window_size])
            tokens += (
                (previous_token, next_token, ),
            )
            previous_token = next_token
        tokens += (
            (previous_token, ("</s>", )),
        )
        return tokens

class NGramModel:
    def __init__(self) -> None:
        # todo: should not use a dictionary, but instead vectors
        self.counts = {}

    def train(self, token_documents):
        for document in token_documents:
            for (prev, next) in document:
                prev = tuple(prev)
                next = tuple(next)
                if prev not in self.counts:
                    self.counts[prev] = {}
                if next not in self.counts[prev]:
                    self.counts[prev][next] = 0
                self.counts[prev][next] += 1
        return self
    
    def p(self, prev, next):
        return self.counts[prev][next] / sum(self.counts[prev].values())
    
    def p_token_stream(self, tokens):
        p_sum = 1
        for (prev, next) in tokens:
            p_sum *= self.p(prev, next)
        return p_sum

def is_close(a, b):
    delta = abs(a - b)
    return delta < 0.01

if __name__ == "__main__":
    # verify with the examples in the book
    dataloader = Dataloader(
        window_size=1
    )
    token_documents = dataloader.load_documents([
        "I am Sam",
        "Sam I am",
        "I do not like green eggs and ham"
    ])
    model = NGramModel().train(token_documents)
    print(model.counts)
    assert is_close(model.p(("<s>", ), ("I", )), 0.67)
    assert is_close(model.p(("I", ), ("do", ), ), 0.33)
    print(dataloader.load_document("I am Sam"))
    print(model.p_token_stream(dataloader.load_document("I am Sam")))
