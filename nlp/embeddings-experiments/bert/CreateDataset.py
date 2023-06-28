
class CreateDataset:
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def process(self, documents):
        X = []
        for doc in documents:
            tokens = []
            for _, token in enumerate(doc):
                print(token, end=" ")
                if token < self.vocab.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
            filtered = list(filter(lambda x: x > self.vocab.SPECIAL_TOKENS, tokens))
            X.append(filtered)
        return X
