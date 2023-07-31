"""
I would assume this won't work well, but we need to test.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from get_tokenized_shellcode import TokenizedShellCode

class TfIdfModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, dataloader, dataset: TokenizedShellCode):
        """
        Raw or not ? 
        """
        self.vectorizer.fit(
            dataset.raw_program
        )
        return self

    def predict(self, dataloader, dataset: TokenizedShellCode):
        return self.vectorizer.transform(dataset.raw_program)
