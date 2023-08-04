"""
I would assume this won't work well, but we need to test.

Update: it does work well, but might not work well when scaled up 
even more 
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
        # Token program or raw_program
        self.vectorizer.fit(
            dataset.token_program
        )
        return self

    def predict(self, dataloader, dataset: TokenizedShellCode):
        return self.vectorizer.transform(dataset.token_program)
