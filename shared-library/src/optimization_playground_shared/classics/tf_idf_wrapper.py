from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfIdfWrapper:
    def __init__(self, **kwargs) -> None:
        self.encoder = TfidfVectorizer(**kwargs)
        self.is_trained = False

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return np.asarray(self.encoder.fit_transform(x).todense().tolist())
    
    def transforms(self, x):
        return np.asarray(self.encoder.transform(x).todense().tolist())

    def get_embedding(self, x):
        return self.transforms(x)
