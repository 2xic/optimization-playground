from sklearn.feature_extraction.text import TfidfVectorizer
from optimization_playground_shared.apis.openai_ada_embeddings import OpenAiAdaEmbeddings
from b25 import BM25

class TfIdfWrapper:
    def __init__(self, **kwargs) -> None:
        self.encoder = TfidfVectorizer(**kwargs)
        self.is_trained = False

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.encoder.fit_transform(x)
    
    def transforms(self, x):
        return self.encoder.transform(x)

class OpenAiEmbeddingsWrapper:
    def __init__(self) -> None:
        self.is_trained = False
        self.encoder = OpenAiAdaEmbeddings()

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        return self.transforms(x)
    
    def transforms(self, x):
        X = []
        for i in x:
            X.append(self.encoder.get_embedding(i))
        return X

class B25Wrapper:
    def __init__(self) -> None:
        self.is_trained = False
        self.encoder = BM25()
        self.x = None

    def train(self, x):
        assert self.is_trained == False
        self.is_trained = True
        self.encoder.fit(x)
        self.x = x
        return self.transforms(x)
    
    def transforms(self, x):
        return self.encoder.transform(x, self.x)
    