from .openai_embeddings import get_embeddings
import numpy as np

class OpenAiAdaEmbeddings:
    def __init__(self):
        self.model = "text-embedding-ada-002"

    def get_embedding(self, X: str):
        return np.asarray(get_embeddings(
                text=X,
                model=self.model,
        )["data"][0]["embedding"]).astype(np.float32).tolist()
