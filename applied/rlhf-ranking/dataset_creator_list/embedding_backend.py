"""
So it's easier test different backends
"""
from optimization_playground_shared.apis.openai import OpenAiEmbeddings
from optimization_playground_shared.classics.bm_25 import BM25
from optimization_playground_shared.classics.tf_idf_wrapper import TfIdfWrapper

class EmbeddingBackend:
    def __init__(self, backend="bm25"):
        self.backend = backend
        if backend == "openai":
            # Backend 
            self._embedding_size = 1536
            self.transformer = OpenAiEmbeddings()
        elif backend == "tf_idf":
            # Backend 
            self.transformer = TfIdfWrapper(
                max_features=512,
                input='content',
                encoding='utf-8', decode_error='replace', strip_accents='unicode',
                lowercase=True, analyzer='word', stop_words='english',
                token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                ngram_range=(1, 2), norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,

            )
            self._embedding_size = 512
        elif backend == "bm25":
            embedding_size = 512
            self.transformer = BM25(max_features=embedding_size)
            self._embedding_size = embedding_size
        else:
            raise Exception(f"Unknown backend {backend}")

    def embedding_size(self):
        return self._embedding_size

    def get_embedding(self, document):
        if type(document) == str:
            document = [document]
        assert len(document) == 1
        return self.transformer.get_embedding(document)[0].tolist()
