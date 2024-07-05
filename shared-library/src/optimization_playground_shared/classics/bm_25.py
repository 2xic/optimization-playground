import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        self.vocab = set()

    def _normalizer_tokens(self, document):
        vectorizer = CountVectorizer()
        _ = vectorizer.fit_transform([document])
        return vectorizer.get_feature_names_out()
            
        
    def train(self, corpus):
        return self.fit(corpus)

    def fit(self, raw_corpus):
        corpus = list(raw_corpus)
        for index, i in enumerate(corpus):
            assert type(i) == str
            corpus[index] = self._normalizer_tokens(i)

        self.avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)
        self.doc_freqs = []
        self.doc_lens = [len(doc) for doc in corpus]
        # doc frequency
        for doc in corpus:
            doc_freq = Counter(doc)
            self.doc_freqs.append(doc_freq)
            self.vocab.update(doc_freq.keys())
        # idf         
        num_docs = len(corpus)
        df = Counter()
        for doc_freq in self.doc_freqs:
            for term in doc_freq:
                df[term] += 1
        
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
        return self.transforms(raw_corpus)

    def score_term(self, term, term_freq, doc_len):
        if term in self.idf:
            idf = self.idf.get(term, 0)
            score = idf * (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
            return score
        else:
            return 0.0

    def transforms(self, document):
        if type(document) == list:
            return [self._transform_document(x) for x in document]
        return [self._transform_document(document)]

    def _transform_document(self, document):
        document = self._normalizer_tokens(document)
        doc_freq = Counter(document)
        doc_len = len(document)
        vector = np.zeros((len(self.vocab)))
        for index, term in enumerate(self.vocab):
            vector[index] = self.score_term(term, doc_freq[term], doc_len)
        return vector

    def transform_debug(self, document):
        doc_freq = Counter(document)
        doc_len = len(document)
        return {term: self.score_term(term, doc_freq[term], doc_len) for term in self.vocab}
