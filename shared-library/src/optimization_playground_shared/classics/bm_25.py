import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BM25:
    def __init__(self, k1=1.5, b=0.75, max_features=-1):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.idf = {}
        self.doc_lens = []
        self.vocab = set()
        self.bm_25_features = {}
        self.max_features = max_features

    def get_vocab_count(self, document):
        vectorizer = CountVectorizer(
            lowercase=True
        )
        # TODO: fix this to make it one vec operation.
        transformed = vectorizer.fit_transform([document]).todense().tolist()
        results = {}
        for key, value in vectorizer.vocabulary_.items():
            results[key] = transformed[0][value]
        return results    

    def fit_transforms(self, corpus):
        return self.fit(corpus)

    def train(self, corpus):
        return self.fit(corpus)

    def fit(self, raw_corpus):
        corpus = list(raw_corpus)
        for index, i in enumerate(corpus):
            assert type(i) == str
            corpus[index] = self.get_vocab_count(i)

        self.avg_doc_len = sum(len(doc) for doc in corpus) / len(corpus)
        doc_freqs = []
        self.doc_lens = [len(doc) for doc in corpus]
        # doc frequency
        for doc in corpus:
            doc_freqs.append(doc)
            self.vocab.update(doc.keys())
        # idf         
        num_docs = len(corpus)
        df = Counter()
        for doc_freq in doc_freqs:
            for term in doc_freq:
                df[term] += doc_freq[term]
        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1)
        self.bm_25_features = self._limit_features()
        return self.transforms(raw_corpus)

    def score_term(self, term, term_freq, doc_len):
        idf = self.idf[term]
        score = idf * (term_freq * (self.k1 + 1)) / (term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
        return score

    def transforms(self, document):
        if type(document) == list:
            return [self._transform_document(x) for x in document]
        return [self._transform_document(document)]

    def _transform_document(self, document):
        doc_freq = self.get_vocab_count(document)
        doc_len = sum(doc_freq.values())
        vector = np.zeros((len(self.bm_25_features)))
        for index, term in enumerate(self.bm_25_features.keys()):
            vector[index] = self.score_term(term, doc_freq.get(term, 0), doc_len)
        return vector
    
    def _limit_features(self):
        scores = {}
        for _, term in enumerate(self.vocab):
            scores[term] = self.idf.get(term, 0)
        return {
            key: value for key, value in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.max_features]
        }

    def get_embedding(self, x):
        return self.transforms(x)

    def transform_debug(self, document):
        doc_freq = self.get_vocab_count(document)
        doc_len = sum(doc_freq.values())
        return {
            term: self.score_term(term, doc_freq.get(term, 0), doc_len) for term in self.vocab
        }
