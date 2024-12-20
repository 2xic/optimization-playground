from optimization_playground_shared.classics.bm_25 import BM25
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    # Example usage
    corpus = [
        "this this is a sample and some bagels",
        "this is another example example",
        "more text data",
        "no more today of those bagels",
    ]

    bm25 = BM25()
    bm25.fit(list(corpus))

    # Transform documents
    encoded_corpus = [bm25.transform_debug(doc) for doc in corpus]

    # Transform a new document
    new_document = "this is the bagel data"
    new_doc_vector = bm25.transform_debug(new_document)

    print("Encoded Corpus:")
    for i, doc_vector in enumerate(encoded_corpus):
        print(f"Document {i} BM25 Vector:")
        #print(json.dumps(doc_vector, indent=2))
        print(doc_vector)

    print("\nNew Document BM25 Vector:")
    print(bm25.transforms(new_document))
    print("\nNew Document tf-idf Vector:")
    tf_idf = TfidfVectorizer()
    tf_idf.fit(corpus)
    print(tf_idf.transform([new_document]).todense())
    print("Limiting bm25 vocab features")
    bm25.max_features = 2
    bm25.fit(list(corpus))
    print(bm25.bm_25_features)
