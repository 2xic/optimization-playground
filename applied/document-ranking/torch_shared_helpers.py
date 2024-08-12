import torch
import pickle
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import os

def get_document_words(text):
    words = text.lower().replace(":", " : ").strip().split(" ")
    words = list(filter(lambda x: len(x), words))
    return words

def encode_document_embed_text(vocab, text, sequence_length):
    X = []
    words = get_document_words(text)
    start_index = 0
    for i in range(start_index, len(words) - 1, sequence_length):
        X.append(vocab.get_tensor(
            " ".join(words[max(i-sequence_length, 0):i]), sequence_length=sequence_length))
    return torch.concat(X)

def create_vocab_dataset(documents) -> SimpleVocab:
    source = ".source_vocab_metadata"
    if not os.path.isfile(source):
        source_vocab = SimpleVocab()
        for i in documents:
            # We first need to figure out the vocab size. We can do random sampling later on to not have 
            # all the batches in memory.
            source_vocab.encode(i)
        with open(source, "wb") as file:
            pickle.dump(source_vocab, file)
        return source_vocab
    else:
        with open(source, "rb") as file:
            return pickle.load(file)
