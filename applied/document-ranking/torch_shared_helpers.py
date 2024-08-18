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
    if len(X) == 0:
        X.append(vocab.get_tensor("", sequence_length=sequence_length))
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

# @jit()
def encode_document_text(vocab: SimpleVocab, text, tensor_x, tensor_y, entries_index, sequence_length):
    words = get_document_words(text)
    count_words = len(words)
    vocab_add = vocab.vocab.get 
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: vocab_add(x), words)), dtype=torch.long)
 #   print(words)
    for i in range(count_words - 1):
        start_index = 0
        if i > sequence_length:
            start_index = i - sequence_length
        context = words[start_index:i]
        next_token = words[i]
        # add the entries
        tensor_x[entries_index, :context.shape[-1]] = context
        tensor_y[entries_index] = next_token
        entries_index += 1
    return tensor_x, tensor_y, entries_index

def get_document_dataset(vocab: SimpleVocab, documents, sequence_length):
    assert type(documents) == list
    entries_count = 0
    for i in documents:
        entries_count += len(get_document_words(i))

    X = torch.full(size=(entries_count, sequence_length), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    y = torch.full(size=(entries_count, ), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, y, entries_index = encode_document_text(vocab, document, X, y, entries_index, sequence_length)
    assert not torch.all(X == 0), "All zeros is bad"
    assert not torch.all(y == 0), "All zeros is bad"
    return X, y
