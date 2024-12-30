from .SimpleVocab import SimpleVocab
import torch
import os
from .DocumentEncoder import get_document_words

def encode_document_text(vocab: SimpleVocab, text, tensor_x, entries_index, SEQUENCE_LENGTH):
    words = get_document_words(text)
    count_words = len(words)
    vocab_add = vocab.vocab.add 
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: vocab_add(x), words)), dtype=torch.long)
    for i in range(0, count_words - 1, SEQUENCE_LENGTH):
        context = words[i:i + SEQUENCE_LENGTH]
        tensor_x[entries_index, :context.shape[-1]] = context
        entries_index += 1
    return tensor_x, None, entries_index

def get_document_dataset(vocab: SimpleVocab, documents, SEQUENCE_LENGTH) -> tuple[torch.Tensor]:
    assert type(documents) == list
    entries_count = 0
    for i in documents:
        # -2 as we need the first token to be placed and the last token
        entries_count += ((len(get_document_words(i))) // SEQUENCE_LENGTH) + 1
    entries_count += 1

    X = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, _, entries_index = encode_document_text(vocab, document, X, entries_index, SEQUENCE_LENGTH)

    if os.environ.get("DEBUG_MODE") is not None:
        assert not torch.all(X == 0), "All zeros is bad"
    return X
