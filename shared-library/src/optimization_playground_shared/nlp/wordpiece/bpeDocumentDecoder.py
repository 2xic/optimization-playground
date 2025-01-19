from .bpe import BPE
import torch


def get_document_words(bpe: BPE, text):
    # todo: create a good tokenizer, this does not really work for code tokens
    if type(text) == bytes:
        text = text.replace(b"\r\n", b"\n")
        text = text.replace(b"\r", b"\n")
        return bpe.encode(text.decode())
    else:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        return bpe.encode(text)

def encode_document_text(vocab: BPE, text, tensor_x, tensor_y, entries_index, SEQUENCE_LENGTH, SKIP_SIZE):
    words = get_document_words(vocab, text)
    count_words = len(words)
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: x, words)), dtype=torch.long)
    for i in range(0, count_words - 1, SKIP_SIZE):
        start_index = 0
        if i > SEQUENCE_LENGTH:
            start_index = i - SEQUENCE_LENGTH
        context = words[start_index:i]
        # Since we are flattening we want this to match ... 
        next_token = words[i:i+SEQUENCE_LENGTH]
        # add the entries
        tensor_x[entries_index, :context.shape[-1]] = context
        tensor_y[entries_index, :next_token.shape[-1]] = next_token
        entries_index += 1
    return tensor_x, tensor_y, entries_index

def get_document_dataset(vocab: BPE, documents, SEQUENCE_LENGTH, SKIPS_SIZE=1) -> tuple[torch.Tensor]:
    assert type(documents) == list
    entries_count = 0
    for i in documents:
        # -2 as we need the first token to be placed and the last token
        entries_count += len(get_document_words(vocab, i)) - 2
    entries_count = (entries_count // SKIPS_SIZE) + len(documents)
    X = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.index.padding_idx, dtype=torch.long)
    y = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.index.padding_idx, dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, y, entries_index = encode_document_text(vocab, document, X, y, entries_index, SEQUENCE_LENGTH, SKIPS_SIZE)
    assert entries_index == X.shape[0]
 #   assert not torch.all(X == 0), "All zeros is bad"
 #   assert not torch.all(y == 0), "All zeros is bad"
    return X, y

def get_document_sequence(vocab: BPE, documents, SEQUENCE_LENGTH) -> tuple[torch.Tensor]:
    X, y = get_document_dataset(vocab, documents, SEQUENCE_LENGTH, SEQUENCE_LENGTH) 
    return X
