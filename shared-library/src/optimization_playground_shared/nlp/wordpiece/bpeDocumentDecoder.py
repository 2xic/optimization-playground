from .bpe import BPE
import torch


def get_document_words(bpe: BPE, text):
    # todo: create a good tokenizer, this does not really work for code tokens
    if type(text) == bytes:
        text = text.replace(b"\r\n", b"\n")
        text = text.replace(b"\r", b"\n")
        return bpe.encode_sentences(text.decode())
    else:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        return bpe.encode_sentences(text)

def encode_document_text(vocab: BPE, text, tensor_x, tensor_y, entries_index, SEQUENCE_LENGTH, SKIP_SIZE):
    words = get_document_words(vocab, text)
    count_words = len(words)
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: x, words)), dtype=torch.long)
    for i in range(1, count_words - 1, SKIP_SIZE):
        start_index = 0
        if i > SEQUENCE_LENGTH:
            start_index = i - SEQUENCE_LENGTH
        context = words[start_index:i]
        next_token = words[i+1:i+SEQUENCE_LENGTH + 1]
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
    entries_count = entries_count // SKIPS_SIZE + 1
    X = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.get_system_token_index("<PADDING>"), dtype=torch.long)
    y = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.get_system_token_index("<PADDING>"), dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, y, entries_index = encode_document_text(vocab, document, X, y, entries_index, SEQUENCE_LENGTH, SKIPS_SIZE)
    assert not torch.all(X == 0), "All zeros is bad"
    assert not torch.all(y == 0), "All zeros is bad"
    # Random sampling out of the dataset for better coverage
    indices = torch.randint(0, X.size(0), size=(entries_count // 32,)) if entries_count > 32 else torch.arange(0, entries_count).reshape((-1, 1)).long()
    return X[indices], y[indices]

