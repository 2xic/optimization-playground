import os
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from pre_generator import get_cache_file

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 64
CACHE_FILE = ".model_state.pkt"

def get_document_words(text):
    # todo: create a good tokenizer, this does not really work for code tokens
    if type(text) == bytes:
        return text.decode().split(" ")
    else:
        return text.split(" ")

def encode_document_text(vocab: SimpleVocab, text, tensor_x, tensor_y, entries_index):
    words = get_document_words(text)
    count_words = len(words)
    vocab_add = vocab.vocab.get 
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: vocab_add(x), words)), dtype=torch.long)
    for i in range(count_words - 1):
        start_index = 0
        if i > SEQUENCE_LENGTH:
            start_index = i - SEQUENCE_LENGTH
        context = words[start_index:i]
        next_token = words[i]
        # add the entries
        tensor_x[entries_index, :context.shape[-1]] = context
        tensor_y[entries_index] = next_token
        entries_index += 1
    return tensor_x, tensor_y, entries_index

hash_documents = {}

def get_document_dataset(vocab: SimpleVocab, documents):
    assert type(documents) == list
    entries_count = 0
    for i in documents:
        entries_count += len(get_document_words(i))

    X = torch.full(size=(entries_count, SEQUENCE_LENGTH), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    y = torch.full(size=(entries_count, ), fill_value=vocab.vocab.PADDING_IDX, dtype=torch.long)
    entries_index = 0
    for document in documents:
        X, y, entries_index = encode_document_text(vocab, document, X, y, entries_index)
    assert not torch.all(X == 0), "All zeros is bad"
    assert not torch.all(y == 0), "All zeros is bad"
    return X, y

def get_model(vocab):
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=8,
        dropout=0.1,
        sequence_size=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
        transformer_layers=2,
        attention_heads=4,
        feed_forward=128,
    )
    model = GptTransformerModel(config)
    return model

def get_cached_model(vocab):
    vocab.lock()
    model = get_model(vocab)
    if os.path.isfile(CACHE_FILE):
        checkpoint = torch.load(CACHE_FILE, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

def train_loop(vocab, model):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))
    dataloader = ZmqDataloader()
    for _ in range(1024):
        print("Starting batch ")
        for document in dataloader:
            # TODO: add batch checkpoints to the training loop?
            X, y = get_document_dataset(vocab, [document])
            raw_dataloader = get_raw_dataloader((
                X.clone(),
                y.clone()
            ),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            _ = trainer.use_tqdm().train(raw_dataloader)
            torch.save({
                "model": model.state_dict(),
            }, CACHE_FILE)
    return model


if __name__ == "__main__":
    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    assert vocab is not None
    print("Loaded vocab")
    model = get_cached_model(vocab)
    print("Loaded model")
    model = train_loop(vocab, model)
    print(model)
