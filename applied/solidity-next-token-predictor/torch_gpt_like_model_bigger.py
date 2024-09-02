import os
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from optimization_playground_shared.nlp.SimpleVocab import splitter
from pre_generator import get_cache_file
import atexit
import time
from optimization_playground_shared.utils.General import save_model_atomic, load_model
from torch.cuda.amp import autocast

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 4096
# Need to try to tain model with long sequence size ....
SEQUENCE_LENGTH = 256
CACHE_FILE = ".model_state.pkt"

def get_document_words(text):
    # todo: create a good tokenizer, this does not really work for code tokens
    if type(text) == bytes:
        return splitter(text.decode())
    else:
        return splitter(text)

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
  #  print(torch.bincount(y).tolist())#, y.unique(sorted=True).float()))
    assert not torch.all(X == 0), "All zeros is bad"
    assert not torch.all(y == 0), "All zeros is bad"
    return X, y

def get_model(vocab):
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=8,
        dropout=0.1,
        sequence_length=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
        transformer_layers=8,
        attention_heads=4,
        feed_forward=256,
    )
    model = GptTransformerModel(config)
    return model

def get_cached_model(vocab):
    vocab.lock()
    model = get_model(vocab)
    if os.path.isfile(CACHE_FILE):
        checkpoint = load_model(CACHE_FILE)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def train_loop(vocab: SimpleVocab, model: GptTransformerModel):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
#    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss())#ignore_index=vocab.vocab.PADDING_IDX))
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss())#ignore_index=vocab.vocab.PADDING_IDX))

    dataloader = ZmqDataloader()
    iterator = iter(dataloader)
    last_save = 0

    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
    model.embedding.cuda(0)
    model.pos_encoder.cuda(0)
    model.transformer_decoder.cuda(1)
    # Output should match device of the dataloader ... 
    model.output.cuda(2)
    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
    print("")

    for _ in range(1024):
        X, y = get_document_dataset(vocab, [
            next(iterator) for _ in range(10)
        ])
        raw_dataloader = get_raw_dataloader((
            X.clone(),
            y.clone()
        ),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        with autocast():
            _ = trainer.use_tqdm().train(raw_dataloader, sharding=True)
        # Try to predict something 
        print(vocab.decode(
            model.rollout(
                vocab.get_tensor("contract ", sequence_length=-1)[0],
                steps=100
            )
        ))
        print(vocab.decode(
            model.rollout(
                vocab.get_tensor("contract Uniswap {", sequence_length=-1)[0],
                steps=100
            )
        ))
        print(vocab.decode(
            model.rollout(
                vocab.get_tensor("uint8 ", sequence_length=-1)[0],
                steps=100
            )
        ))

        if (time.time() - last_save) > 60:
            save_model_atomic(CACHE_FILE, model)
            last_save = time.time()
    return model

def save_model():
    print("STARTING TO SAVE MODEL")
#    save_model_atomic(CACHE_FILE, model)
    print("DONE SAVING MODEL")

atexit.register(save_model)

if __name__ == "__main__":
    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    assert vocab is not None
    print("Loaded vocab")
    model = get_cached_model(vocab)
    print("Loaded model")
    model = train_loop(vocab, model)
    print(model)
