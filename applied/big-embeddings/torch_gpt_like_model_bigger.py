import os
import torch
import torch.optim as optim
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from optimization_playground_shared.dataloaders.data_portal.Client import ZmqDataloader
from optimization_playground_shared.utils.General import save_model_atomic
from optimization_playground_shared.metrics_tracker.producer import Tracker, Metrics
import optimization_playground_shared
#from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
from optimization_playground_shared.nlp.wordpiece.bpeDocumentDecoder import get_document_dataset
from optimization_playground_shared.dataloaders.Utils import find_batch_size
import atexit
import time
import pickle

BATCH_SIZE = 1
# Need to try to tain model with long sequence size ....
SEQUENCE_LENGTH = 128
CACHE_FILE = ".model_state.pkt"
DOCUMENTS_TO_LOAD_PER_BATCH = 4
EPOCHS = 1024


def get_cache_file() -> BPE:
    source = "fineweb/bpe"
    if os.path.isfile(source):
        with open(source, "rb") as file:
            return pickle.load(file)
    return None

def get_model(vocab: BPE):
    vocab.lock()
    embedding_dim = 32
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=embedding_dim,
        dropout=0,
        sequence_length=SEQUENCE_LENGTH,
        padding_index=vocab.index.padding_idx,
        transformer_layers=8,
        attention_heads=4,
        feed_forward=embedding_dim * 4,
    )
    model = GptTransformerModel(config)
    return model

def flatten_view(x, y):
    return x, y.view(-1)

def _accuracy_check(predicted, y, vocab: BPE):
    predicted = torch.argmax(
        predicted,
        dim=1
    )
    indices = torch.where(y != vocab.index.padding_idx)
    y_filtered = y[indices]
    accuracy = (y[indices] == predicted[indices]).sum()

    return accuracy, y_filtered.shape[0]

def train_loop(vocab: BPE):
    value = torch.load("fineweb/tensors.pt")
    docs = [
        vocab.decode(i)
        for i in value
    ][:2]
    X, y = get_document_dataset(vocab, docs, SEQUENCE_LENGTH=SEQUENCE_LENGTH)
    raw_dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=1,
        shuffle=True,
    )
    model = get_model(vocab)
    print("Loaded model")

    optimizer = optim.Adam(
        model.parameters(),
        lr=13e-4
    )
    trainer = TrainingLoop(
        model, 
        optimizer, 
        loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.index.padding_idx),
        callback=flatten_view,
    )
    raw_dataloader = find_batch_size(trainer, raw_dataloader, device=torch.device('cuda:0'), max_size=64)

    trainer._accuracy_check = lambda x, y: _accuracy_check(x, y, vocab)

    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
    print("{:.3f}MB allocated".format(torch.cuda.memory_allocated()/1024**2))
    print("")
    for _ in range(EPOCHS):
        (_, _) = trainer.use_tqdm().train(
            raw_dataloader,
            callback=flatten_view,
        )
        name = docs[0].split(" ")[0]
        print(name, model.rollout_output(vocab.encode(name), 10, "temperature"))
        print(vocab.decode(model.rollout_output(vocab.encode(name), 10, "temperature")))
    return model


if __name__ == "__main__":
    vocab = get_cache_file()
    assert vocab is not None
    print("Loaded vocab")
    model = train_loop(vocab)
    print(model)
