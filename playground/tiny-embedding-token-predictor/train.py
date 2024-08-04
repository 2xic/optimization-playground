from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from tiny_model import TinyModel, Config

SEQUENCE_LENGTH = 32
batch_size = 4
source_vocab = SimpleVocab()

print("Starting .... ")

def sliding_vocab(text):
    X = []
    y = []

    words = text.lower().replace(":", " : ").strip().split(" ")
    words = list(filter(lambda x: len(x), words))
    for i in range(1, len(words) - 1):
        X.append(source_vocab.get_tensor(
            " ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(source_vocab.get_tensor(
            " ".join(words[i:i+1]), sequence_length=1)[0])
    if len(X) == 0:
        return source_vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH), source_vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH)
    return torch.concat(X), torch.concat(y)

def encode_document(text):
    X = []
    words = text.lower().replace(":", " : ").strip().split(" ")
    words = list(filter(lambda x: len(x), words))
    for i in range(1, len(words)):
        X.append(source_vocab.get_tensor(
            " ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
    return torch.concat(X)

def get_dataset():
    text = None
    with open("tinyshakespeare.txt", "r") as file:
        text = file.read()
    X, y = sliding_vocab(text)
    source_vocab.lock()
    return X, y

def get_document_dataset(document):
    text = "\n".join(document)
    X, y = sliding_vocab(text)
    source_vocab.lock()
    return X, y

def predict(model):
    results = []
    seed = source_vocab.get_tensor(
        "second citizen : ", sequence_length=-1).reshape(-1)

    with torch.no_grad():
        y = model.rollout(
            seed=seed,
            steps=512,
        )
        for i in y:
            results.append(source_vocab.vocab.index_vocab[i])
        return " ".join(results)

def get_model(config=None):
    if config is None:
        config = Config(
            vocab_size=source_vocab.size,
            embedding_dim=32,
            dropout=0.1,
            sequence_size=SEQUENCE_LENGTH,
            padding_index=source_vocab.vocab.PADDING_IDX,
        )
    model = TinyModel(config)
    return model

def train(X, y, model=None):
    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=False,
    )
    if model is None:
        model = get_model()
    optimizer = optim.Adam(model.parameters())
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX))
    (loss, acc) = trainer.use_tqdm().train(dataloader)
    print((loss, acc))
    return model

if __name__ == "__main__":
    X, y = get_dataset()
    train(X, y)
