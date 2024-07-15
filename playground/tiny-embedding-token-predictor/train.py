from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from tiny_model import TinyModel, Config

SEQUENCE_LENGTH = 128
batch_size = 32
source_vocab = SimpleVocab()
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        if 32 * 2048 < len(X):
            break
    return torch.concat(X), torch.concat(y)

def get_dataset():
    text = None
    with open("tinyshakespeare.txt", "r") as file:
        text = file.read()

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

def train():
    X, y = get_dataset()
    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=batch_size,
        shuffle=False,
    )

    config = Config(
        vocab_size=source_vocab.size,
        embedding_dim=32,
        dropout=0.1,
        sequence_size=SEQUENCE_LENGTH,
        padding_index=source_vocab.vocab.PADDING_IDX,
    )
    model = TinyModel(config)
    optimizer = optim.Adam(model.parameters())
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=source_vocab.vocab.PADDING_IDX))
    (loss, acc) = trainer.use_tqdm().train(dataloader)
    print((loss, acc))
    print(predict(model))

if __name__ == "__main__":
    train()
