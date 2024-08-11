import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
#from optimization_playground_shared.training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from dataset import get_dataset
import torch
import os
import tqdm 
import random
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
import hashlib
from torch_shared_helpers import encode_document_embed_text, create_vocab_dataset, get_document_words

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 256
SEQUENCE_LENGTH = 128
CACHE_FILE = ".model_state_gpt_bigger.pkt"


def encode_document_text(vocab, text):
    X = []
    y = []
    words = get_document_words(text)
    start_index = 0
    for i in range(start_index, len(words) - 1):
        X.append(vocab.get_tensor(
            " ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(vocab.get_tensor(
            " ".join(words[i:i+1]), sequence_length=1)[0])
        #if len(X) > max_batch_size:
        #    break
    return X, y

hash_documents = {}

def get_document_dataset(vocab, documents):
    assert type(documents) == list
    X = []
    y = []
    for text in tqdm.tqdm(documents, desc="Encoding documents ... "):
        hash_id = hashlib.sha256(text.encode()).hexdigest()
        if hash_id in hash_documents:
            (X_mini, y_mini) = hash_documents[hash_id]
            X += X_mini
            y += y_mini
        else:
            X_mini, y_mini = encode_document_text(vocab, text)
            X += X_mini
            y += y_mini
            hash_documents[hash_id] = [X_mini, y_mini]
            break
    if len(X) == 0:
        return vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH), vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH)
    return torch.concat(X), torch.concat(y)

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

def train_loop(vocab, model, X_raw_documents):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))
    if torch.cuda.is_available():
        for _ in range(1024):
            X, y = get_document_dataset(vocab, X_raw_documents)
            dataloader = get_raw_dataloader((
                X.clone(),
                y.clone()
            ),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            _ = trainer.use_tqdm().train(dataloader)
            """
            metrics_tracker.log(
                Metrics(
                    epoch=epoch,
                    loss=sum_loss,
                    training_accuracy=acc,
                    prediction=None
                )
            )
            """
            torch.save({
                "model": model.state_dict(),
            }, CACHE_FILE)
    else:
        #  trainer.device = torch.device("cpu") # easier to debug with cpu
        for _  in range(32):
            document = X_raw_documents[random.randint(0, len(X_raw_documents) - 1)]
            X, y = get_document_dataset(vocab, [document])
            dataloader = get_raw_dataloader((
                X.clone(),
                y.clone()
            ),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            trainer.use_tqdm().train(dataloader)
            torch.save({
                "model": model.state_dict(),
        #     "config": model.config
            }, CACHE_FILE)
    return model

def train_model(vocab, X):
    assert vocab is not None
    model = get_cached_model(vocab)
    model = train_loop(vocab, model, X)
    return model

def get_embed(model, vocab, document):
    X = encode_document_embed_text(vocab,document, sequence_length=SEQUENCE_LENGTH)
    return model.embeddings(X)

class RandomModel:
    def __init__(self) -> None:
        pass

    def embeddings(self, _x):
        return torch.rand((1, 1024))

class EmbeddingWrapperBigger:
    def __init__(self, trained=True) -> None:
        X, _ = get_dataset()
        self.vocab = create_vocab_dataset(X)
        if trained == False:
            self.model = RandomModel()
        else:
            self.model = get_cached_model(self.vocab).eval()

    # pre trained
    def train(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, self.vocab, i)
            output.append(out[0])
        return output

    def transforms(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, self.vocab, i)
            output.append(out[0])
        return output

if __name__ == "__main__":
    X, _ = get_dataset()
    vocab = create_vocab_dataset(X)
    train_model(vocab, X)
    print("Done?")
