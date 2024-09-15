from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import torch
import torch.optim as optim
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from optimization_playground_shared.plot.Plot import Plot, Figure
from dataset import get_dataset
import torch
import os
import tqdm 
import random
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from torch_shared_helpers import encode_document_embed_text, create_vocab_dataset
from optimization_playground_shared.nlp.DocumentEncoder import get_document_dataset
import time

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 256
SEQUENCE_LENGTH = 128
CACHE_FILE = ".model_state_gpt_bigger_lr.pkt"


def get_model(vocab):
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=8,
        dropout=0.1,
        sequence_length=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
        transformer_layers=2,
        attention_heads=4,
        feed_forward=128,
    )
    model = GptTransformerModel(config)
    return model

def get_cached_model(vocab, cache_file):
    vocab.lock()
    model = get_model(vocab)
    if os.path.isfile(cache_file):
        checkpoint = torch.load(cache_file, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

def train_loop(vocab, model, X_raw_documents):
    optimizer = optim.Adam(model.parameters())
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))
    X, y = get_document_dataset(vocab, X_raw_documents, SEQUENCE_LENGTH)
    dataloader = get_raw_dataloader((
        X.clone(),
        y.clone()
    ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    training_accuracy = []
    training_loss = []
    for _ in range(1024):
        (loss, accuracy) = trainer.use_tqdm().train(dataloader)
        training_accuracy.append(accuracy.item())
        training_loss.append(loss.item())
        plot = Plot()
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Loss": training_loss,
                    },
                    title="Training loss",
                    x_axes_text="Epochs",
                    y_axes_text="Loss",
                ),
                Figure(
                    plots={
                        "Training accuracy": training_accuracy,
                    },
                    title="Accuracy",
                    x_axes_text="Epochs",
                    y_axes_text="accuracy",
                ),
            ],
            name=f'training_bigger_model_lr.png'
        )
        
        torch.save({
            "model": model.state_dict(),
        }, CACHE_FILE)
    return model

def train_model(vocab, X):
    assert vocab is not None
    model = get_cached_model(vocab, CACHE_FILE)
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
        self.trained = trained
        self.model = None
        self.cache_file = CACHE_FILE

    def load(self, new_cache_file):
        self.cache_file = new_cache_file
        return self

    def _load(self):
        if self.trained == False:
            self.model = RandomModel()
        else:
            self.model = get_cached_model(self.vocab, cache_file=self.cache_file).eval()

    # pre trained
    def train(self, X):
        if self.model is None:
            self._load()
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, self.vocab, i)
            output.append(out[0])
        return output

    def transforms(self, X):
        if self.model is None:
            self._load()
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, self.vocab, i)
            output.append(out[0])
        return output

if __name__ == "__main__":
    benchmark = False
    X, _ = get_dataset()
    vocab = create_vocab_dataset(X)

    if benchmark:
        total = 0
        for index in range(100):
            random_index = random.randint(0, len(X) - 1)
            document = X[random_index]
            start = time.time()
            _ = get_document_dataset(vocab, [document])
            end = time.time()
            print((f"({index}) document {random_index}", end - start))
            print("")
            total += (end - start)
        print(f"Total {total}")
    else:
        train_model(vocab, X)
        print("Done?")
