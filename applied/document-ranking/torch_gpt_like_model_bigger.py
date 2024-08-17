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
from torch_shared_helpers import encode_document_embed_text, create_vocab_dataset, get_document_words
import time

os.makedirs(".cache", exist_ok=True)

BATCH_SIZE = 256
SEQUENCE_LENGTH = 128
CACHE_FILE = ".model_state_gpt_bigger.pkt"


# @jit()
def encode_document_text(vocab: SimpleVocab, text, tensor_x, tensor_y, entries_index):
    words = get_document_words(text)
    count_words = len(words)
    vocab_add = vocab.vocab.get 
    # preload all the words fast
    words = torch.tensor(list(map(lambda x: vocab_add(x), words)), dtype=torch.long)
 #   print(words)
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

def train_loop(vocab, model, X_raw_documents):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
    trainer = TrainingLoop(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))
    X, y = get_document_dataset(vocab, X_raw_documents)
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
            name=f'training_bigger_model.png'
        )
        
        torch.save({
            "model": model.state_dict(),
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
