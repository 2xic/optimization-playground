import torch.nn as nn
import torch
import torch.optim as optim
from torch import Tensor
from dataclasses import dataclass
from optimization_playground_shared.nlp.PositionalEncoding import PositionalEncoding
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate
import pickle
from dataset import get_dataset
import torch
import os
import tqdm 
import random

BATCH_SIZE = 1
SEQUENCE_LENGTH = 64
CACHE_FILE = ".model_state_gpt.pkt"

"""
GPT is a decoder only 
"""
@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    sequence_size: int
    dropout: float
    padding_index: int

class TinyModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(TinyModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim * config.sequence_size, config.vocab_size),
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout
        )
        self.sequence_size = config.sequence_size

    def forward(self, X: Tensor):
        assert len(X.shape) == 2
        source = self.embedding(X) + self.pos_encoder(X)
        transformer_out = source.reshape(X.shape[0], -1)
        return self.output(transformer_out)

    def forward_argmax(self, x):
        prediction = self.forward(x)
        return prediction.argmax(dim=1)
    
    def embeddings(self, X):
        values = torch.zeros((1, self.config.embedding_dim * self.config.sequence_size))
        for x in X:
            x = x.reshape((1, ) + X.shape[1:])
            assert len(X.shape) == 2
            with torch.no_grad():
                values += (self.embedding(x) + self.pos_encoder(x)).reshape(1, -1)
        return values

    def rollout(self, seed, steps, device=torch.device('cpu')):
        output = []
        for index in range(steps):
            next_predicted = None
            if (len(seed) - 1) < index:
                X = torch.zeros(1, self.sequence_size).reshape(1, -1).to(device).long().fill_(self.config.padding_index)
                copy = torch.tensor(output[-self.sequence_size:]).long()
                X[0, :copy.shape[0]] = copy

                X = self.forward(X)
                next_predicted = temperature_sampling(
                    X
                ).item()
                output.append(next_predicted)
            else:
                next_predicted = seed[index].item()
                output.append(next_predicted)
        return output

def create_vocab_dataset(documents):
    source = ".source_vocab_metadata"
    if not os.path.isfile(source):
        source_vocab = SimpleVocab()
        for i in documents:
            # We first need to figure out the vocab size. We can do random sampling later on to not have 
            # all the batches in memory.
            source_vocab.encode(i)
        with open(source, "wb") as file:
            pickle.dump(source_vocab, file)
    else:
        with open(source, "rb") as file:
            return pickle.load(file)

def encode_document_text(vocab, text):
    X = []
    y = []
    words = text.lower().replace(":", " : ").strip().split(" ")
    words = list(filter(lambda x: len(x), words))
    max_size = 64
    start_index = random.randint(0, max(len(words), len(words) - max_size)) if len(words) > 0 else 0
    for i in range(start_index, len(words) - 1):
        X.append(vocab.get_tensor(
            " ".join(words[max(i-SEQUENCE_LENGTH, 0):i]), sequence_length=SEQUENCE_LENGTH))
        y.append(vocab.get_tensor(
            " ".join(words[i:i+1]), sequence_length=1)[0])
        if len(X) > max_size:
            break
    if len(X) == 0:
        return vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH), vocab.get_tensor("", sequence_length=SEQUENCE_LENGTH)
    return torch.concat(X), torch.concat(y)

def get_document_dataset(vocab, document):
    assert type(document) == list
    text = "\n".join(document)
    X, y = encode_document_text(vocab, text)
    return X, y

def get_model(vocab):
    config = Config(
        vocab_size=vocab.size,
        embedding_dim=8,
        dropout=0.1,
        sequence_size=SEQUENCE_LENGTH,
        padding_index=vocab.vocab.PADDING_IDX,
    )
    model = TinyModel(config)
    return model

def get_cached_model(vocab):
    vocab.lock()
    model = get_model(vocab)
    if os.path.isfile(CACHE_FILE):
        checkpoint = torch.load(CACHE_FILE)
        model.load_state_dict(checkpoint['model'])
    return model

def train_loop(vocab, model, X_raw_documents):
    optimizer = optim.Adam(model.parameters(), lr=13e-4)
    trainer = TrainingLoopAccumulate(model, optimizer, loss=torch.nn.CrossEntropyLoss(ignore_index=vocab.vocab.PADDING_IDX))
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
    model = get_cached_model(vocab)
    model = train_loop(vocab, model, X)
    return model

def get_embed(model, vocab, document):
    X, _ = get_document_dataset(vocab, [document])
    return model.embeddings(X)

class RandomModel:
    def __init__(self) -> None:
        pass

    def embeddings(self, _x):
        return torch.rand((1, 1024))

class EmbeddingWrapper:
    def __init__(self, trained=True) -> None:
        X, _ = get_dataset()
        self.vocab = create_vocab_dataset(X)
        if trained == False:
            self.model = RandomModel()
        else:
            self.model = get_cached_model(self.vocab)

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
