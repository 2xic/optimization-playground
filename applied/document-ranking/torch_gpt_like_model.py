import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from dataclasses import dataclass
from optimization_playground_shared.nlp.PositionalEncoding import PositionalEncoding
from optimization_playground_shared.dataloaders.RawTensorToDataloader import get_dataloader as get_raw_dataloader
from optimization_playground_shared.training_loops.TrainingLoop import TrainingLoop
from dataset import get_dataset
import torch
import os
import tqdm 
from torch_shared_helpers import encode_document_embed_text, create_vocab_dataset, get_document_dataset

BATCH_SIZE = 256
SEQUENCE_LENGTH = 128
CACHE_FILE = ".model_state_gpt_lr.pkt"

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
        with torch.no_grad():
            values = torch.zeros((1, self.config.embedding_dim * self.config.sequence_size))
            for x in X:
                x = x.reshape((1, ) + X.shape[1:])
                assert len(X.shape) == 2
                values += (self.embedding(x) + self.pos_encoder(x)).reshape(1, -1)
        return values

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

def get_cached_model(vocab, cache_file):
    vocab.lock()
    model = get_model(vocab)
    if os.path.isfile(cache_file):
        checkpoint = torch.load(cache_file, map_location="cpu")
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
    for _ in range(1024):
        _ = trainer.use_tqdm().train(dataloader)
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

class EmbeddingWrapper:
    def __init__(self, trained=True) -> None:
        X, _ = get_dataset()
        self.vocab = create_vocab_dataset(X)
        self.trained = trained
        self.model = None
        self.cache_file = CACHE_FILE

    def load(self, new_cache_file):
        self.cache_file = new_cache_file
        return self
    
    # pre trained
    def train(self, X):
        if self.trained == False:
            self.model = RandomModel()
        else:
            self.model = get_cached_model(self.vocab, self.cache_file).eval()

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
