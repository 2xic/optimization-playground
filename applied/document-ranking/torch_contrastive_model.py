import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
import torch
from optimization_playground_shared.nlp.PositionalEncoding import PositionalEncoding
from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import pickle
import os
from dataset import get_dataset
import torch.optim as optim
import random
import tqdm 

cache_path = ".model_contrastive_state.pkt"

BATCH_SIZE = 32

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

@dataclass
class ContrastiveConfig:
    vocab_size: int
    embedding_dim: int
    sequence_size: int
    dropout: float
    padding_index: int

class TinyDeltaModel(nn.Module):
    def __init__(self, config: ContrastiveConfig) -> None:
        super(TinyDeltaModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim * config.sequence_size, 512),
            nn.Sigmoid(),
            nn.Linear(512, 128),
            nn.Tanh(),
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout,
            max_len=config.sequence_size
        )
        self.sequence_size = config.sequence_size

    def forward(self, X: Tensor):
        assert len(X.shape) == 2
        source = self.embedding(X) + self.pos_encoder(X)
        embeddings = source.reshape(X.shape[0], -1)
        return self.output(embeddings)
   
    def embeddings(self, X):
        values = torch.zeros((1, self.config.embedding_dim * self.config.sequence_size))
        for x in X:
            x = x.reshape((1, ) + X.shape[1:])
            assert len(X.shape) == 2
            with torch.no_grad():
                values += (self.embedding(x) + self.pos_encoder(x)).reshape(1, -1)
        return values

# modified from https://gist.github.com/ShairozS/1a5e6953f0533cf19240ae1473eaedde
class ContrastiveLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(ContrastiveLoss, self).__init__()
        self.distance = lambda x, y: ((x - y) ** 2).sum(dim=1)
        self.margin = 0.3

    def forward(self, x, y, label):
        distance = self.distance(x, y)
        # When the label is 1 (similar) - the loss is the distance between the embeddings
        # When the label is 0 (dissimilar) - the loss is the distance between the embeddings
        loss_contrastive = torch.mean((label) * distance +
                                      (1-label) *  torch.clamp(self.margin - distance, min=0.0))

        return loss_contrastive        
    
    def pseudo_accuracy(self, x, y, label):
        distance = self.distance(x, y)
        delta = (distance < self.margin) * 1
        eq = (delta == label).sum()
        return eq / x.shape[0] * 100

def sample_document(source_vocab, documents, config):
    X = []
    y = []
    label = []
    for _ in range(BATCH_SIZE):
        index_a = random.randint(0, len(documents) - 1)
        index_b = random.randint(0, len(documents) - 1)

        # one items should at least be the same
        if random.randint(0, BATCH_SIZE) < BATCH_SIZE // 2:
            index_a = index_b
        
        encoded_a = source_vocab.encode(documents[index_a])
        encoded_b = source_vocab.encode(documents[index_b])

        document_a_index = random.randint(0, len(encoded_a) - 1)
        document_b_index = random.randint(0, len(encoded_b) - 1)

        if index_a == index_b:
            document_b_index = min(document_a_index - document_b_index, 200)

        tensor_a = source_vocab.get_tensor_from_tokens(
            encoded_a[document_a_index:document_a_index+config.sequence_size],
            config.sequence_size
        )
        tensor_b = source_vocab.get_tensor_from_tokens(
            encoded_b[document_b_index:document_b_index+config.sequence_size],
            config.sequence_size
        )
        X.append(tensor_a)
        y.append(tensor_b)
        label.append(torch.tensor([int(index_a == index_b)]))
    return (
        torch.concat(X),
        torch.concat(y),
        torch.concat(label)
    )

def get_contrastive_model(source_vocab):
    config_ = ContrastiveConfig(
        sequence_size=8192,
        embedding_dim=32,
        vocab_size=source_vocab.size,
        padding_index=source_vocab.vocab.PADDING_IDX,
        dropout=0.1,
    )
    model = TinyDeltaModel(config_)
    if os.path.isfile(cache_path):
        checkpoint = torch.load(cache_path)
        model.load_state_dict(checkpoint['model'])
    return model, config_

def train_model(source_vocab, model, config, documents):
    optimizer = optim.Adam(model.parameters())
    contrastive_loss = ContrastiveLoss()

    progress = tqdm.tqdm(range(1000 * (len(documents) // BATCH_SIZE)))
    sum_loss = None
    sum_accuracy = None
    index = 1
    for _ in progress:
        (x, y, z) = sample_document(source_vocab, documents, config)
        a = model(x)
        b = model(y)
        loss = contrastive_loss(a, b, z)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pseudo_accuracy = contrastive_loss.pseudo_accuracy(a, b, z)
        if sum_loss is None:
            sum_loss = loss
            sum_accuracy = pseudo_accuracy
        else:
            sum_loss += loss
            sum_accuracy += pseudo_accuracy
            index += 1
        progress.set_description(f"avg_loss: {(sum_loss/index)}, pseudo_accuracy: {sum_accuracy / index}")

    return model

def rollout_model(model, document, source_vocab: SimpleVocab, config: ContrastiveConfig):
    encoded_a = source_vocab.encode(document)
    output = torch.zeros((1, 128))
    for i in range(0, len(encoded_a), config.sequence_size):
        with torch.no_grad():
            input_embed = source_vocab.get_tensor_from_tokens(encoded_a[i:i+config.sequence_size], config.sequence_size)
            output += model(input_embed).reshape((1, -1))
    return output[0]

class ContrastiveEmbeddingWrapper:
    def __init__(self) -> None:
        # None as the model should already been trained
        vocab = create_vocab_dataset(None)
        model, config = get_contrastive_model(vocab)
        self.model = model
        self.config = config
        self.vocab = vocab
        self.model.eval()
        
    # pre trained
    def train(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = rollout_model(self.model, i, self.vocab, self.config)
            output.append(out)
        return output

    def transforms(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = rollout_model(self.model, i, self.vocab, self.config)
            output.append(out)
        return output


if __name__ == "__main__":
    X, _ = get_dataset()
    vocab = create_vocab_dataset(X)
    model, config = get_contrastive_model(vocab)

    model = train_model(vocab, model, config, X)
    # 
    torch.save({
        "model": model.state_dict(),
#        "config": model.config
    }, cache_path)


