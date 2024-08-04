"""
Terrible. Don't look at this.
"""

import sys
sys.path.append('../../playground/tiny-embedding-token-predictor')
from train import train as train_embed, get_document_dataset, get_model
# train = models
from dataset import get_dataset
import torch
import os
import tqdm 
cache = ".model_state.pkt"
model = None

def get_cached_model():
    global model
    model = get_model()
    if os.path.isfile(cache):
        checkpoint = torch.load(cache)
        model = get_model(checkpoint["config"])
        model.load_state_dict(checkpoint['model'])
    return model

def train_model():
    X, _ = get_dataset()
    X, y = get_document_dataset(X)
    model = get_cached_model()
    model = train_embed(X, y, model)
    torch.save({
        "model": model.state_dict(),
        "config": model.config
    }, cache)

def get_embed(model, document):
    X, _ = get_document_dataset([document])
    return model.embeddings(X)

class RandomModel:
    def __init__(self) -> None:
        pass

    def embeddings(self, _x):
        return torch.rand((1, 1024))

class EmbeddingWrapper:
    def __init__(self, trained=True) -> None:
        if trained == False:
            self.model = RandomModel()
        else:
            self.model = get_cached_model()

    # pre trained
    def train(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, i)
            output.append(out[0])
        return output

    def transforms(self, X):
        output = []
        for i in tqdm.tqdm(X):
            out = get_embed(self.model, i)
            output.append(out[0])
        return output

if __name__ == "__main__":
    train_model()
#    print(get_embed("I love bagels"))
#    print(get_embed("I love bagels"))
