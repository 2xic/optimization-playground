import torch
import torch.nn as nn

class EmbeddingsModel(torch.nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, 32, padding_idx=-1)
        prev_layer =  32
        layers = [
            vocab_size * 2,
            vocab_size * 4,
            vocab_size * 2,
            vocab_size,
        ]
        torch_layers = []
        for i in layers:
            torch_layers.append(
                nn.Linear(
                    prev_layer, i
                )
            )
            torch_layers.append(
                nn.Sigmoid()
            )
            prev_layer = i

        self.model = nn.Sequential(
            *torch_layers,
        )
    
    def forward(self, X):
        batch_size = X.shape[0]
        X = X.reshape((-1))
        emb = self.embedding_layer(X)
        emb = emb.reshape((batch_size, 4, 32))
        emb = emb.sum(dim = 1)
        out = self.model(emb)
        return out
    