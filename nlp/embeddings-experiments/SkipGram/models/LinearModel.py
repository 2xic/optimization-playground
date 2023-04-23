import torch
import torch.nn as nn

class LinearModel(torch.nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(*[
            nn.Linear(vocab_size, 32),
            nn.Sigmoid(),
        ])
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
        self.vocab_size = vocab_size
        
    def forward(self, X):
        X = torch.nn.functional.one_hot(X, num_classes=self.vocab_size).float()
        emb = self.embedding_layer(X)
        return self.model(emb.reshape(X.shape[0], -1))
