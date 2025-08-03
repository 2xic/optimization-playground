"""
If the embedding contain semantic information, we should be able to go form the embedding into a distribution
which mimics the token inputs

model(document) -> embedding
embedding -> decoder -> ~document word distribution
"""

from abc import ABC, abstractmethod
from typing import Generator, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    simple_temperature_sampling,
)


class AbstractDataset(ABC):
    @abstractmethod
    def get_file_content_tokenized(self) -> List[int]:
        pass


class Decoder:
    def __init__(self, embedding_size, vocab_size):
        self.model = nn.Sequential(
            *(
                nn.Linear(embedding_size, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, vocab_size),
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.vocab_size = vocab_size

    def loss(self, X, y, raw_tokens):
        y_prediction = self.model(X)

        # Aggregate model predictions across the batch
        model_distribution = y_prediction.log_softmax(dim=-1)

        # Target word frequency distribution
        word_counts = torch.bincount(raw_tokens, minlength=self.vocab_size)
        target_distribution = word_counts.float() / word_counts.sum()

        loss = F.kl_div(
            model_distribution,
            target_distribution,
            reduction="sum",
        )
        #      print(target_distribution)
        #       print(model_distribution)
        #        print(loss)
        return loss

    def sample(self, embedding, raw_tokens):
        with torch.no_grad():
            predicted = self.model(embedding).softmax(dim=-1)
            word_counts = torch.bincount(raw_tokens, minlength=self.vocab_size)
            target_distribution = word_counts.float() / word_counts.sum()

            print("Expected: ")
            for i in raw_tokens:
                print(target_distribution[i].item(), end=", ")
            print("")
            print("Inferred: ")
            for i in raw_tokens:
                print(predicted[0][i].item(), end=", ")
            print("")
            print("")


def train(dataset: AbstractDataset, embedding_model):
    output_dimension = embedding_model.output_layer
    vocab_size = output_dimension.out_features

    # currently just training on the output, but this should change.
    model = Decoder(
        vocab_size,
        vocab_size,
    )
    sum_loss = 0
    for index, i in enumerate(dataset.get_file_content_tokenized(32)):
        raw_tokens = i.reshape((1, -1))
        with torch.no_grad():
            embedding = embedding_model(raw_tokens).mean(dim=1)
        assert len(embedding.shape) == 2
        loss = model.loss(embedding, raw_tokens, i)
        loss.backward()
        sum_loss += loss.item()
        if index % 100 == 0 and index > 0:
            print(f"Index: {index}, loss: {sum_loss}")
            model.sample(embedding, i)
            model.optimizer.step()
            model.model.zero_grad()
            sum_loss = 0
