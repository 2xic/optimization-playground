"""
If the embedding contain semantic information, we should be able to go form the embedding into a distribution
which mimics the token inputs

model(document) -> embedding
embedding -> decoder -> ~document word distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.dataset import BaseDataset
import matplotlib.pyplot as plt
from training.trainer import TrainingTimer


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

    def loss(self, X, _, raw_tokens):
        #   print(X.shape)
        #   print(self.model)
        y_prediction = self.model(X)

        model_distribution = y_prediction.log_softmax(dim=-1)
        word_counts = torch.bincount(raw_tokens, minlength=self.vocab_size)
        target_distribution = word_counts.float() / word_counts.sum()

        loss = F.kl_div(
            model_distribution,
            target_distribution,
            reduction="sum",
        )
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


def plot_loss_over_time(batches, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(batches, losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("plots/bytecode_dataset_small/autoencoder_test_loss.png")


def train(dataset: BaseDataset, embedding_model):
    embedding_size = embedding_model.embedding_size
    vocab_size = embedding_model.config.vocab_size

    model = Decoder(
        embedding_size,
        vocab_size,
    )
    sum_loss = 0
    batches = []
    loss_history = []

    timer = TrainingTimer(10)

    for index, i in enumerate(dataset.get_file_content_tokenized(32)):
        raw_tokens = i.reshape((1, -1))
        with torch.no_grad():
            embedding = embedding_model(raw_tokens)
            # Flatten the sequence embeddings into single dimension
            embedding = embedding.mean(dim=1)
        assert len(embedding.shape) == 2
        loss = model.loss(embedding, raw_tokens, i)
        loss.backward()
        sum_loss += loss.item()
        if index % 100 == 0 and index > 0:
            batches.append(index)
            loss_history.append(sum_loss)

            print(f"Index: {index}, loss: {sum_loss}")
            model.sample(embedding, i)
            model.optimizer.step()
            model.model.zero_grad()
            sum_loss = 0

        if timer.done():
            break
    plot_loss_over_time(batches, loss_history)
