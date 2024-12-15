"""
Let's use OpenAI embeddings for the backend for now.
"""
from torch.utils.data import Dataset
import json
import torch
import os
import torch.nn.functional as F
import itertools
import random

class DocumentRankDataset(Dataset):
    def __init__(self, train, embedding_backend="openai", dataset_format="softmax", row_size=-1, create_synthetic_rows=True):
        assert dataset_format in ["softmax", "binary"]
        self.create_synthetic_rows = create_synthetic_rows
        path = os.path.join(
            os.path.dirname(__file__),
            f"dataset_{embedding_backend}.json"
        )
        with open(path, "r") as file:
            self.raw_dataset = json.load(file)
        idx = len(self.raw_dataset)
        start_offset = 0.8
        if train:
            self.raw_dataset = self.raw_dataset[:int(idx * start_offset)]
        else:
            self.raw_dataset = self.raw_dataset[int(idx * start_offset):]
        self.dataset_format = dataset_format
        # need to make sure the dataset is binary
        if self.dataset_format == "binary":
            row_size = 2
        # Update the dataset and create it with permutations
        self.dataset = []
        self.labels = []
        self.embedding_size = -1
        for row in self.raw_dataset:
            embeddings = row["embeddings"]
            print(row["scores"])
            scores = list(map(float, row["scores"]))

            if self.embedding_size == -1:
                self.embedding_size = len(embeddings[0])
            else:
                assert self.embedding_size == len(embeddings[0]), f"{self.embedding_size} != {len(embeddings)}"

            if row_size == -1:
                self._add_dataset(embeddings, scores)
                row_size = len(embeddings)
            else:
                for index in range(0, len(embeddings) - row_size, row_size):
                    items = []
                    n_scores = []
                    for row_index_size in range(0, row_size):
                        items.append(embeddings[index + row_index_size])
                        n_scores.append(scores[index + row_index_size])
                    # if all the scores are zero, then we don't have any signal
                    if sum(n_scores) == 0:
                        continue
                    self._add_dataset(items, n_scores)
        print(f"Raw dataset rows : {len(self.raw_dataset)}")
        print(f"Dataset rows : {len(self.dataset)}")
        self.row_size = row_size

    def _add_dataset(self, items, scores):
        self.dataset.append(items)
        if self.dataset_format == "binary":
            # Binary labels are better than floating ones actually.
            self.labels.append(torch.tensor([1.0])) # scores[0]]) / max(scores))
            # Create the reverse case
            self.dataset.append(items[::-1])
            self.labels.append(torch.tensor([0.0])) # scores[-1]]) / max(scores))
        else:
            self.labels.append(F.softmax(torch.tensor(scores).float(), dim=0))            
            # Can create any n permutation ... Will this actually help the model ?
            if self.create_synthetic_rows:
                indices = torch.arange(len(items))
                permutations = itertools.permutations(indices)
                entries = []
                for index, row in enumerate(permutations):
                    entries.append(row)
                # Generate a random samples
                for row in random.sample(entries, min(20, len(entries))):
                    item_x = []
                    item_y = []
                    for index in row:
                        item_x.append(items[index])
                        item_y.append(scores[index])
                    self.dataset.append(item_x)
                    self.labels.append(F.softmax(torch.tensor(item_y).float(), dim=0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        results = []
        for v in self.dataset[idx]:
            results.append(torch.tensor(v))
        if self.dataset_format == "softmax":
            return *results, self.labels[idx]       
        # Then do the swap
        assert len(results) == 2
        winner, looser = results
        label = self.labels[idx]
        return winner, looser, label
