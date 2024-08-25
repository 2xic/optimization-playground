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
    def __init__(self, train, dataset_format="softmax", row_size=-1):
        assert dataset_format in ["softmax", "binary"]
        path = os.path.join(
            os.path.dirname(__file__),
            "dataset.json"
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
        for row in self.raw_dataset:
            if row_size == -1:
                self._add_dataset(row)
            else:
                for index in range(0, len(row) - row_size, row_size):
                    items = []
                    for row_index_size in range(0, row_size):
                        items.append(row[index + row_index_size])
                    self._add_dataset(items)
        print(f"Dataset rows : {len(self.dataset)}")

    def _add_dataset(self, items):
        self.dataset.append(items)
        if self.dataset_format == "binary":
            self.labels.append(torch.tensor([1.0]))
            # Create the reverse case
            self.dataset.append(items[::-1])
            self.labels.append(torch.tensor([0.0]))
        else:
            # Can create any n permutation ... WIll this actually help the model ? 
            self.labels.append(F.softmax(torch.arange(len(items), 0, -1).float(), dim=0))
            indices = torch.arange(len(items))
            permutations = itertools.permutations(indices)
            entries = []
            for index, row in enumerate(permutations):
                entries.append(row)

            for row in random.sample(entries, min(20, len(entries))):
                item_x = []
                item_y = []
                for index in row:
                    item_x.append(items[index])
                    item_y.append(len(items) - 1 - index)
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
