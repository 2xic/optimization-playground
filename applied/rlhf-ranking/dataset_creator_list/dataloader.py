"""
Let's use OpenAI embeddings for the backend for now.
"""
from torch.utils.data import Dataset
import json
import torch
import os
import random
import torch.nn.functional as F

class DocumentRankDataset(Dataset):
    def __init__(self, train, dataset_format="softmax", row_size=-1):
        assert dataset_format in ["softmax", "binary"]
        self.dataset = []
        path = os.path.join(
            os.path.dirname(__file__),
            "dataset.json"
        )
        with open(path, "r") as file:
            self.dataset = json.load(file)
        
        idx = len(self.dataset)
        start_offset = 0.8
        if train:
            self.dataset = self.dataset[:int(idx * start_offset)]
        else:
            self.dataset = self.dataset[int(idx * start_offset):]
        self.dataset_format = dataset_format
        # need to make sure the dataset is binary
        if self.dataset_format == "binary":
            row_size = 2
        if row_size != -1:
            binary_dataset = []
            for row in self.dataset:
                for index in range(0, len(row) - row_size, row_size):
                    items = []
                    for row_index_size in range(0, row_size):
                        items.append(row[index + row_index_size])
                    binary_dataset.append(items)
            self.dataset = binary_dataset

        print(f"Dataset rows : {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        results = []
        for v in self.dataset[idx]:
            results.append(torch.tensor(v))
        if self.dataset_format == "softmax":
            return *results, F.softmax(torch.arange(len(self.dataset[idx]), 0, -1).float(), dim=0)
        # Then do the swap
        assert len(results) == 2
        winner, looser = results
        label = torch.tensor([1.0])
        if random.randint(0, 1) == 1:
            winner, looser = looser, winner
            label = torch.tensor([0.0])
        return winner, looser, label
