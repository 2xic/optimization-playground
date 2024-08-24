"""
Let's use OpenAI embeddings for the backend for now.
"""
from torch.utils.data import Dataset
import json
import random
import torch
import os
import torch.nn.functional as F

class DocumentRankDataset(Dataset):
    def __init__(self, train):
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

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        results = []

        for v in self.dataset[idx]:
            results.append(torch.tensor(v))

        return *results, torch.arange(len(self.dataset[idx]), 0, -1).float()
