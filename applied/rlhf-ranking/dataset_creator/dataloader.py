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
        winner = self.dataset[idx][0]
        looser = self.dataset[idx][1]
        label = 1
        # do a swap
        if random.randint(0, 1) == 1:
            winner, looser = looser, winner
            label = 0
        return torch.tensor(winner), torch.tensor(looser), torch.tensor([label]).float()

class DocumentListDataset(DocumentRankDataset):
    def __init__(self, train):
        super().__init__(train)

    def __getitem__(self, idx):
        winner = self.dataset[idx][0]
        looser = self.dataset[idx][1]
        label = torch.tensor([2.0, 1.0])
        # do a swap
        if random.randint(0, 1) == 1:
            winner, looser = looser, winner
            label = torch.tensor([1.0, 2.0])
        return torch.tensor(winner), torch.tensor(looser), F.softmax(label, dim=0)

