from datasets import inspect_dataset, load_dataset_builder
import dataset
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import FSMTTokenizer

class Wmt16Dataloader(Dataset):
    def __init__(self, train):
        builder = load_dataset_builder(
            "wmt16",
            "de-en",
        )
        mname = "allenai/wmt16-en-de-dist-12-1"
        self.tokenizer = FSMTTokenizer.from_pretrained(mname)

        # Standard version
        builder.download_and_prepare()
        self.ds = builder.as_dataset()
        self.data = self.ds["train"] if train else self.ds["test"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]["translation"]
        encoded_de, encoded_en = self.tokenizer.encode(data["de"]), self.tokenizer.encode(data["en"])
        return encoded_en, encoded_de

train_dataloader = DataLoader(Wmt16Dataloader(train=True), batch_size=1, shuffle=True)
test_dataloader = DataLoader(Wmt16Dataloader(train=False), batch_size=1, shuffle=True)
