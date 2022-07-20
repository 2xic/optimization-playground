from datasets import load_dataset_builder
import dataset
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import FSMTTokenizer

class CustomTokenizer: 
    def __init__(self) -> None:
        self.words = {

        }
        self.idx_to_words = {

        }
        self.DEBUG_MODE = True
        self.LIMIT = 150

        self.beginning_token = self.add("<s>")
        self.end_token = self.add("</s>")
        self.UNKNOWN = self.add("<unknown>")

    def add(self, word):
        idx = len(self.words)
        if idx < self.LIMIT and word not in self.words:
            self.words[word] = idx
            self.idx_to_words[idx] = word
            return idx
        else:
            return self.words.get(word, self.UNKNOWN)
        
    def encode(self, words):
        idx = []
        for i in words.split(" "):
            idx.append(self.add(i))
        idx.append(self.end_token)
        return idx

    def decode(self, idx):
        words = []
        for i in idx:
            words.append(self.idx_to_words.get(i, None))
        return words

    @property
    def vocab_size(self):
        if self.DEBUG_MODE:
            # we add words dynamically.
            return self.LIMIT
        return len(self.words)


class Wmt16Dataloader(Dataset):
    def __init__(self, train=True, try_to_overfit=-1):
        builder = load_dataset_builder(
            "wmt16",
            "de-en",
        )
        mname = "allenai/wmt16-en-de-dist-12-1"
#        self.tokenizer: FSMTTokenizer = FSMTTokenizer.from_pretrained(mname)
        self.tokenizer = CustomTokenizer()
        self.beginning_token = self.tokenizer.beginning_token
        self.end_token = self.tokenizer.end_token
        self.vocab_size = self.tokenizer.vocab_size + 1

        # Standard version
        builder.download_and_prepare()
        self.ds = builder.as_dataset()
        self.data = self.ds["train"] if train else self.ds["test"]
        self.try_to_overfit = try_to_overfit != -1
        self.try_to_fit_size = try_to_overfit

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def __len__(self):
        if self.try_to_overfit:
            return self.try_to_fit_size
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]["translation"]
        encoded_de, encoded_en = self.tokenizer.encode(data["de"]), self.tokenizer.encode(data["en"])
        return encoded_en, encoded_de

def get_data_loader(try_to_overfit):
    train = Wmt16Dataloader(train=True, try_to_overfit=try_to_overfit)
    train_dataloader = DataLoader(train, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(Wmt16Dataloader(train=False, try_to_overfit=try_to_overfit), batch_size=1, shuffle=True)
    return (
        train_dataloader,
        test_dataloader,
        train
    )