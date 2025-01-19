import glob
from torch.utils.data import Dataset
from ..nlp.DocumentEncoderSequence import SimpleVocab
import torch
import torch
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
import numpy as np

class TextDataloader(Dataset):
    def __init__(self, document_encoder: SimpleVocab, variant):
        self.IS_DEBUG_MODE = False

        self.SEQUENCE_LENGTH = 256
        self.glob = sorted(glob.glob("/root/bpe/tensors/*"))
        # Limit documents if we are in debug mode to do research faster.
        if self.IS_DEBUG_MODE:
            print("Debug mode, skipping some files ... ")
            self.glob = self.glob[:1_000]
        
        self.document_encoder = document_encoder
        self.tensors = [None, ] * len(self.glob)

        self.variant = variant

    def __len__(self):
        return len(self.glob)
    
    def get_tensor(self, idx):
        if self.tensors[idx] == None:
            with open(self.glob[idx], "rb") as file:
                self.tensors[idx] = torch.frombuffer(bytearray(file.read()), dtype=torch.long)
        return self.tensors[idx]

    def __getitem__(self, idx):
        tensor = self.get_tensor(idx)
        tensor_next = self.tensors[(idx + 1) % len(self.tensors)]
        index = random.randint(0, tensor.shape[0] - 1)
        if self.variant == "next_token_prediction":
            delta = self.SEQUENCE_LENGTH - (tensor.shape[-1] % self.SEQUENCE_LENGTH) if self.SEQUENCE_LENGTH < tensor.shape[-1] else self.SEQUENCE_LENGTH - tensor.shape[-1]
            new_size = tensor.shape[-1] + delta
            if new_size < (self.SEQUENCE_LENGTH * 2):
                delta += self.SEQUENCE_LENGTH
            resized = F.pad(tensor, (0, delta), value=self.document_encoder.PADDING_IDX).long()
            i = random.randint(0, resized.shape[0] - 1 - self.SEQUENCE_LENGTH)
            return resized[i:i+self.SEQUENCE_LENGTH], resized[i+1:i+1+self.SEQUENCE_LENGTH]
        elif self.variant == "triplet_loss":
            index_neg = random.randint(0, tensor_next.shape[0] - 1)
            x =     self._make_size(tensor[index])
            x_pos = self._make_size(tensor[(index+1) % tensor.shape[0]])
            x_neg = self._make_size(tensor_next[index_neg])
            return x, x_pos, x_neg
        else:
            x = self._make_size(tensor[index])
            return x, torch.zeros((1))

    def _make_size(self, a: torch.Tensor):
        if a.shape[-1] == self.SEQUENCE_LENGTH:
            return a.long()
        else:
            return F.pad(a, (0, self.SEQUENCE_LENGTH - a.shape[-1]), value=self.document_encoder.PADDING_IDX).long()

def get_dataloader(document_encoder: SimpleVocab, variant, batch_size=256):
    dataset = TextDataloader(document_encoder, variant)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    return loader, dataset
