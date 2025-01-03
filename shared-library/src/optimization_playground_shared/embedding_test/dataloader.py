import glob
from torch.utils.data import Dataset
from ..nlp.DocumentEncoderSequence import SimpleVocab
import torch
import torch
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F

class TextDataloader(Dataset):
    def __init__(self, document_encoder: SimpleVocab, variant):
        self.IS_DEBUG_MODE = True

        self.SEQUENCE_LENGTH = 2048
        self.glob = sorted(glob.glob("/root/shared-library/shared-library/tensors/*"))
        # Limit documents if we are in debug mode to do research faster.
        if self.IS_DEBUG_MODE:
            print("Debug mode, skipping some files ... ")
            self.glob = self.glob[:1_000]
        
        self.document_encoder = document_encoder
        self.tensors = self.load_tensors()

        self.variant = variant
        self.batch_size = 32

    def load_tensors(self):
        tensors = []
        for i in self.glob:
            with open(i, "rb") as file:
                tensors.append(torch.load(file, weights_only=True, map_location=torch.device('cpu')))
                file.close()
        return tensors

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        tensor_next = self.tensors[(idx + 1) % len(self.tensors)]
        a = random.randint(0, max(tensor.shape[0] - self.SEQUENCE_LENGTH, self.SEQUENCE_LENGTH))
        b = random.randint(0, max(tensor_next.shape[0] - self.SEQUENCE_LENGTH, self.SEQUENCE_LENGTH))
        if self.variant == "next_token_prediction":
            x = self._make_size(tensor[a:a+self.SEQUENCE_LENGTH])
            y = self._make_size(tensor[a+self.SEQUENCE_LENGTH])
            return x, y
        elif self.variant == "triplet_loss":
            x =     self._make_size(tensor[a:a+self.SEQUENCE_LENGTH])
            x_pos = self._make_size(tensor[a+self.SEQUENCE_LENGTH:a+self.SEQUENCE_LENGTH*2])
            x_neg = self._make_size(tensor_next[b:b+self.SEQUENCE_LENGTH])
            return x, x_pos, x_neg
        else:
            x = self._make_size(tensor[:self.SEQUENCE_LENGTH])
            return x, torch.zeros((1))

    def _make_size(self, a: torch.Tensor):
        if a.shape[-1] == self.SEQUENCE_LENGTH:
            return a
        else:
            return F.pad(a, (0, self.SEQUENCE_LENGTH - a.shape[-1]), value=self.document_encoder.PADDING_IDX)

def get_dataloader(document_encoder: SimpleVocab, variant):
    dataset = TextDataloader(document_encoder, variant)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=8)
    return loader
