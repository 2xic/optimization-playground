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
        self.IS_DEBUG_MODE = False

        self.SEQUENCE_LENGTH = 256
        self.glob = sorted(glob.glob("/root/tensors_processed/*"))
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
                data = torch.load(file, weights_only=True, map_location=torch.device('cpu'))
                if data.shape[-1] > 0:
                    tensors.append(data)
                else:
                    print("Skipped ... ")
        return tensors

    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        tensor_next = self.tensors[(idx + 1) % len(self.tensors)]
        index = random.randint(0, tensor.shape[0] - 1)
        if self.variant == "next_token_prediction":
            b = random.randint(0, tensor.shape[1] - 1)
            x = self._make_size(tensor[index][b:])
            y = self._make_size(tensor[index][b+1:])
            return x, y
        elif self.variant == "triplet_loss":
            index_neg = random.randint(0, tensor_next.shape[0] - 1)
            x =     self._make_size(tensor[index])
            x_pos = self._make_size(tensor[(index+1) % tensor.shape[0]])
            x_neg = self._make_size(tensor_next[index_neg])
            #print(x)
            #print(x_pos)
            #print(x_neg)
            #print()
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    return loader, dataset
