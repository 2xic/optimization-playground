import asyncio
import aiofiles
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ..nlp.DocumentEncoderSequence import SimpleVocab
from ..nlp.DocumentEncoder import get_document_dataset
from ..nlp.DocumentEncoderSequence import get_document_dataset as get_document_dataset_sequence
from ..utils.asyncio_utils import gather_batch
import torch
import random

class TextDataloader(Dataset):
    def __init__(self, document_encoder: SimpleVocab, variant):
        self.SEQUENCE_LENGTH = 2048
        self.glob = sorted(glob.glob("/root/text_document_dataset/*"))
        self.document_encoder = document_encoder
        self.variant = variant
        self.docs = asyncio.run(self.load_documents(document_encoder))

    async def load_documents(self, document_encoder: SimpleVocab):
        docs = []
        async for i in gather_batch(
            self.glob,
            lambda x: self.read_file(x),
            batch_size=128,
        ):
            document_encoder.fit([i])
            docs.append(i)
        document_encoder.lock()
        return docs

    async def read_file(self, filename):
        async with aiofiles.open(filename, mode='r') as f:
            contents = await f.read()
            return contents

    def _add_document(self, file):
        with open(file, "r") as file:
            return file.read()
        
    def __iter__(self):
        documents_index = list(range(len(self.docs)))
        random.shuffle(documents_index)
        for idx, index in enumerate(documents_index):
            document = self.docs[index]
            if self.variant == "next_token_prediction":
                batch_x, batch_y = get_document_dataset(self.document_encoder, [document], self.SEQUENCE_LENGTH)
                for i in range(0, batch_x.shape[0], 32):
                    yield batch_x[i:i+32], batch_y[i:i+32]
            elif self.variant == "triplet_loss":
                batch_x = get_document_dataset_sequence(self.document_encoder, [document], self.SEQUENCE_LENGTH)
                n = 4

                max_size = max(batch_x.shape[0] - n, 0)
                a = random.randint(0, max_size)
                batch_n = batch_x[a:a + n]

                batch_x_negative = self.get_x_negative(idx, batch_n.shape[0])
                batch_x_positives = self.get_x_positive(batch_x, batch_n.shape[0])

                yield batch_n, batch_x_positives, batch_x_negative
            else:
                batch_x = get_document_dataset_sequence(self.document_encoder, [document], self.SEQUENCE_LENGTH)
                for i in range(0, batch_x.shape[0], 32):
                    yield batch_x[i:i+32], torch.zeros((1))

    def get_x_positive(self, batch_x, batch_n):
        indices_b = torch.randint(0, batch_x.shape[1], (3, ))
        if batch_x.shape[0] == 1:
            a = torch.zeros_like(batch_x).fill_(self.document_encoder.PADDING_IDX)
            moved = torch.randint(0, 128)
            a[:-moved] = batch_x[0, moved:]
            return a
        else:
            batch_x_positive = torch.randint(0, batch_x.shape[0], (batch_n,))
            indices_a = torch.randint(0, batch_x_positive.shape[0], (3, ))
            batch_x_positives = batch_x[batch_x_positive].clone()
            batch_x_positives[indices_a, indices_b] = self.document_encoder.PADDING_IDX
            return batch_x_positives

    def get_x_negative(self, idx, batch_n):
        next_document = self.docs[(idx + 1) % len(self.docs)]
        batch_y = get_document_dataset_sequence(self.document_encoder, [next_document], self.SEQUENCE_LENGTH)
        batch_x_negative = torch.randint(0, batch_y.shape[0], (batch_n,))
        return batch_y[batch_x_negative]
    
def get_dataloader(document_encoder: SimpleVocab, variant):
    dataset = TextDataloader(document_encoder, variant)
    return dataset
