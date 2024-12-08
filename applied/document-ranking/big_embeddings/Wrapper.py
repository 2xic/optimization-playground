from .big_gpt import get_model_bpe, get_document_dataset, SEQUENCE_LENGTH
from typing import List
import os
import glob
import torch
from tqdm import tqdm

class Wrapper:
    def __init__(self) -> None:
        self.bpe, self.model = get_model_bpe()
        self.load()
    
    def load(self):
        print(self.model.embedding.weight)
        full_state_dict = self.model.state_dict()
        for i in glob.glob(os.path.join(os.path.dirname(__file__), "*.pth")):
            state_dict = torch.load(i, map_location='cpu', weights_only=True)
            full_state_dict.update(state_dict)
        self.model.load_state_dict(full_state_dict)

    def train(self, x):
        return self.transforms(x)
    
    def transforms(self, documents: List[str]):
        embeddings = []
        for v in tqdm(documents):
            encoded_X, _ = get_document_dataset(self.bpe, [v], SEQUENCE_LENGTH, SKIPS_SIZE=SEQUENCE_LENGTH)
            output = self.model.embeddings(encoded_X[0])
            embeddings.append(output.detach())
        return embeddings


if __name__ == "__main__":
    model = Wrapper()
    print(model.transforms([
        "hello bro",
        "what bro"
    ]))
