"""
Precompute all the vectors and vocab so that training goes vroom.
"""
from ..nlp.DocumentEncoderSequence import SimpleVocab
import glob
import torch
import os 
from tqdm import tqdm

os.makedirs("tensors", exist_ok=True)

vocab = SimpleVocab()
files = sorted(glob.glob("/root/text_document_dataset/*"))
documents = []
for i in tqdm(files, "fitting documents"):
    with open(i, "r") as file:
        documents.append(file.read())
vocab.fit(documents)
vocab.lock()
vocab.save(".", prefix="pretrained")

for (i, filename) in tqdm(list(zip(documents, files)), "storing pre computed tensors"):
    encoded = torch.tensor(vocab.encode(i))
    name = os.path.basename(filename)
    torch.save(encoded, os.path.join("tensors", name))
