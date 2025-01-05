"""
Precompute all the vectors and vocab so that training goes vroom.
"""
from ..nlp.DocumentEncoderSequence import SimpleVocab
import glob
import torch
import os 
from tqdm import tqdm
import torch.nn.functional as F

root_path = "/root/"
os.makedirs(os.path.join(root_path, "tensors"), exist_ok=True)
os.makedirs(os.path.join(root_path, "tensors_processed"), exist_ok=True)

vocab = SimpleVocab()
if not os.path.isfile(vocab.get_path(".", prefix="pretrained")):
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
        torch.save(encoded, os.path.join(root_path, "tensors", name))
else:
    vocab.load(".", prefix="pretrained")

# aggregated batch size
SEQUENCE_LENGTH = 256
for filename in glob.glob("/root/tensors/*"):
    load = torch.load(filename, weights_only=True)
    if load.shape[-1] < SEQUENCE_LENGTH:
        continue
    delta = SEQUENCE_LENGTH - (load.shape[-1] % SEQUENCE_LENGTH) if SEQUENCE_LENGTH < load.shape[-1] else SEQUENCE_LENGTH - load.shape[-1]
    # Now all vectors loaded will be divisible by the sequence length. 
    resized = F.pad(load, (0, delta), value=vocab.PADDING_IDX).long()
    resized = resized.reshape((resized.shape[0] // SEQUENCE_LENGTH, SEQUENCE_LENGTH))
    name = os.path.basename(filename)
    torch.save(resized, os.path.join(root_path, "tensors_processed", name))
