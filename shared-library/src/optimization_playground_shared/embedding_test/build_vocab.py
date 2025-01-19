
"""
Precompute all the vectors and vocab so that training goes vroom.

python3 -m optimization_playground_shared.embedding_test.build_vocab

"""
from ..nlp.DocumentEncoderSequence import SimpleVocab
from ..nlp.wordpiece.bpe import BPE
import glob
import torch
from torch import Tensor
import os 
from tqdm import tqdm
import torch.nn.functional as F
import torch
from tqdm import tqdm
import zipfile
import io

"""
TODO: we could optimize the tensor by chaining the dimensions

(sequences, 2, SEQUENCE_SIZE)
Where we store both the 
(X, y) pair in the second dim.
"""
"""
# Aggregate a lot of tensors into one large tensors to optimize the loading time. 
CHUNK_SIZE = 4096 * 32
# aggregated batch size
SEQUENCE_LENGTH = 256
current_tensor_batch = None
chunk_number = 0
for filename in tqdm(glob.glob("/root/tensors/*")):
    load = torch.load(filename, weights_only=True)
    if load.shape[-1] < SEQUENCE_LENGTH:
        # We skip this as there a
        continue
    else:
        delta = SEQUENCE_LENGTH - (load.shape[-1] % SEQUENCE_LENGTH) if SEQUENCE_LENGTH < load.shape[-1] else SEQUENCE_LENGTH - load.shape[-1]
        resized = F.pad(load, (0, delta), value=vocab.PADDING_IDX).long()
        # reshape_tensor = vector_to_3d(resized, SEQUENCE_LENGTH)
        reshape_tensor = resized.reshape((resized.shape[0] // SEQUENCE_LENGTH, SEQUENCE_LENGTH))

        if current_tensor_batch is None:
            current_tensor_batch = reshape_tensor
        else:
            current_tensor_batch = torch.concat((
                reshape_tensor,
                current_tensor_batch,
            ), dim=0)
            if current_tensor_batch.shape[0] > CHUNK_SIZE:
                torch.save(current_tensor_batch[:CHUNK_SIZE], os.path.join(root_path, "tensors_processed", str(chunk_number)))
                current_tensor_batch = current_tensor_batch[CHUNK_SIZE:]
                chunk_number += 1
"""

def vector_to_3d(tensor_1d, size):
    assert tensor_1d.ndim == 1, "Input must be a 1D tensor"
    num_windows = len(tensor_1d) - size + 1
    indices = torch.arange(size).unsqueeze(0) + torch.arange(num_windows).unsqueeze(1)    
    slices = tensor_1d[indices]
    tensor_3d = torch.stack([slices[:-1], slices[1:]], dim=1)    
    return tensor_3d

def main():
    root_path = "/root/"
    os.makedirs(os.path.join(root_path, "tensors"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "tensors_processed"), exist_ok=True)
    tensor_dir = os.path.join(root_path, "tensors_processed")

 #   vocab = SimpleVocab()
    vocab = BPE()

    path = vocab.get_path(".", prefix="pretrained")
    print(tensor_dir)
    print(path, os.path.isfile(path), len(glob.glob(tensor_dir)))
    if not os.path.isfile(path) or len(glob.glob(os.path.join(tensor_dir, "*"))) == 0:
        print("Rebuilding index ... ")
        files = sorted(glob.glob("/root/text_document_dataset/*"))
        assert len(files) > 0
        documents = []
        for i in tqdm(files, "fitting documents"):
            with open(i, "r") as file:
                document = file.read()
                if len(document) == 0:
                    continue
                documents.append(document)
        print("Fitting the documents ... ")
        if os.path.isfile(path):
            vocab.load(".", prefix="pretrained")
            assert len(vocab.index.word_index) > 0
            assert len(vocab.index.index_tokens) > 0
        else:
            vocab.fit(documents)
            vocab.lock()
            vocab.save(".", prefix="pretrained")
        for (i, filename) in tqdm(list(zip(documents, files)), "storing pre computed tensors"):
            output = vocab.encode(i)
            assert not (None in output)
            encoded = torch.tensor(output)
            assert i == vocab.decode(output)
            name = os.path.basename(filename)
            torch.save(encoded, os.path.join(tensor_dir, name))
    else:
        vocab.load(".", prefix="pretrained")
    
    """
    Then we iterate over things to store it into a zip file.    
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw_file_name in tqdm(glob.glob(os.path.join(tensor_dir, "*"))):
            t: Tensor = torch.load(raw_file_name)
            zip_file_name = os.path.join(
                "tensors",
                os.path.basename(raw_file_name)
            )
            zf.writestr(zip_file_name, t.numpy().tobytes())
        with open(vocab.get_path(".", prefix="pretrained"), "rb") as file:
            zf.writestr("vocab.pkl", file.read())

    zip_data = zip_buffer.getvalue()
    version = vocab.__class__.__name__
    with open(f"text_document_dataset_tensors_{version}.zip", "wb") as f:
        f.write(zip_data)

    print("OKI")
    exit(0)
        
if __name__ == "__main__":
    main()
