import zipfile
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from optimization_playground_shared.nlp.SimpleVocab import splitter
from tqdm import tqdm
import torch 
import time
from concurrent.futures import ThreadPoolExecutor
import pickle

bpe = BPE()

# Open the zip file
file_contents = {}
with zipfile.ZipFile('/mnt/blockstorage/smart_contract_dataset.zip', 'r') as zip_ref:
    # Iterate over all files in the zip
    for file_name in tqdm(zip_ref.namelist()):
        if not ".sol" in file_name:
            continue
        # Open each file in the zip
        with zip_ref.open(file_name, "r") as file:
            # Read file content
            content = file.read().decode()
            tokens = splitter(content)
            bpe.add_tokens(tokens)
            file_contents[file_name] = content

with open("bpe_pre_merge", "wb") as file:
    pickle.dump(bpe, file)

start = time.time()

minutes = 5
while (time.time() - start) < minutes * 60:
    bpe.merge()

with ThreadPoolExecutor(max_workers=16) as executor:
    tensors = []
    for files in tqdm(file_contents):
        tensors.append(executor.submit(bpe.encode, file_contents[files]))
    for index, v in enumerate(tensors):
        tensors[index] = v.result()
    torch.save(tensors, 'tensors.pt')

