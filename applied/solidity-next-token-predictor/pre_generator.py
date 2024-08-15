"""
So the idea would be to have this file run on the data portal machine to generate a metadata file with 
vocab data etc.
"""

from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab
import os
import pickle
import glob

path = "/mnt/blockstorage/smart-contract-fiesta/organized_contracts/**/main.sol"
source = ".source_vocab_metadata"

def get_cache_file() -> SimpleVocab:
    if os.path.isfile(source):
        with open(source, "rb") as file:
            return pickle.load(file)
    return None

def create_vocab_dataset() -> SimpleVocab:
    cache_file = get_cache_file()
    if cache_file is None:
        source_vocab = SimpleVocab()
        for i in glob.iglob(path, recursive=True):
            print(i)
            with open(i, "r") as file:
                source_vocab.encode(file.read())
        with open(source, "wb") as file:
            pickle.dump(source_vocab, file)
        return source_vocab

if __name__ == "__main__":
    create_vocab_dataset()
