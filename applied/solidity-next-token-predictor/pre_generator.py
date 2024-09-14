"""
So the idea would be to have this file run on the data portal machine to generate a metadata file with 
vocab data etc.
"""

from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab, splitter
import os
import pickle
import glob
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
from collections import defaultdict

path = "/mnt/blockstorage/smart-contract-fiesta/organized_contracts/**/main.sol"
source = ".source_vocab_metadata"
source_bpe = ".source_vocab_metadata_bpe"

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
            distributions = defaultdict(int)
            with open(i, "r") as file:
                content = file.read()
                source_vocab.encode(content)
                for i in splitter(content):
                    distributions[i] += 1
            print(distributions)

        with open(source, "wb") as file:
            pickle.dump(source_vocab, file)
        return source_vocab

def create_vocab_dataset_bpe() -> SimpleVocab:
    print("Creating BPE dataset")
    bpe = BPE()
    for i in glob.iglob(path, recursive=True):
        print(i)
        with open(i, "r") as file:
            tokens = splitter(file.read())
            bpe.add_tokens(tokens)
    print("Add tokens ... starting merger")
    with open(source_bpe + "_pre_merge", "wb") as file:
        pickle.dump(bpe, file)
    bpe = None
    with open(source_bpe + "_pre_merge", "rb") as file:
        bpe = pickle.load(file)
    bpe.merge()
    with open(source_bpe, "wb") as file:
        pickle.dump(bpe, file)
    return bpe

if __name__ == "__main__":
    results = create_vocab_dataset_bpe()
#    results = create_vocab_dataset_bpe()
#    print(results.index.word_index)
    print(results.index.tokens_index)

