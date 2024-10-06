"""
So the idea would be to have this file run on the data portal machine to generate a metadata file with 
vocab data etc.
"""

from optimization_playground_shared.nlp.SimpleVocab import SimpleVocab, splitter
import os
import pickle
import glob
from optimization_playground_shared.nlp.wordpiece.bpe import BPE
import time

path = "/mnt/blockstorage/text-dataset-2/**/*.txt"
source = ".source_vocab_metadata"
source_bpe = ".source_vocab_metadata_bpe"
bpe_pre_merge = source_bpe + "_pre_merge"

def get_bpe() -> BPE:
    path = os.path.join(
        os.path.dirname(__file__),
        source_bpe
    )
    assert os.path.isfile(path)
    with open(path, "rb") as file:
        return pickle.load(file)

def create_vocab_dataset_bpe(from_scratch=False, validate=False) -> SimpleVocab:
    print("Creating BPE dataset")
    # we create it no files are found
    bpe = None
    if not os.path.isfile(source_bpe) or from_scratch:
        bpe = BPE()
        test_tokens = None
        for _, i in enumerate(glob.iglob(path, recursive=True)):
            print(i)
            with open(i, "r") as file:
                content = file.read()
                bpe.add_vocab(content)
                print(len(content))
                test_tokens = splitter(content)
        print(sorted(bpe.index.word_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
        print("Added tokens ... starting merger")
        print("Before ", len(bpe.index.tokens_index))
    else:
        bpe = get_bpe()
    
    start = time.time()
    # Run for 10 minutes and let's see what it comes up with
    while time.time() - start < 60 * 10:
        bpe.merge(
            n=1
        )
    with open(source_bpe, "wb") as file:
        pickle.dump(bpe, file)

    # validation
    if validate:
        for v in test_tokens:
            tokens = []
            for i in bpe.encode(v):
                tokens.append(bpe.index.tokens_index[i])
            assert v == bpe.decode(tokens)

if __name__ == "__main__":
    results = create_vocab_dataset_bpe(
        from_scratch=False,
        validate=False,
    )
  #  print(results)
  #  print(results.encode("Developed"))
