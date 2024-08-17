from torch_gpt_like_model_bigger import get_cache_file
from optimization_playground_shared.nlp.SimpleVocab import splitter
from pre_generator import path
import glob

if __name__ == "__main__":
    # todo: vocab needs to be pre-generated on the dataloader side.
    vocab = get_cache_file()
    assert vocab is not None
    file = next(glob.iglob(path, recursive=True))

    with open(file, "r") as file:
#        encoded = (vocab.encode(file.read()))
#        print(vocab.decode(encoded))
        print(splitter(file.read()))
