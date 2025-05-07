from dataset_tokenizer import WordPiece, WordPieceBuilder, SimpleTextEncoder, HuggingFaceTokenizerWrapper
from transformer_dataset import TransformerTextDataset, TransformerTextDatasetLazy
import glob 

def get_file_path_iterator():
    # https://github.com/huggingface/transformers/issues/13844
    # https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#train-tokenizer
    #files = glob.glob( "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol", recursive=True)
    files = glob.iglob( "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol", recursive=True)
    for path in files:
        yield path

def get_file_content_tokenizer():
    files = glob.glob( "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol", recursive=True)
    for index in range(0, len(files), 1_00):
        batch = files[index:index+1_00]
        dataset = []
        for i in batch:
            with open(i, "r") as file:
                dataset.append(file.read())
        yield dataset

def get_tokenizer(name):
#    name = f"smart_contract_fiesta_word_idx_hf"
    (new_tokenizer, cached) = HuggingFaceTokenizerWrapper.load_cache(name)
    if not cached:
        new_tokenizer.train_tokenizer(get_file_content_tokenizer())
        new_tokenizer.save_cache()
    print("Done building the dataset.")
    return new_tokenizer, name, cached

def get_dataset(name, sequence_length):
    new_tokenizer, name, cached = get_tokenizer(name)
    new_tokenizer.is_locked = True

    # We now have the dataset and can try to train the model on it.
    text_dataset = TransformerTextDatasetLazy.load(name, new_tokenizer)
   # text_dataset = None
    if text_dataset is None or cached == False:
        print("Starting dataset generation .. ", text_dataset)
        text_dataset = TransformerTextDataset.from_iterator_single(
            name,
            new_tokenizer, 
            get_file_path_iterator(), 
            sequence_length=sequence_length,
        )
    # Just decrease the max size of the model, to force it to train
    # text_dataset.max_size = 1
    return new_tokenizer, text_dataset

if __name__ == "__main__":
    sequence_length = 32
    get_dataset(
        name=f"smart_contract_fiesta_word_idx_hf_{sequence_length}",
        sequence_length=sequence_length
    )
