from dataset_tokenizer import WordPiece, WordPieceBuilder
from transformer_dataset import TransformerTextDataset
import glob

def main():
    new_tokenizer = None
    name = "smart_contract_fiesta"
    create_iterator = lambda: glob.iglob(
        "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol",
        recursive=True
    )
    try:
        (new_tokenizer, cached) = WordPiece.load_cache(name)
        assert cached
    except Exception as e:
        print(e)
        print("Creating dataset")
        new_tokenizer = WordPieceBuilder(name).build_from_iterator(
            create_iterator()
        ).build(
            100_000
        )
        new_tokenizer.save_cache()
    # We now have the dataset and can try to train the model on it.
    text_dataset = None #TransformerTextDataset.load(name, new_tokenizer)
    if text_dataset is None:
        print("Starting dataset generation .. ", text_dataset)
        text_dataset = TransformerTextDataset.from_iterator_single(
            name,
            new_tokenizer, 
            create_iterator(), 
            sequence_length=256,
        )
        text_dataset.save(name)
    print("Starting training .. ", text_dataset)


if __name__ == "__main__":
    main()
