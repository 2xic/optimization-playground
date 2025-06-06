import glob
from utils.dataset_tokenizer import (
    HuggingFaceTokenizerWrapper,
)
from utils.transformer_dataset import TransformerTextDataset, TransformerTextDatasetLazy
import asyncio


class BytecodeDataset:
    def __init__(self, name="bytecode", kind="next_token"):
        # 1. Tokenize
        # 2. Create dataset of a given sequence size
        # 3. Train on it
        self._name = name
        self.base = "/home/brage/bigdrive/evm-contract-data/evm-opcodes-data/**/*.txt"
        self.kind = kind

    def create_tokenizer(self, name):
        (new_tokenizer, cached) = HuggingFaceTokenizerWrapper.load_cache(name)
        if not cached:
            new_tokenizer.train_tokenizer(self.get_file_content_tokenizer())
            new_tokenizer.save_cache()
        return (new_tokenizer, cached)

    def create_dataset(self, sequence_size=512, recreate=False):
        name = self.get_dataset_name(sequence_size)
        # We now have the dataset and can try to train the model on it.
        (new_tokenizer, cached) = self.create_tokenizer("bytecode")
        text_dataset = TransformerTextDatasetLazy.load(name, new_tokenizer)
        # text_dataset = None
        if text_dataset is None or not cached or recreate:
            print("Starting dataset generation .. ", text_dataset)
            text_dataset = asyncio.run(
                TransformerTextDataset.from_iterator_single(
                    name,
                    new_tokenizer,
                    self.get_file_path(),
                    sequence_length=sequence_size,
                )
            )
        text_dataset._sequence_size = sequence_size
        # Just decrease the max size of the model, to force it to train
        # text_dataset.max_size = 1
        return new_tokenizer, text_dataset

    def get_dataset_name(self, sequence_size):
        if self.kind != "next_token":
            return "masked_" + self._name + "_" + str(sequence_size)
        else:
            return self._name + "_" + str(sequence_size)

    def get_file_content_tokenizer(self):
        files = self.get_files()
        for index in range(0, len(files), 1_00):
            batch = files[index : index + 1_00]
            dataset = []
            for i in batch:
                with open(i, "r") as file:
                    dataset.append(file.read())
            yield dataset

    def get_file_path(self):
        for path in self.get_files():
            yield path

    def get_files(self):
        return glob.glob(
            self.base,
            recursive=True,
        )


class BytecodeDatasetTiny(BytecodeDataset):
    def __init__(self, name="bytecode_tiny", kind="next_token"):
        super().__init__(name, kind)

    def get_files(self):
        return sorted(
            glob.glob(
                self.base,
                recursive=True,
            )
        )[:1_000]


if __name__ == "__main__":
    #    BytecodeDataset().create_dataset(sequence_size=512, recreate=True)
    #    BytecodeDataset().create_dataset(sequence_size=32, recreate=True)
    pass
    BytecodeDatasetTiny().create_dataset(sequence_size=512)
