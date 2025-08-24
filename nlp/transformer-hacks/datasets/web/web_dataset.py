import glob
from utils.dataset_tokenizer import (
    HuggingFaceTokenizerWrapper,
)
from utils.transformer_dataset import (
    TransformerTextDataset,
    TransformerTextDatasetLazy,
    MaskedTransformerTextDataset,
)
import asyncio
from datasets.dataset import BaseDataset


class WebDataset(BaseDataset):
    def __init__(self, name="web_dataset_big", kind="next_token"):
        self.name = name
        self.base = "/home/brage/bigdrive/text_dataset_rss/*.txt"
        self.kind = kind
        self._dataset = None

    def create_tokenizer(self):
        (new_tokenizer, cached) = HuggingFaceTokenizerWrapper.load_cache(self.name)
        if not cached:
            new_tokenizer.train_tokenizer(self.get_file_content_tokenizer())
            new_tokenizer.save_cache()
        return (new_tokenizer, cached)

    def create_dataset(self, sequence_size=512, recreate=False):
        # We now have the dataset and can try to train the model on it.
        (new_tokenizer, cached) = self.create_tokenizer()
        name = self.get_dataset_name()
        text_dataset = TransformerTextDatasetLazy.load(name, new_tokenizer)
        # text_dataset = None
        if text_dataset is None or not cached or recreate:
            print("Starting dataset generation .. ", text_dataset)
            if self.kind == "next_token":
                text_dataset = asyncio.run(
                    TransformerTextDataset.from_iterator_single(
                        name,
                        new_tokenizer,
                        self.get_file_path(),
                        sequence_length=sequence_size,
                    )
                )
            else:
                text_dataset = asyncio.run(
                    MaskedTransformerTextDataset.from_iterator_single(
                        name,
                        new_tokenizer,
                        self.get_file_path(),
                        sequence_length=sequence_size,
                    )
                )
        text_dataset._sequence_size = sequence_size
        # Just decrease the max size of the model, to force it to train
        # text_dataset.max_size = 1
        self._dataset = text_dataset
        return self

    @property
    def dataset(self):
        if self._dataset is None:
            raise Exception("Dataset is not yet created")
        return self._dataset

    def get_dataset_name(self):
        if self.kind != "next_token":
            return "masked_" + self.name
        else:
            return self.name

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
        return sorted(
            glob.glob(
                self.base,
                recursive=True,
            )
        )


class WebDatasetSmall(WebDataset):
    def __init__(self, kind):
        super().__init__("web_dataset", kind)

    def get_files(self):
        files = sorted(super().get_files())
        return files[: min(len(files) * 0.25, 100)]
