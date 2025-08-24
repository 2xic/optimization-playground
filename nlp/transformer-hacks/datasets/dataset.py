from abc import ABC, abstractmethod
from utils.transformer_dataset import TransformerTextDatasetLazy
from typing import Generator
from utils.dataset_tokenizer import (
    HuggingFaceTokenizerWrapper,
)


class BaseDataset(ABC):
    @property
    @abstractmethod
    def dataset(self) -> TransformerTextDatasetLazy:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> HuggingFaceTokenizerWrapper:
        pass

    @abstractmethod
    def get_file_content_tokenizer(self) -> Generator[list, None, None]:
        pass

    @abstractmethod
    def get_file_content_tokenized(self, sequence_size):
        pass
