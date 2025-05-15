from model import (
    Config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Model,
)
from layers_mixture_of_experts import MoE
from plot import plot_accuracy_loss, Results, MinMaxAvgArray
from transformer_dataset import XorDataset
from typing import Callable
from train import train
from tqdm import tqdm
import requests
from dataset_tokenizer import HuggingFaceTokenizerWrapper
from transformer_dataset import TransformerTextDataset
from optimization_playground_shared.nlp.SimpleVocab import splitter
from typing import List
import os

SAMPLE_SIZE = 1_00
LEARNING_RATE = 3e-4
SEQUENCE_LENGTH = 32


def get_text_dataset():
    content = requests.get(
        "https://raw.githubusercontent.com/ibz/bitcoin-whitepaper-markdown/refs/heads/master/bitcoin-whitepaper.md"
    ).text
    tokenizer = HuggingFaceTokenizerWrapper(
        "example",
        vocab_size=len(list(set(splitter(content)))) * 32,
    )
    tokenizer.train_tokenizer([content])
    dataset = TransformerTextDataset.from_documents(
        tokenizer, [content], SEQUENCE_LENGTH
    )
    return dataset


class NamedDataset:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset


DATASETS: List["NamedDataset"] = [
    NamedDataset("xor", XorDataset()),
    NamedDataset("satoshi_whitepaper", get_text_dataset())
]


def get_output_path(dataset: NamedDataset, filename):
    dir = os.path.join(
        os.path.dirname(__file__),
        "plots",
        dataset.name,
    )
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, filename)


def config_override(
    positional_embedding: PositionalEmbeddingType = None,
    transformer_layer: TransformerLayerType = None,
    normalization_layer: NormalizationLayerType = None,
) -> Callable[[Config], Config]:
    def override(config: Config):
        config.learning_rate = LEARNING_RATE
        # Conditionals
        if positional_embedding is not None:
            config.positional_embedding = positional_embedding
        if transformer_layer is not None:
            config.transformer_layer = transformer_layer
        if normalization_layer is not None:
            config.normalization_layer = normalization_layer
        return config

    return override


def positional_embeddings():
    for dataset in DATASETS:
        data = {}
        for positional_embedding in [
            PositionalEmbeddingType.NONE,
            PositionalEmbeddingType.NN_EMBEDDING,
            PositionalEmbeddingType.SINUSOIDAL,
            PositionalEmbeddingType.ROTARY_POSITION_ENCODING,
        ]:
            epochs_accuracy = MinMaxAvgArray()
            epochs_loss = MinMaxAvgArray()
            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {positional_embedding}"):
                (_, accuracy, loss, _model) = train(
                    dataset.dataset,
                    config_override(
                        positional_embedding=positional_embedding,
                    ),
                )
                epochs_accuracy.add(accuracy)
                epochs_loss.add(loss)
                assert len(epochs_accuracy.min_max_avg) == len(accuracy)
            data[positional_embedding] = Results(
                accuracy=epochs_accuracy,
                loss=epochs_loss,
            )
        plot_accuracy_loss(data, get_output_path(dataset, "positional_embeddings.png"))


def transformer_layer():
    for dataset in DATASETS:
        data = {}
        for transformer_layer in [
            TransformerLayerType.DEEPSEEK,
            TransformerLayerType.LLAMA2,
            TransformerLayerType.LLAMA3,
            TransformerLayerType.GPT2,
            TransformerLayerType.SIMPLE,
            TransformerLayerType.SIMPLE_NO_ATTENTION,
        ]:
            print(f"Testing {transformer_layer}")
            epochs_accuracy = MinMaxAvgArray()
            epochs_loss = MinMaxAvgArray()
            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {transformer_layer}"):
                (_, accuracy, loss, _model) = train(
                    XorDataset(),
                    config_override(
                        transformer_layer=transformer_layer,
                    ),
                )
                epochs_accuracy.add(accuracy)
                epochs_loss.add(loss)
                assert len(epochs_accuracy.min_max_avg) == len(accuracy)
            data[transformer_layer] = Results(
                accuracy=epochs_accuracy,
                loss=epochs_loss,
            )
        plot_accuracy_loss(data, get_output_path(dataset, "transformer_layer.png"))


def normalization_layer():
    for dataset in DATASETS:
        data = {}
        for normalization_layer in [
            NormalizationLayerType.LAYER_NORM,
            NormalizationLayerType.DyT,
        ]:
            epochs_accuracy = MinMaxAvgArray()
            epochs_loss = MinMaxAvgArray()
            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {normalization_layer}"):
                (_, accuracy, loss, _model) = train(
                    XorDataset(),
                    config_override(
                        normalization_layer=normalization_layer,
                    ),
                )
                assert accuracy != 0
                epochs_accuracy.add(accuracy)
                epochs_loss.add(loss)
                assert len(epochs_accuracy.min_max_avg) == len(accuracy)
            data[normalization_layer] = Results(
                accuracy=epochs_accuracy,
                loss=epochs_loss,
            )
        plot_accuracy_loss(data, get_output_path(dataset, "normalization_layer.png"))


def mixture_of_expert_model_vs_standard():
    for dataset in DATASETS:
        data = {}
        for create_model, name in [
            ((lambda x: Model(x), "normal")),
            ((lambda x: MoE(x), "moe")),
        ]:
            epochs_accuracy = MinMaxAvgArray()
            epochs_loss = MinMaxAvgArray()
            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Training {name}"):
                (_, accuracy, loss, _model) = train(
                    XorDataset(),
                    create_model=create_model,
                )
                epochs_accuracy.add(accuracy)
                epochs_loss.add(loss)
                assert len(epochs_accuracy.min_max_avg) == len(accuracy)
            data[name] = Results(
                accuracy=epochs_accuracy,
                loss=epochs_loss,
            )
        plot_accuracy_loss(data, get_output_path(dataset, "model_vs_moe.png"))


def test_pass():
    (_epochs, _epochs_accuracy, _epochs_loss, _model) = train(
        XorDataset(),
        config_override(
            transformer_layer=TransformerLayerType.DEEPSEEK,
        ),
    )
    assert len(_epochs_loss) > 0
    assert len(_epochs_accuracy) > 0
    assert len(_epochs) > 0


if __name__ == "__main__":
 #   test_pass()
#    mixture_of_expert_model_vs_standard()
    positional_embeddings()
    transformer_layer()
    normalization_layer()
