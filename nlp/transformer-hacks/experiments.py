from model import Config, PositionalEmbeddingType, TransformerLayerType, NormalizationLayerType
from plot import plot_accuracy_loss, Results, MinMaxArray
from transformer_dataset import XorDataset
from typing import Callable
from train import train

"""
TODO
- We should run on multiple datasets
"""


def positional_override(
    positional_embedding: PositionalEmbeddingType,
) -> Callable[[Config], Config]:
    def override(config: Config):
        config.positional_embedding = positional_embedding
        return config

    return override


def positional_embeddings():
    data = {}
    for positional_embedding in [
        PositionalEmbeddingType.NONE,
        PositionalEmbeddingType.NN_EMBEDDING,
        PositionalEmbeddingType.SINUSOIDAL,
        PositionalEmbeddingType.ROTARY_POSITION_ENCODING,
    ]:
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(10):
            (_, accuracy, loss) = train(
                XorDataset(), positional_override(positional_embedding)
            )
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
        data[positional_embedding] = Results(
            accuracy=epochs_accuracy,
            loss=epochs_loss,
        )
    plot_accuracy_loss(data, "positional_embeddings.png")

def attention_override(
    attention_type: TransformerLayerType,
) -> Callable[[Config], Config]:
    def override(config: Config):
        config.transformer_layer = attention_type
        return config

    return override

def transformer_layer():
    data = {}
    for positional_embedding in [
        TransformerLayerType.LLAMA2,
        TransformerLayerType.LLAMA3,
        TransformerLayerType.GPT2,
        TransformerLayerType.SIMPLE,
        TransformerLayerType.SIMPLE_NO_ATTENTION,
    ]:
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(10):
            (_, accuracy, loss) = train(
                XorDataset(), attention_override(positional_embedding)
            )
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
        data[positional_embedding] = Results(
            accuracy=epochs_accuracy,
            loss=epochs_loss,
        )
    plot_accuracy_loss(data, "transformer_layer.png")


def normalization_layer_override(
    normalization_layer: NormalizationLayerType,
) -> Callable[[Config], Config]:
    def override(config: Config):
        config.normalization_layer = normalization_layer
        return config

    return override

def normalization_layer():
    data = {}
    for positional_embedding in [
        NormalizationLayerType.LAYER_NORM,
        NormalizationLayerType.DyT,
    ]:
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(10):
            (_, accuracy, loss) = train(
                XorDataset(), normalization_layer_override(positional_embedding)
            )
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
        data[positional_embedding] = Results(
            accuracy=epochs_accuracy,
            loss=epochs_loss,
        )
    plot_accuracy_loss(data, "normalization_layer.png")


if __name__ == "__main__":
    positional_embeddings()
    transformer_layer()
    normalization_layer()
