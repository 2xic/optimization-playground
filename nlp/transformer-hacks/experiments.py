from model import (
    Config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Model
)
from layers_mixture_of_experts import MoE
from plot import plot_accuracy_loss, Results, MinMaxArray
from transformer_dataset import XorDataset
from typing import Callable
from train import train

"""
TODO
- We should run on multiple datasets
"""
SAMPLE_SIZE = 1_00


def config_override(
    positional_embedding: PositionalEmbeddingType = None,
    transformer_layer: TransformerLayerType = None,
    normalization_layer: NormalizationLayerType = None,
) -> Callable[[Config], Config]:
    def override(config: Config):
        if positional_embedding is not None:
            config.positional_embedding = positional_embedding
        if transformer_layer is not None:
            config.transformer_layer = transformer_layer
        if normalization_layer is not None:
            config.normalization_layer = normalization_layer
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
        for _ in range(SAMPLE_SIZE):
            (_, accuracy, loss) = train(
                XorDataset(),
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
    for transformer_layer in [
        TransformerLayerType.DEEPSEEK,
        TransformerLayerType.LLAMA2,
        TransformerLayerType.LLAMA3,
        TransformerLayerType.GPT2,
        TransformerLayerType.SIMPLE,
        TransformerLayerType.SIMPLE_NO_ATTENTION,
    ]:
        print(f"Testing {transformer_layer}")
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(SAMPLE_SIZE):
            (_, accuracy, loss) = train(
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
    for normalization_layer in [
        NormalizationLayerType.LAYER_NORM,
        NormalizationLayerType.DyT,
    ]:
        print(f"Testing {normalization_layer}")
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(SAMPLE_SIZE):
            (_, accuracy, loss) = train(
                XorDataset(),
                config_override(
                    normalization_layer=normalization_layer,
                ),
            )
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
        data[normalization_layer] = Results(
            accuracy=epochs_accuracy,
            loss=epochs_loss,
        )
    plot_accuracy_loss(data, "normalization_layer.png")


def mixture_of_expert_model_vs_standard():
    data = {}
    for create_model, name in [
        ((lambda x: Model(x), "normal")),
        ((lambda x: MoE(x), "moe"))
    ]:
        print(f"Testing {name}")
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(SAMPLE_SIZE):
            (_, accuracy, loss) = train(
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
    plot_accuracy_loss(data, "model_vs_moe.png")


def test_pass():
    (_, accuracy, loss) = train(
        XorDataset(),
        config_override(
            transformer_layer=TransformerLayerType.DEEPSEEK,
        ),
    )
#    print(f"Accuracy {accuracy}, Loss: {loss}")


if __name__ == "__main__":
    test_pass()
    mixture_of_expert_model_vs_standard()
    positional_embeddings()
    transformer_layer()
    normalization_layer()
