from model import Config, PositionalEmbeddingType
from plot import plot_accuracy_loss, Results, MinMaxArray
from transformer_dataset import XorDataset
from typing import Callable
from train import train

def positional_override(positional_embedding: PositionalEmbeddingType) -> Callable[[Config], Config]:
    def override(config: Config):
        config.positional_embedding = positional_embedding
        return config
    return override

def positional_embeddings():
    data = {}
    for positional_embedding in [PositionalEmbeddingType.NN_EMBEDDING, PositionalEmbeddingType.SINUSOIDAL, PositionalEmbeddingType.ROTARY_POSITION_ENCODING]:
        epochs_accuracy = MinMaxArray()
        epochs_loss = MinMaxArray()
        for _ in range(10):
            (_, accuracy, loss) = train(XorDataset(), positional_override(positional_embedding))
            epochs_accuracy.add(accuracy)
            epochs_loss.add(loss)
            assert len(epochs_accuracy.min_max_avg) == len(accuracy)
        data[positional_embedding] = Results(
            accuracy=epochs_accuracy,
            loss=epochs_loss,
        )
    plot_accuracy_loss(
        data,
        "positional_embeddings.png"
    )

if __name__ == "__main__":
    positional_embeddings()
