import torch
from experiments import (
    create_default_config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Experiment,
    Datasets,
    TrainingOptions,
)
from training.trainer import EpochData


def epoch_callback(data: EpochData):
    print("Storing model to disk")
    model = data.model
    torch.save(model.state_dict(), "big_webdataset_model_debug")


def main():
    DEBUG_DEVICE = torch.device("cuda:1")
    dataset = Datasets.big_webdataset()  # tiny_webdataset()
    experiment = Experiment(dataset)
    experiment.skip_thread = True
    #    positional_embedding = PositionalEmbeddingType.SINUSOIDAL
    debug_type = TransformerLayerType.BERT
    # debug_type = NormalizationLayerType.DyT
    config = create_default_config(
        dataset,
    ).with_transformer_layer(TransformerLayerType.BERT)
    # .with_normalization_layer(debug_type)

    experiment.queue(
        config,
        debug_type,
        training_options=TrainingOptions(
            epochs=1_000,
            batch_size=32,
            device=torch.device(DEBUG_DEVICE),
            epoch_callback=epoch_callback,
        ),
    )
    experiment.plot("debug_masked_tokens_embeddings.png")


if __name__ == "__main__":
    main()
