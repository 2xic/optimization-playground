import torch
from experiments import (
    create_default_config,
    PositionalEmbeddingType,
    TransformerLayerType,
    NormalizationLayerType,
    Experiment,
    Datasets,
    TrainingOptions,
    BytecodeDatasetTiny,
)
from training.trainer import EpochData
from training.model import Model
from evals.autoencoder_embedding_eval import train


def main():
    model_state = torch.load("evm_bytecode_model_debug")
    dataset = Datasets.evm_bytecode()
    # This needs to match whatever was trained on
    # We should probably make this more reusable somehow
    model_config = create_default_config(
        dataset,
    ).with_transformer_layer(TransformerLayerType.BERT)
    # .with_normalization_layer(debug_type)
    model = Model(model_config)
    model.load_state_dict(model_state)
    # l(model_config)

    dataset = BytecodeDatasetTiny(kind="masked")

    train(
        dataset,
        model,
    )


if __name__ == "__main__":
    main()


"""
def epoch_callback(data: EpochData):
    print("Storing model to disk")
    model = data.model
    torch.save(model.state_dict(), "evm_bytecode_model_debug")


def main():
    DEBUG_DEVICE = torch.device("cuda:0")
    dataset = Datasets.evm_bytecode()  # tiny_webdataset()
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
"""
