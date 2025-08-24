import torch
from experiments import (
    create_default_config,
    TransformerLayerType,
    Experiment,
    Datasets,
    TrainingOptions,
)
from training.trainer import EpochData
import time


def epoch_callback(data: EpochData):
    print("Storing model to disk")
    model = data.model
    torch.save(model.state_dict(), "evm_big_bytecode_model_debug")


last_storage = time.time()


def batch_callback(data: EpochData):
    global last_storage
    if (time.time() - last_storage) > 1800:
        print("Storing model to disk")
        model = data.model
        torch.save(model.state_dict(), "evm_big_bytecode_model_debug")
        last_storage = time.time()


def main():
    DEBUG_DEVICE = torch.device("cuda:0")
    dataset = Datasets.big_evm_bytecode()
    experiment = Experiment(dataset)
    experiment.skip_thread = True
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
            batch_callback=batch_callback,
        ),
    )
    experiment.plot("debug_masked_tokens_embeddings.png")


if __name__ == "__main__":
    main()
