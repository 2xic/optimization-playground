from evals.geometry_test import test_model
import torch
from experiments import (
    create_default_config,
    TransformerLayerType,
    Datasets,
)
from training.model import Model


def eval_model(use_big):
    dataset = (
        Datasets.tiny_evm_bytecode() if not use_big else Datasets.big_evm_bytecode()
    )
    model_name = (
        "evm_bytecode_model_debug" if not use_big else "evm_big_bytecode_model_debug"
    )
    print(f"{model_name}:")
    # This needs to match whatever was trained on
    # We should probably make this more reusable somehow
    model_config = create_default_config(
        dataset,
    ).with_transformer_layer(TransformerLayerType.BERT)

    model = Model(model_config)
    model_state = torch.load(model_name)
    model.load_state_dict(model_state)
    model.eval()

    embedding_model = model.create_embedding_model()

    test_model(dataset.dataset, embedding_model)


def main():
    eval_model(use_big=False)
    eval_model(use_big=True)


if __name__ == "__main__":
    main()
