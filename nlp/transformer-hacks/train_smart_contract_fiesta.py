from training.trainer import (
    train,
    ModelStateSaver,
    create_config,
    Model,
    DEVICE,
    Config,
    TrainingOptions,
    TransformerLayerType,
    BETA_1,
    BETA_2,
)
import torch
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    simple_sampling,
)
import argparse
from training.model import PositionalEmbeddingType
from smart_contract_fiesta_dataset_creator import get_dataset
from utils.transformer_dataset import TransformerTextDatasetLazy
from training.scheduler import NoamScheduler
from tqdm import tqdm

# GB_8 = 8 * 1000 * 1000 * 1000
# print(GB_8)
# import resource
# resource.setrlimit(resource.RLIMIT_AS, (GB_8, GB_8))

NAME = "smart_contract_fiesta_word_idx_hf"
MODEL_NAME = "smart_contract_fiesta"
SEQUENCE_LENGTH = 256
TEMPERATURE = 0.7


def sampling_predictions(y_predictions):
    return temperature_sampling(y_predictions, TEMPERATURE, top_k=10, top_p=0.2)


def override_config(config: Config) -> Config:
    config.num_transformer_layers = 6
    config.dim_embeddings = 256
    config.dropout = 0
    config.positional_embedding = PositionalEmbeddingType.NN_EMBEDDING
    config.transformer_layer = TransformerLayerType.TORCH_TRANSFORMER_DECODE_LAYER
    config.num_attention_heads = 8
    # Getting the learning rate is important for stable training
    config.learning_rate = 3e-4  # Used to be 1e-3
    config.max_grad_norm = 1
    #
    config.model_name = MODEL_NAME
    return config


def create_optimizer(module: torch.nn.Module, config: Config):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(BETA_1, BETA_2),
    )
    scheduler = (
        NoamScheduler(optimizer, d_model=config.dim_embeddings, warmup_steps=10_00)
        if False
        else None
    )
    return optimizer, scheduler


def train_model():
    tokenizer, text_dataset = get_dataset(NAME, SEQUENCE_LENGTH)
    assert tokenizer.is_locked
    print("Starting training .. ", text_dataset)
    # text_dataset.max_size = 1_00

    (_, _, _, model) = train(
        text_dataset,
        override=override_config,
        options=TrainingOptions(
            batch_size=32,
            epochs=128,
        ),
        create_optimizer=create_optimizer,
        sampling=sampling_predictions,
        checkpoint=True,
        progress=tqdm,
    )
    X, _ = text_dataset.sample(1)[0]
    sample_from_model(X, text_dataset, model)


def sample_from_model(
    X: torch.Tensor, text_dataset: TransformerTextDatasetLazy, model: Model
):
    # Now we need to sample from the model.
    model.to(DEVICE)
    ORIGINAL_INPUT = X.clone().to(DEVICE)

    print("")
    print("Naive sampling")
    tokens = model.generate(ORIGINAL_INPUT, 128, sampler=simple_sampling)
    print(text_dataset.decode_tokens(tokens))
    print("")
    print("")
    print("Less naive sampling")
    tokens = model.generate(ORIGINAL_INPUT, 128, sampler=sampling_predictions)
    print(text_dataset.decode_tokens(tokens))


def sample_model():
    with torch.no_grad():
        tokenizer, text_dataset = get_dataset(NAME, SEQUENCE_LENGTH)
        model = Model(
            override_config(
                create_config(
                    text_dataset.vocab_size,
                    text_dataset.padding_index,
                    text_dataset.sequence_size,
                )
            )
        )
        model.eval()
        state_saver = ModelStateSaver("loading-test")
        state_saver.load_model_state(model)

        X = torch.zeros(SEQUENCE_LENGTH, dtype=torch.long).fill_(
            text_dataset.padding_index
        )
        y = torch.LongTensor(tokenizer.encode(input("model context: ")))
        X[-(y.shape[0]) :] = y
        print(X.shape)
        sample_from_model(X, text_dataset, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process training or sampling.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sample"],
        default="train",
        help="Mode to run: train or sample (default: train)",
    )
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")

    args = parser.parse_args()
    if args.device == "gpu":
        assert DEVICE.type == "cuda", DEVICE.type

    if args.mode == "train":
        train_model()
    elif args.mode == "sample":
        sample_model()
    else:
        raise Exception(f"not implemented {args.mode}")
