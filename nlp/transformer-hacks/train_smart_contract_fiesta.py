from train import train,  ModelStateSaver, create_config, Model, DEVICE, Config, TrainingOptions
import torch
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling
)
import argparse
from train import train,  ModelStateSaver, create_config, Model, DEVICE, Config, TransformerLayerType, TrainingOptions
from model import PositionalEmbeddingType
from smart_contract_fiesta_dataset_creator import get_dataset

#GB_8 = 8 * 1000 * 1000 * 1000
#print(GB_8)
#import resource
#resource.setrlimit(resource.RLIMIT_AS, (GB_8, GB_8))

NAME = "smart_contract_fiesta_word_idx_hf"
SEQUENCE_LENGTH = 256
TEMPERATURE = 0.7

def sampling_predictions(y_predictions):
    return temperature_sampling(
        y_predictions,
        TEMPERATURE,
        top_k=10,
        top_p=0.2
    )

def override_config(config: Config) -> Config:
    config.num_transformer_layers = 6
    config.dim_embeddings = 256
    config.dropout = 0.4
    config.positional_embedding = PositionalEmbeddingType.SINUSOIDAL
    config.transformer_layer = TransformerLayerType.TORCH_TRANSFORMER_DECODE_LAYER
    config.num_attention_heads = 8
    return config

def train_model():
    tokenizer, text_dataset = get_dataset(NAME, SEQUENCE_LENGTH)
    assert tokenizer.is_locked
    print("Starting training .. ", text_dataset)
    text_dataset.max_size = 1_00

    train(
        text_dataset,
        override=override_config,
        options=TrainingOptions(
            batch_size=32,
            learning_rate=1e-3,
            epochs=1_0,
        ),
        sampling=sampling_predictions,
    )

def sample_model():
   # DEVICE = "cpu"
    with torch.no_grad():
        new_tokenizer, text_dataset = get_dataset(NAME, SEQUENCE_LENGTH)
        model = Model(
            override_config(create_config(
                text_dataset.vocab_size,
                text_dataset.padding_index,
                text_dataset.sequence_size,
            ))
        )
        state_saver = ModelStateSaver("loading-test")
        state_saver.load_model_state(model)

        # Now we need to sample from the model.
        model.to(DEVICE)
        tokens = new_tokenizer.encode(
            "// This contract is part of Zellic"
        )
        INPUT_TOKENS = torch.zeros((text_dataset.sequence_size)).long().to(DEVICE)
        INPUT_TOKENS[-len(tokens):] = torch.tensor(tokens)

        ORIGINAL_INPUT = INPUT_TOKENS.clone()

        X, _y = text_dataset.sample(1)[0]
        INPUT_TOKENS = X.to(DEVICE)
        ORIGINAL_INPUT = X.clone().to(DEVICE)

        for i in range(SEQUENCE_LENGTH):
            predicted = model(INPUT_TOKENS.reshape((1, -1)))
            # y_sample_next = argmax_sampling(predicted[:, -1, :])
            y_sample_next = sampling_predictions(predicted[:, -1, :])
            INPUT_TOKENS[:-1] = INPUT_TOKENS[1:].clone()
            INPUT_TOKENS[-1] = y_sample_next

        next_tokens = text_dataset.decode_tokens(ORIGINAL_INPUT.tolist())
        print(next_tokens)
        print("=" * 32)
        next_tokens = text_dataset.decode_tokens(INPUT_TOKENS.tolist())
        print(next_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process training or sampling.')
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], 
                        default='train', help='Mode to run: train or sample (default: train)')
    parser.add_argument("--device", type=str, choices=["gpu", "cpu"], default="gpu")
     
    args = parser.parse_args()
    if args.device == "gpu":
        assert DEVICE.type == "cuda", DEVICE.type

    if args.mode == 'train':
        train_model()
    elif args.mode == 'sample':
        sample_model()
    else:
        raise Exception(f"not implemented {args.mode}")