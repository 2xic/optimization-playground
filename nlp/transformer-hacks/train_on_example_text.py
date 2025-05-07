from dataset_tokenizer import HuggingFaceTokenizerWrapper, SimpleTextEncoder
from transformer_dataset import TransformerTextDataset
from train import train,  ModelStateSaver, create_config, Model, DEVICE, Config, TransformerLayerType, TrainingOptions
from model import PositionalEmbeddingType
import torch
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling
)
from optimization_playground_shared.nlp.SimpleVocab import splitter
import argparse

SEQUENCE_LENGTH = 32
TEMPERATURE = 0.8

def get_dataset():
    with open("example_small.text", "r") as file:
        content = file.read()
    tokenizer = HuggingFaceTokenizerWrapper(
        "example",
        vocab_size=len(list(set(splitter(content)))) * 32,
    )
    tokenizer.train_tokenizer([content])
#    tokenizer = SimpleTextEncoder("test")
#    tokenizer.build_from_iterator([content])
   
    dataset = TransformerTextDataset.from_documents(tokenizer, [content], SEQUENCE_LENGTH)
    return tokenizer, dataset

def sampling_predictions(y_predictions):
    return temperature_sampling(y_predictions)
    """
    return temperature_sampling(
        y_predictions,
        TEMPERATURE,
        top_k=10,
        top_p=0.6
    )
    """

def override_config(config: Config) -> Config:
    config.num_transformer_layers = 3
    config.dim_embeddings = 256
    config.num_attention_heads = 8
    config.dropout = 0
    config.positional_embedding = PositionalEmbeddingType.ROTARY_POSITION_ENCODING
    config.transformer_layer = TransformerLayerType.TORCH_TRANSFORMER_DECODE_LAYER
    return config

def train_model():
    tokenizer, text_dataset = get_dataset()
    assert tokenizer.is_locked
    assert len(text_dataset) > 2
    (_, _, _, model) = train(
        text_dataset,
        override=override_config,
        options=TrainingOptions(
            batch_size=32,
            learning_rate=1e-3,
            epochs=50
        ),
        sampling=sampling_predictions
    )
    sample_from_model(text_dataset, model)

def sample_from_model(text_dataset, model):
    # Now we need to sample from the model.
    model.to(DEVICE)

    X, _ = text_dataset.sample(1)[0]
    INPUT_TOKENS = X.to(DEVICE)
    ORIGINAL_INPUT = X.clone().to(DEVICE)

    next_tokens = text_dataset.decode_tokens(ORIGINAL_INPUT.tolist())
    print(next_tokens)
    print("=" * 32)

    raw_outputs = []
    for i in range(128):
        predicted = model(INPUT_TOKENS.reshape((1, -1)))
        y_sample_next = sampling_predictions(predicted[:, -1, :])
        INPUT_TOKENS[:-1] = INPUT_TOKENS[1:].clone()
        INPUT_TOKENS[-1] = y_sample_next
        raw_outputs.append(y_sample_next.item())
    print("Model decoded:")
    print(raw_outputs)
    next_tokens = text_dataset.decode_tokens(raw_outputs)
    print(next_tokens)
    print("=" * 32)

def sample_model():
    with torch.no_grad():
        _, text_dataset = get_dataset()
        model = Model(
            override_config(create_config(
                text_dataset.vocab_size,
                text_dataset.padding_index,
                text_dataset.sequence_size,
            ))
        )
        model.eval()
        state_saver = ModelStateSaver("loading-test")
        state_saver.load_model_state(model)

        sample_from_model(text_dataset, model)

        accuracy = 0
        for index, (X, y) in enumerate(text_dataset.sample(SEQUENCE_LENGTH)):
            X = X.to(DEVICE)
            predicted = model(X.reshape((1, -1)))
            y_sample_next = sampling_predictions(predicted[:, -1, :])

            y_predicted_next = y_sample_next.item()
            y_actual_next = y[-1].item()
            assert  type(y_predicted_next) == type(y_actual_next) and type(y_actual_next) == int
            accuracy += (y_predicted_next == y_actual_next)
        print((accuracy / index) * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process training or sampling.')
    parser.add_argument('--mode', type=str, choices=['train', 'sample'], 
                        default='train', help='Mode to run: train or sample (default: train)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'sample':
        sample_model()
    else:
        raise Exception(f"not implemented {args.mode}")
