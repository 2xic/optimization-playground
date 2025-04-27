from dataset_tokenizer import WordPiece, WordPieceBuilder, SimpleTextEncoder, HuggingFaceTokenizerWrapper
from transformer_dataset import TransformerTextDataset, TransformerTextDatasetLazy
import glob
from train import train,  ModelStateSaver, create_config, Model, DEVICE, Config, TrainingOptions
import torch
from optimization_playground_shared.nlp.utils.sampling import (
    temperature_sampling,
    argmax_sampling
)
from tqdm import tqdm
import argparse

USE_WORDPIECE_TOKENIZER = False
RETOKENIZE = True

def get_file_reader_tokenizer():
    for i in glob.iglob( "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol",
        recursive=True):
        with open(i, "r") as file:
            yield file.read()

def get_tokenizer():
    """
    new_tokenizer = None
    if USE_WORDPIECE_TOKENIZER:
        name = "smart_contract_fiesta"
        try:
            (new_tokenizer, cached) = WordPiece.load_cache(name)
            assert cached
        except Exception as e:
            print(e)
            print("Creating dataset")
            new_tokenizer = WordPieceBuilder(name).build_from_iterator(
                create_iterator()
            ).build(
                100_000
            )
            new_tokenizer.save_cache()
    else:
        name = "smart_contract_fiesta_word_idx"
        create_iterator = lambda: glob.iglob(
            "/home/brage/bigdrive/smart-contract-fiesta/organized_contracts/**/**/main.sol",
            recursive=True
        )
        try:
            (new_tokenizer, cached) = SimpleTextEncoder.load_cache(name)
            assert cached
            # raise Exception("ops")
        except Exception as e:
            print(e)
            print("Creating dataset")
            new_tokenizer = SimpleTextEncoder(name).build_from_iterator(
                create_iterator()
            )
            new_tokenizer.save_cache()
    """
    name = "smart_contract_fiesta_word_idx_hf"
    (new_tokenizer, cached) = HuggingFaceTokenizerWrapper.load_cache(name)
    if not cached or RETOKENIZE:
        new_tokenizer.encode_document(get_file_reader_tokenizer())
        new_tokenizer.save_cache()
        cached = False
    print("Done building the dataset.")
    return new_tokenizer, name, cached

def get_dataset():
    new_tokenizer, name, cached = get_tokenizer()
    new_tokenizer.is_locked = True

    # We now have the dataset and can try to train the model on it.
    text_dataset = TransformerTextDatasetLazy.load(name, new_tokenizer)
    if text_dataset is None or cached == False:
        print("Starting dataset generation .. ", text_dataset)
        text_dataset = TransformerTextDataset.from_iterator_single(
            name,
            new_tokenizer, 
            get_file_reader_tokenizer(), 
            sequence_length=256,
        )
    # Just decrease the max size of the model, to force it to train
    # text_dataset.max_size = 1
    return new_tokenizer, text_dataset

def override_config(config: Config) -> Config:
    config.num_transformer_layers = 4
    config.dim_embeddings = 128
    config.num_attention_heads = 16
    return config

def train_model():
    tokenizer, text_dataset = get_dataset()
    assert tokenizer.is_locked
    print("Starting training .. ", text_dataset)
   # print(text_dataset.vocab_size)
   # print(len(tokenizer.vocab_idx))
   # exit(0)
    # Encoded: 72009
    #   vocab; 71971
    #          71970

    train(
        text_dataset,
        override=override_config,
        options=TrainingOptions(
            batch_size=32,
            epochs=5,
        )
    )

def sample_model():
    with torch.no_grad():
        new_tokenizer, text_dataset = get_dataset()
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

        #X, _y = text_dataset.sample(1)[0]
        #INPUT_TOKENS = X.to(DEVICE)
        #ORIGINAL_INPUT = X.clone().to(DEVICE)

        for i in range(128):
            predicted = model(INPUT_TOKENS.reshape((1, -1)))
            y_sample_next = argmax_sampling(predicted[:, -1, :])
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
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'sample':
        sample_model()
    else:
        raise Exception(f"not implemented {args.mode}")