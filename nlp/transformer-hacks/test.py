from model import Model, Config
import torch
import glob
from dataset_tokenizer import SimpleTextEncoder, BpeTokenizer

def test_basic_model():
    config = Config(
        sequence_length=3, 
        dim_embeddings=8, 
        vocab_size=8, 
        num_transformer_layers=1,
        num_attention_heads=2,
        padding_index=-1,
    )
    batch_size = 32
    model = Model(config)
    tensor = torch.zeros((batch_size, config.sequence_length), dtype=torch.long)
    assert (batch_size, config.sequence_length, config.vocab_size) == model(
        tensor
    ).shape

def test_text_encoder():
    tokenizer = SimpleTextEncoder(
        "test"
    ).build_from_files(
        glob.iglob("*.py")
    )
    tokenizer.save_cache()
    
    new_tokenizer = SimpleTextEncoder(
        "test"
    ).load_cache()
    assert len(new_tokenizer.idx_vocab) == len(tokenizer.idx_vocab)

def test_text_encoder():
    tokenizer = SimpleTextEncoder(
        "test"
    ).build_from_files(
        glob.iglob("*.py")
    )
    tokenizer.save_cache()
    
    new_tokenizer = SimpleTextEncoder(
        "test"
    ).load_cache()
    assert len(new_tokenizer.idx_vocab) == len(tokenizer.idx_vocab)

def test_bpe_encoder():
    bpe = BpeTokenizer()
    bpe.add_document("hello world")
    len_tokens = len(bpe.index_word)
    bpe.merge(n=1)
    print(bpe.index_word)
    bpe.merge(n=1000)
    print(bpe.index_word)
    assert len(bpe.index_word) != len_tokens
    assert bpe.decode(bpe.encode("hello")) == "hello"

    bpe.add_document("hello world")
    bpe.add_document("hello bagel")
    bpe.add_document("hello world")
    bpe.merge()
    assert bpe.decode(bpe.encode("hello")) == "hello"
    assert bpe.decode(bpe.encode("bagel")) == "bagel"
