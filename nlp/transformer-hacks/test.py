from model import Model, Config
import torch
import glob
from dataset_tokenizer import SimpleTextEncoder, WordPieceBuilder, WordPiece
from transformer_dataset import TransformerTextDataset

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
    
    new_tokenizer, cache = SimpleTextEncoder(
        "test"
    ).load_cache()
    assert cache
    assert len(new_tokenizer.idx_vocab) == len(tokenizer.idx_vocab)

def test_bpe_encoder():
    bpe = WordPieceBuilder("test")
    bpe.add_document("hello world")
    bpe.add_document("hello bagel")
    bpe.add_document("hello world")
        
    new_tokenizer = bpe.build(20)
    assert new_tokenizer.decode(new_tokenizer.encode("hello")) == "hello"
    assert new_tokenizer.decode(new_tokenizer.encode("bagel")) == "bagel"

def test_bpe_encoder_from_file():
    bpe = WordPieceBuilder("test")
    bpe.build_from_iterator(
        glob.iglob("*.py")
    )
    new_tokenizer = bpe.build(20)
    assert new_tokenizer.decode(new_tokenizer.encode("hello")) == "hello"
    assert new_tokenizer.decode(new_tokenizer.encode("bagel")) == "bagel"

    new_tokenizer.save_cache()
    (new_tokenizer, cached) = WordPiece.load_cache(bpe.name)
    assert cached
    assert new_tokenizer.decode(new_tokenizer.encode("hello")) == "hello"
    assert new_tokenizer.decode(new_tokenizer.encode("bagel")) == "bagel"

def test_transformer_dataset():
    bpe = WordPieceBuilder("test")
    bpe.build_from_iterator(
        glob.iglob("*.py")
    )
    new_tokenizer = bpe.build(20)
    assert new_tokenizer.decode(new_tokenizer.encode("hello")) == "hello"
    assert new_tokenizer.decode(new_tokenizer.encode("bagel")) == "bagel"

    dataset = TransformerTextDataset.from_iterator_single(
        "test",
        new_tokenizer,
        glob.iglob("*.py"),
        128
    )
  #  dataset.save("test")
#    reloaded_dataset = TransformerTextDataset.load("test", new_tokenizer)
 #   assert len(reloaded_dataset.X) == len(reloaded_dataset.y)
 #   assert len(reloaded_dataset.X) > 0
 #   assert len(reloaded_dataset.X) == len(dataset.y)
