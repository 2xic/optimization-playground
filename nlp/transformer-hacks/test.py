from model import Model, Config
import torch

def test_basic_model():
    config = Config(
        sequence_length=3,
        dim_embeddings=8,
        vocab_size=8,
        num_transformer_layers=1
    )
    batch_size = 32
    model = Model(config)
    tensor = torch.zeros((batch_size, config.sequence_length), dtype=torch.long)
    assert (batch_size, config.sequence_length, config.vocab_size) == model(tensor).shape



