"""
Same as GptTransformer, but modified to be able to run on multiple GPUs 

https://pytorch.org/docs/stable/pipeline.html
https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html
"""

import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor
from .PositionalEncoding import PositionalEncoding
from .GptTransformer import GptTransformerModel
import torch

"""
GPT is a decoder only 
"""


@dataclass
class Config:
    vocab_size: int
    embedding_dim: int
    sequence_size: int
    # transformer decoder config
    transformer_layers: int
    attention_heads: int
    dropout: int
    feed_forward: int
    # vocab config
    padding_index: int


def _verify_module(module: nn.Sequential) -> None:
    if not isinstance(module, nn.Sequential):
        raise TypeError(
            f"module must be nn.Sequential to be partitioned, got {type(module)}")

    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError("module with duplicate children is not supported")


class EmbeddingPositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout
        )

    def forward(self, X: Tensor) -> Tensor:
        source = self.embedding(X) + self.pos_encoder(X)
        return source


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class DecoderWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer = nn.TransformerDecoder(**kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return self.layer(X, X)


def get_model_from_config(config: Config):
    # gpu 1
    embedding = nn.Sequential(*[
        EmbeddingPositionalEncoding(config)
    ]).to("cuda:0")
    decoder_layer = nn.TransformerDecoderLayer(
        config.embedding_dim, config.attention_heads, config.feed_forward, config.dropout)
    # blocks -> n
    transformer_decoder_1 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(1))])
    transformer_decoder_2 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(2))])
    transformer_decoder_3 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(3))])
    transformer_decoder_4 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(4))])
    transformer_decoder_5 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(5))])
    transformer_decoder_6 = nn.Sequential(*[DecoderWrapper(decoder_layer=decoder_layer, num_layers=(
        8)).to("cuda:" + str(6))])

    output = nn.Sequential(*[
        Reshape(),
        nn.Linear(config.embedding_dim *
                  config.sequence_size, config.vocab_size),
    ]).to("cuda:7")

    return torch.nn.Sequential(*[
        embedding,
        transformer_decoder_1,
        transformer_decoder_2,
        transformer_decoder_3,
        transformer_decoder_4,
        transformer_decoder_5,
        transformer_decoder_6,
        output
    ])
