from dataclasses import dataclass
import torch.nn as nn
import torch
from optimization_playground_shared.nlp.PositionalEncoding import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEncoding,
)
from enum import Enum
import torch.nn.functional as F
from .layers import (
    SimpleGQA,
    MultiheadAttention,
    DyT,
    MultiHeadLatentAttention,
    DEVICE,
    BidirectionalAttention,
)
from typing import Optional


class PositionalEmbeddingType(Enum):
    NONE = 0
    NN_EMBEDDING = 1
    SINUSOIDAL = 2
    ROTARY_POSITION_ENCODING = 3


class TransformerLayerType(Enum):
    SIMPLE_NO_ATTENTION = 0
    SIMPLE = 1
    GPT2 = 2
    LLAMA2 = 3
    LLAMA3 = 4
    DEEPSEEK = 5
    # Useful for debugging, use the built in nn.TransformerDecoderLayer
    TORCH_TRANSFORMER_DECODE_LAYER = 6
    # cont.
    BERT = 7


class NormalizationLayerType(Enum):
    LAYER_NORM = 0
    DyT = 1


@dataclass
class Config:
    sequence_length: int
    vocab_size: int
    dim_embeddings: int
    num_attention_heads: int
    num_transformer_layers: int
    # Used by the embeddings layer.
    padding_index: int
    positional_embedding: PositionalEmbeddingType = PositionalEmbeddingType.SINUSOIDAL
    transformer_layer: TransformerLayerType = TransformerLayerType.SIMPLE
    normalization_layer: NormalizationLayerType = NormalizationLayerType.LAYER_NORM
    dropout: float = 0.01
    feed_forward_layer: int = 2048
    bias: bool = False
    # Optimizer
    # TODO: this can be removed.
    weight_decay = 1e-1
    learning_rate = 3e-4
    max_grad_norm: Optional[float] = 1
    # TODO: this can be removed.
    model_name: Optional[str] = None

    def with_positional_embedding(self, positional_embedding: PositionalEmbeddingType):
        self.positional_embedding = positional_embedding
        return self

    def with_transformer_layer(self, transformer_layer: TransformerLayerType):
        self.transformer_layer = transformer_layer
        return self

    def with_normalization_layer(self, normalization_layer: NormalizationLayerType):
        self.normalization_layer = normalization_layer
        return self


class NormalizationLayer(nn.Module):
    def __init__(self, config: Config, size):
        super().__init__()
        self.config = config

        if self.config.normalization_layer == NormalizationLayerType.DyT:
            self.norm = DyT(size)
        elif self.config.normalization_layer == NormalizationLayerType.LAYER_NORM:
            self.norm = nn.LayerNorm(size)
        else:
            raise Exception(f"Unknown {self.config.normalization_layer}")

    def forward(self, X):
        return self.norm(X)


class LlamaFeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.l1 = nn.Linear(config.dim_embeddings, 256)
        self.l2 = nn.Linear(256, config.dim_embeddings)
        self.gate = nn.Linear(config.dim_embeddings, 256)

    def forward(self, X):
        return self.l2(F.silu(self.l1(X) * self.gate(X)))


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len, gqa):
        super().__init__()
        if gqa:
            self.mha = SimpleGQA(
                embed_dim=embed_dim,
                num_query_heads=num_heads,
                num_groups=1,
            )
        else:
            self.mha = MultiheadAttention(
                embed_dim=embed_dim,
                num_query_heads=num_heads,
            )
        self.rope = RotaryPositionalEncoding(d_model=embed_dim, max_len=max_len)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.rope(q)
        k = self.rope(k)

        attn_output, _ = self.mha(q, k, v, attn_mask=mask)
        return attn_output


class Llama2TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.dim_embeddings)
        self.rope_attention = RoPEMultiheadAttention(
            config.dim_embeddings,
            config.num_attention_heads,
            config.sequence_length,
            gqa=False,
        )
        self.norm_2 = nn.RMSNorm(config.dim_embeddings)
        self.linear = LlamaFeedForward(config)

    def forward(self, X, mask):
        first_half = self.rope_attention(self.norm_1(X), mask)
        first_half += X
        second_half = self.linear(first_half)
        return self.norm_2(second_half + X)


class Llama3TransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.dim_embeddings)
        self.rope_attention = RoPEMultiheadAttention(
            config.dim_embeddings,
            config.num_attention_heads,
            config.sequence_length,
            gqa=True,
        )
        self.norm_2 = nn.RMSNorm(config.dim_embeddings)
        self.linear = LlamaFeedForward(config)

    def forward(self, X, mask):
        first_half = self.rope_attention(self.norm_1(X), mask)
        first_half += X
        second_half = self.linear(first_half)
        return self.norm_2(second_half + X)


class DeepSeekLikeTransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.dim_embeddings)
        self.attention = MultiHeadLatentAttention(
            config.dim_embeddings,
            config.num_attention_heads,
            config.sequence_length,
            num_groups=None,
        )
        self.norm_2 = nn.RMSNorm(config.dim_embeddings)
        self.linear = LlamaFeedForward(config)

    def forward(self, X, mask):
        X_norm = self.norm_1(X)
        first_half = self.attention(X_norm, X_norm, X_norm, mask)
        first_half += X
        second_half = self.linear(first_half)
        return self.norm_2(second_half + X)


class BertLikeTransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.dim_embeddings)
        self.attention = BidirectionalAttention(
            config.dim_embeddings,
            config.num_attention_heads,
        )
        self.norm_2 = nn.RMSNorm(config.dim_embeddings)
        self.linear = LlamaFeedForward(config)

    def forward(self, X, mask):
        X_norm = self.norm_1(X)
        first_half = self.attention(X_norm, X_norm, X_norm, mask)
        first_half += X
        second_half = self.linear(first_half)
        return self.norm_2(second_half + X)


class GptTransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.configs = config
        self.self_attention = MultiheadAttention(
            embed_dim=config.dim_embeddings,
            num_query_heads=config.num_attention_heads,
        )
        self.dropout = nn.Dropout(p=self.configs.dropout)
        self.linear = nn.Sequential(
            *[
                nn.Linear(self.configs.dim_embeddings, 256),
                nn.GELU(),
                nn.Linear(256, self.configs.dim_embeddings),
            ]
        )
        self.layer_norm_in = NormalizationLayer(config, config.dim_embeddings)
        self.layer_norm_out = NormalizationLayer(config, config.dim_embeddings)

    def forward(self, X, mask):
        attn_output = self.get_attention_output(X, mask)
        first_half = self.layer_norm_in(X + self.dropout(attn_output))
        second_half = self.dropout(self.layer_norm_out(self.linear(first_half)))
        return first_half + second_half

    def get_attention_output(self, X, mask):
        attn_output, _ = self.self_attention(X, X, X, attn_mask=mask)
        return attn_output


class SimpleTransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.configs = config
        self.self_attention = MultiheadAttention(
            embed_dim=config.dim_embeddings,
            num_query_heads=config.num_attention_heads,
        )
        self.module = nn.Sequential(
            *[
                nn.Linear(self.configs.dim_embeddings, self.configs.dim_embeddings),
                nn.ReLU(),
            ]
        )
        self.layer_norm_in = NormalizationLayer(config, config.dim_embeddings)
        self.layer_norm_out = NormalizationLayer(config, config.dim_embeddings)

    def forward(self, X, mask):
        attn_output = self.get_attention_output(X, mask)
        return self.layer_norm_out(self.module(self.layer_norm_in(X + attn_output)))

    def get_attention_output(self, X, mask):
        if self.configs.transformer_layer == TransformerLayerType.SIMPLE:
            attn_output, _ = self.self_attention(X, X, X, attn_mask=mask)
            return attn_output
        elif self.configs.transformer_layer == TransformerLayerType.SIMPLE_NO_ATTENTION:
            return torch.zeros_like(X, device=DEVICE)
        else:
            raise Exception(f"Unknown attention type {self.configs.attention_type}")


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, layer: nn.TransformerDecoderLayer) -> None:
        super(TransformerDecoderWrapper, self).__init__()
        self.layer = layer

    def forward(self, x, mask):
        return self.layer(x, torch.zeros_like(x), tgt_mask=mask)


def get_transformer_layer(config: Config):
    if config.transformer_layer in [
        TransformerLayerType.SIMPLE,
        TransformerLayerType.SIMPLE_NO_ATTENTION,
    ]:
        return SimpleTransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.GPT2:
        return GptTransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.LLAMA2:
        return Llama2TransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.LLAMA3:
        return Llama3TransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.DEEPSEEK:
        return DeepSeekLikeTransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.BERT:
        return BertLikeTransformerLayer(config)
    elif (
        config.transformer_layer == TransformerLayerType.TORCH_TRANSFORMER_DECODE_LAYER
    ):
        return TransformerDecoderWrapper(
            nn.TransformerDecoderLayer(
                d_model=config.dim_embeddings,
                nhead=config.num_attention_heads,
                dim_feedforward=config.feed_forward_layer,
                dropout=config.dropout,
                batch_first=True,
                activation=nn.functional.gelu,
            )
        )
    else:
        raise Exception(f"unknown type {config.transformer_layer}")


class NnPositionalEmbedding(nn.Module):
    def __init__(self, sequence_length: int, dim_embeddings: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(sequence_length, dim_embeddings)

    def forward(self, x: torch.Tensor):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return x + self.positional_embeddings(positions)


class PositionalEmbeddings(nn.Module):
    def __init__(
        self,
        positional_embedding: PositionalEmbeddingType,
        sequence_length: int,
        dim_embeddings: int,
    ):
        super().__init__()
        if positional_embedding == PositionalEmbeddingType.NONE:
            self.positional_embeddings = self._none
        elif positional_embedding == PositionalEmbeddingType.NN_EMBEDDING:
            self.positional_embeddings = NnPositionalEmbedding(
                sequence_length, dim_embeddings
            )
        elif positional_embedding == PositionalEmbeddingType.SINUSOIDAL:
            self.positional_embeddings = SinusoidalPositionalEncoding(
                max_len=sequence_length,
                d_model=dim_embeddings,
            )
        elif positional_embedding == PositionalEmbeddingType.ROTARY_POSITION_ENCODING:
            self.positional_embeddings = RotaryPositionalEncoding(
                max_len=sequence_length,
                d_model=dim_embeddings,
            )
        else:
            raise Exception(f"Unknown {positional_embedding}")

    def _none(self, x):
        return x

    def forward(self, x: torch.Tensor):
        return self.positional_embeddings(x)


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        """
        All the transformer models are usually like this:
            1. Text and position embeddings
            2. Transformer layer
            3. Dense layer
        """
        self.embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim_embeddings,
            padding_idx=config.padding_index,
        )
        self.positional_embeddings = PositionalEmbeddings(
            self.config.positional_embedding,
            self.config.sequence_length,
            self.config.dim_embeddings,
        )
        self.transformer_layers = nn.ModuleList(
            [
                get_transformer_layer(config)
                for _ in range(self.config.num_transformer_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = NormalizationLayer(config, self.config.dim_embeddings)
        self.output_layer = nn.Linear(
            self.config.dim_embeddings,
            self.config.vocab_size,
            bias=False,
        )

        # Weight Tying
        # https://paperswithcode.com/method/weight-tying
        # Found using minigpt
        self.embeddings.weight = self.output_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        x = self.embeddings(x)
        x = self.positional_embeddings(x)
        x = self.dropout(x)
        # Attention mask
        mask = torch.triu(
            torch.ones(
                self.config.sequence_length, self.config.sequence_length, device=DEVICE
            ),
            diagonal=1,
        ).bool()
        for layer in self.transformer_layers:
            x = layer(x, mask)
        # Output
        x = self.layer_norm(x)
        return self.output_layer(x)

    @torch.no_grad()
    def generate(self, input_tokens: torch.Tensor, num_tokens_generate, sampler):
        output = input_tokens
        output = output.unsqueeze(0)

        for _ in range(num_tokens_generate):
            logits = self.forward(output[:, -self.config.sequence_length :])
            last_token = logits[:, -1, :]
            next_token = sampler(last_token).unsqueeze(0)
            output = torch.cat((output, next_token), dim=1)
        return output[0].tolist()
