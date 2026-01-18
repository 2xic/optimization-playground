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
    BidirectionalAttention,
    SimpleMultiHeadAttention,
)
from dataclasses import dataclass, asdict, fields


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
    BERT = 7
    SIMPLE_ATTENTION_AT_HOME = 8
    OLMO = 9
    OLMO_HYPER_CONNECTIONS = 10
    OLMO_CONSTRAINED_HYPER_CONNECTIONS = 11


class NormalizationLayerType(Enum):
    LAYER_NORM = 0
    DyT = 1


class MaskOrder(Enum):
    TRIU = 0
    TRIL = 1
    NONE = 2


# TODO: this should likely live somewhere else
class SamplingMethod(Enum):
    ARGMAX = 0
    TEMPERATURE = 1


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
    masked_order: MaskOrder = MaskOrder.TRIL
    sampling_method: SamplingMethod = SamplingMethod.TEMPERATURE
    dropout: float = 0.01
    feed_forward_layer: int = 2048
    bias: bool = False
    hc_n: int = 4

    def with_positional_embedding(self, positional_embedding: PositionalEmbeddingType):
        self.positional_embedding = positional_embedding
        return self

    def with_transformer_layer(self, transformer_layer: TransformerLayerType):
        self.transformer_layer = transformer_layer
        return self

    def with_normalization_layer(self, normalization_layer: NormalizationLayerType):
        self.normalization_layer = normalization_layer
        return self

    def to_json(self) -> str:
        def enum_dict_factory(data):
            return {k: v.value if isinstance(v, Enum) else v for k, v in data}

        return asdict(self, dict_factory=enum_dict_factory)

    @classmethod
    def from_json(cls, data: dict) -> "Config":
        converted = {}
        field_types = {f.name: f.type for f in fields(cls)}

        for key, value in data.items():
            field_type = field_types.get(key)
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                converted[key] = field_type(value)
            else:
                converted[key] = value

        return cls(**converted)


class NormalizationLayer(nn.Module):
    def __init__(self, config: Config, size):
        super().__init__()
        self.config = config

        if self.config.normalization_layer == NormalizationLayerType.DyT:
            self.norm = DyT(size)
        elif self.config.normalization_layer == NormalizationLayerType.LAYER_NORM:
            self.norm = nn.LayerNorm(size, bias=config.bias)
        else:
            raise Exception(f"Unknown {self.config.normalization_layer}")

    def forward(self, X):
        return self.norm(X)


class LlamaFeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.l1 = nn.Linear(
            config.dim_embeddings,
            config.feed_forward_layer,
            bias=config.bias,
        )
        self.l2 = nn.Linear(
            config.feed_forward_layer,
            config.dim_embeddings,
            bias=config.bias,
        )
        self.gate = nn.Linear(
            config.dim_embeddings,
            config.feed_forward_layer,
            bias=config.bias,
        )

    def forward(self, X):
        return self.l2(F.silu(self.l1(X) * self.gate(X)))


# https://github.com/meta-llama/llama/blob/v2/llama/model.py#L132-L304
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len, gqa):
        super().__init__()
        head_dim = embed_dim // num_heads
        rope = RotaryPositionalEncoding(d_model=head_dim, max_len=max_len)

        if gqa:
            self.mha = SimpleGQA(
                embed_dim=embed_dim, num_query_heads=num_heads, num_groups=1, rope=rope
            )
        else:
            self.mha = MultiheadAttention(
                embed_dim=embed_dim, num_query_heads=num_heads, rope=rope
            )

    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        return attn_output


# https://github.com/meta-llama/llama/blob/v2/llama/model.py#L351-L410
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
        second_half = self.linear(self.norm_2(first_half))
        second_half += first_half
        return second_half


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
        second_half = self.linear(self.norm_2(first_half))
        second_half += first_half
        return second_half


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
        second_half = self.linear(self.norm_2(first_half))
        second_half += first_half
        return second_half


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
        second_half = self.linear(self.norm_2(first_half))
        second_half += first_half
        return second_half


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
            nn.Linear(self.configs.dim_embeddings, self.configs.dim_embeddings * 4),
            nn.GELU(),
            nn.Linear(self.configs.dim_embeddings * 4, self.configs.dim_embeddings),
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

    def get_attention_output(self, X: torch.Tensor, mask):
        if self.configs.transformer_layer == TransformerLayerType.SIMPLE:
            attn_output, _ = self.self_attention(X, X, X, attn_mask=mask)
            return attn_output
        elif self.configs.transformer_layer == TransformerLayerType.SIMPLE_NO_ATTENTION:
            return torch.zeros_like(X, device=X.device)
        else:
            raise Exception(f"Unknown attention type {self.configs.attention_type}")


class SimpleAttentionAtHomeTransformerLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.configs = config
        self.self_attention = SimpleMultiHeadAttention(
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
        attn_output = self.self_attention(X, X, X, mask)
        return self.layer_norm_out(self.module(self.layer_norm_in(X + attn_output)))


# https://arxiv.org/pdf/2512.24880
# Deepseek builds on top of the hyperConnection, but adds a step to normalize
def sinkhorn(H, iterations=20):
    P = torch.exp(H)
    for _ in range(iterations):
        P = P / P.sum(dim=-1, keepdim=True)
        P = P / P.sum(dim=-2, keepdim=True)
    return P


class mHyperConnection(nn.Module):
    def __init__(self, n: int = 4, layer_idx: int = 0):
        super().__init__()
        self.n = n
        alpha = torch.zeros(n, n + 1)
        alpha[layer_idx % n, 0] = 1.0
        alpha[:, 1:] = torch.eye(n)
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(torch.ones(n))

    def forward(self, H: torch.Tensor, layer_fn) -> torch.Tensor:
        # H: (B, S, n, D)
        # (n, 1)
        A_m = self.alpha[:, :1]
        # (n, n)
        A_r = self.alpha[:, 1:]
        A_r = sinkhorn(A_r)
        # Bound beta with sigmoid
        beta = torch.sigmoid(self.beta)

        # Weighted sum of copies -> layer input
        # (B, S, D)
        h_in = torch.einsum("bsnd,nk->bskd", H, A_m).squeeze(2)
        h_out = layer_fn(h_in)
        # Distribute output + residual
        out = h_out.unsqueeze(2) * beta.view(1, 1, self.n, 1) + torch.einsum(
            "nm,bsmd->bsnd", A_r, H
        )

        return out


# https://arxiv.org/pdf/2409.19606#page=14
# https://arxiv.org/pdf/2409.19606#page=28
# https://arxiv.org/pdf/2409.19606#page=29
class HyperConnection(nn.Module):
    def __init__(self, n: int = 4, layer_idx: int = 0):
        super().__init__()
        self.n = n
        self.beta = nn.Parameter(torch.ones(n))

        alpha = torch.zeros(n, n + 1)
        alpha[layer_idx % n, 0] = 1.0
        alpha[:, 1:] = torch.eye(n)
        self.alpha = nn.Parameter(alpha)

    def forward(self, H: torch.Tensor, layer_fn) -> torch.Tensor:
        # H: (B, S, n, D)
        # (n, 1)
        A_m = self.alpha[:, :1]
        # (n, n)
        A_r = self.alpha[:, 1:]

        # Weighted sum of copies -> layer input
        # (B, S, D)
        h_in = torch.einsum("bsnd,nk->bskd", H, A_m).squeeze(2)
        h_out = layer_fn(h_in)
        # Distribute output + residual
        out = h_out.unsqueeze(2) * self.beta.view(1, 1, self.n, 1) + torch.einsum(
            "nm,bsmd->bsnd", A_r, H
        )

        return out


# Basically the same as LLAMA
# (todo: maybe we should just replace it with the LLAMA model, only diff is bias is just turned off)
# https://arxiv.org/pdf/2501.00656#page=5
class OlmoFeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.l1 = nn.Linear(
            config.dim_embeddings,
            config.feed_forward_layer,
            bias=False,
        )
        self.l2 = nn.Linear(
            config.feed_forward_layer,
            config.dim_embeddings,
            bias=False,
        )
        self.gate = nn.Linear(
            config.dim_embeddings,
            config.feed_forward_layer,
            bias=False,
        )

    def forward(self, X):
        return self.l2(F.silu(self.l1(X) * self.gate(X)))


class OlmoTransformerLayer(nn.Module):
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
        self.linear = OlmoFeedForward(config)

    def forward(self, X, mask):
        first_half = self.rope_attention(self.norm_1(X), mask)
        first_half += X
        second_half = self.linear(self.norm_2(first_half))
        second_half += first_half
        return second_half


class OlmoTransformerLayerHyperConnectivity(nn.Module):
    def __init__(self, config: Config, layer_idx: int, use_manifold_constraint=False):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.dim_embeddings)
        self.rope_attention = RoPEMultiheadAttention(
            config.dim_embeddings,
            config.num_attention_heads,
            config.sequence_length,
            gqa=False,
        )
        self.norm_2 = nn.RMSNorm(config.dim_embeddings)
        self.linear = OlmoFeedForward(config)

        if use_manifold_constraint:
            self.hc_attn = mHyperConnection(n=config.hc_n, layer_idx=layer_idx * 2)
            self.hc_mlp = mHyperConnection(n=config.hc_n, layer_idx=layer_idx * 2 + 1)
        else:
            self.hc_attn = HyperConnection(n=config.hc_n, layer_idx=layer_idx * 2)
            self.hc_mlp = HyperConnection(n=config.hc_n, layer_idx=layer_idx * 2 + 1)

    def forward(self, H, mask=None):
        H = self.hc_attn(H, lambda x: self.rope_attention(self.norm_1(x), mask))
        H = self.hc_mlp(H, lambda x: self.linear(self.norm_2(x)))
        return H


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, layer: nn.TransformerDecoderLayer) -> None:
        super(TransformerDecoderWrapper, self).__init__()
        self.layer = layer

    def forward(self, x, mask):
        return self.layer(x, torch.zeros_like(x), tgt_mask=mask)


def get_transformer_layer(config: Config, layer_idx):
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
    elif config.transformer_layer == TransformerLayerType.SIMPLE_ATTENTION_AT_HOME:
        return SimpleAttentionAtHomeTransformerLayer(config)
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
    elif config.transformer_layer == TransformerLayerType.OLMO:
        return OlmoTransformerLayer(config)
    elif config.transformer_layer == TransformerLayerType.OLMO_HYPER_CONNECTIONS:
        return OlmoTransformerLayerHyperConnectivity(config, layer_idx)
    elif (
        config.transformer_layer
        == TransformerLayerType.OLMO_CONSTRAINED_HYPER_CONNECTIONS
    ):
        return OlmoTransformerLayerHyperConnectivity(
            config, layer_idx, use_manifold_constraint=True
        )
    else:
        raise Exception(f"unknown type {config.transformer_layer}")


class NnPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)

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
                get_transformer_layer(config, layer_idx)
                for layer_idx in range(self.config.num_transformer_layers)
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

        # Attention mask
        if self.config.masked_order == MaskOrder.TRIU:
            mask = torch.triu(
                torch.ones(
                    self.config.sequence_length,
                    self.config.sequence_length,
                ),
                diagonal=1,
            ).bool()
            self.register_buffer("mask", mask, persistent=True)
        elif self.config.masked_order == MaskOrder.TRIL:
            mask = torch.tril(
                torch.ones(
                    self.config.sequence_length,
                    self.config.sequence_length,
                ),
                diagonal=0,
            ).bool()
            self.register_buffer("mask", mask, persistent=True)
        else:
            self.register_buffer("mask", None, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2, f"X = {x.shape}"
        if self.config.transformer_layer in [
            TransformerLayerType.OLMO_HYPER_CONNECTIONS,
            TransformerLayerType.OLMO_CONSTRAINED_HYPER_CONNECTIONS,
        ]:
            x = self.embeddings(x)
            H = x.unsqueeze(2).expand(-1, -1, self.config.hc_n, -1).contiguous()
            H = self.dropout(H)
            for layer in self.transformer_layers:
                H = layer(H, self.mask)
            x = H.sum(dim=2)
            x = self.layer_norm(x)
            return self.output_layer(x)
        else:
            x = self.embeddings(x)
            x = self.positional_embeddings(x)
            x = self.dropout(x)
            for layer in self.transformer_layers:
                x = layer(x, self.mask)
            # Output
            x = self.layer_norm(x)
            return self.output_layer(x)

    def embed(self, x):
        hidden = self.get_hidden_states(x)
        emb = hidden.mean(dim=1)
        return F.normalize(emb, p=2, dim=-1)

    def get_hidden_states(self, x: torch.Tensor):
        assert len(x.shape) == 2, f"X = {x.shape}"
        x = self.embeddings(x)
        x = self.positional_embeddings(x)
        x = self.dropout(x)
        for layer in self.transformer_layers:
            x = layer(x, self.mask)
        x = self.layer_norm(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        input_tokens: torch.Tensor,
        num_tokens_generate,
        sampler,
        end_token_id=None,
    ):
        output = input_tokens
        output = output.unsqueeze(0)

        for _ in range(num_tokens_generate):
            logits = self.forward(output[:, -self.config.sequence_length :])
            last_token = logits[:, -1, :]
            next_token = sampler(last_token).unsqueeze(0)
            output = torch.cat((output, next_token), dim=1)
            # Early stopping
            if end_token_id == next_token:
                break
        return output[0].tolist()

    def beam_search(self, input_ids, max_len=50, beam_width=3, end_token_id=None):
        # (score, seq, is_finished)
        beams = [(0.0, input_ids, False)]

        for _ in range(max_len):
            candidates = []

            for score, seq, finished in beams:
                if finished:
                    candidates.append((score, seq, True))
                    continue

                # logits = self.forward(seq.unsqueeze(0))[:, -1, :]
                logits = self.forward(seq[-self.config.sequence_length :].unsqueeze(0))[
                    :, -1, :
                ]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_ids = log_probs.topk(beam_width)

                for log_p, token_id in zip(top_log_probs, top_ids):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                    is_eos = (
                        end_token_id is not None and token_id.item() == end_token_id
                    )
                    candidates.append((score + log_p.item(), new_seq, is_eos))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

            # All done?
            if all(b[2] for b in beams):
                break

        return beams[0][1].tolist()
