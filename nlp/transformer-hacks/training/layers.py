import torch
import torch.nn as nn
import torch.nn.functional as F

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
# https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms
# TODO: this should implement kv-cache
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        # eq: 1 https://arxiv.org/pdf/1706.03762
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        attn = query @ key.transpose(-1, -2)
        attn = attn / torch.sqrt(torch.tensor(key.shape[-1]))
        L, S = query.size(-2), key.size(-2)
        attn_mask_tensor = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if attn_mask is not None:
            attn_mask_tensor.masked_fill_(attn_mask.logical_not(), float("-inf"))
        attn = attn + attn_mask_tensor
        attn = torch.softmax(attn, dim=-1)
        context = attn @ value
        return context


"""
Read more 
- https://benjaminwarner.dev/2023/07/01/attention-mechanism
- https://pytorch.org/blog/flexattention/
"""


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_groups=None, dropout_p=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_groups = num_query_heads if num_groups is None else num_groups
        self.head_dim = embed_dim // num_query_heads
        self.dropout_p = dropout_p

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_groups * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.enable_gqa = num_groups is not None

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        # Project to Q, K, V
        q: torch.Tensor = self.q_proj(query).view(
            batch_size, seq_len, self.num_query_heads, self.head_dim
        )
        k: torch.Tensor = self.k_proj(key).view(
            batch_size, seq_len, self.num_groups, self.head_dim
        )
        v: torch.Tensor = self.v_proj(value).view(
            batch_size, seq_len, self.num_groups, self.head_dim
        )

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
            enable_gqa=self.enable_gqa,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None


# GroupedQueryAttention
class SimpleGQA(MultiheadAttention):
    def __init__(self, embed_dim, num_query_heads, num_groups):
        super().__init__(
            embed_dim,
            num_query_heads,
            num_groups,
        )
        assert num_query_heads % num_groups == 0, (
            "num_query_heads must be divisible by num_groups"
        )

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        # Project to Q, K, V
        q, k, v = self._project(query, key, value)

        # Use PyTorch's scaled_dot_product_attention with GQA
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True,  # Enable Grouped Query Attention
        )

        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None

    def _project(self, query, key, value):
        batch_size, seq_len, _ = query.size()
        # Project to Q, K, V
        q = self.q_proj(query).view(
            batch_size, seq_len, self.num_query_heads, self.head_dim
        )
        k = self.k_proj(key).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_groups, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v


# QK- NORM https://magazine.sebastianraschka.com/i/168650848/qk-norm
class SimpleGQKV(SimpleGQA):
    def __init__(self, embed_dim, num_query_heads, num_groups):
        super().__init__(
            embed_dim,
            num_query_heads,
            num_groups,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        q, k, v = self._project(query, key, value)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Use PyTorch's scaled_dot_product_attention with GQA
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True,  # Enable Grouped Query Attention
        )

        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None


# https://arxiv.org/abs/2405.04434
# https://medium.com/@zaiinn440/coding-deepseek-v2-from-scratch-in-pytorch-06dd89917067


# The idea is to use a latent dimension to compress the keys and values.
# TODO: Add ROPE
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, latent_dim=64, num_groups=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_groups = num_query_heads if num_groups is None else num_groups
        self.head_dim = embed_dim // num_query_heads
        self.latent_dim = latent_dim

        # Latent space projections
        self.q_latent_proj = nn.Linear(embed_dim, latent_dim)
        self.k_latent_proj = nn.Linear(embed_dim, latent_dim)

        # Projections from latent space to query/key/value
        self.q_proj = nn.Linear(latent_dim, embed_dim)
        self.k_proj = nn.Linear(latent_dim, self.num_groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_groups * self.head_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.enable_gqa = num_groups is not None

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()

        # Project to latent space
        q_latent = self.q_latent_proj(query)
        k_latent = self.k_latent_proj(key)

        # Project from latent space to attention space
        q = self.q_proj(q_latent).view(
            batch_size, seq_len, self.num_query_heads, self.head_dim
        )
        k = self.k_proj(k_latent).view(
            batch_size, seq_len, self.num_groups, self.head_dim
        )
        v = self.v_proj(value).view(batch_size, seq_len, self.num_groups, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=self.enable_gqa,
        )

        # Reshape and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


class BidirectionalAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_query_heads, num_groups=None):
        super().__init__(embed_dim, num_query_heads, num_groups)

    def forward(self, query, key, value, mask):
        mask = torch.ones_like(mask)
        results, _ = super().forward(query, key, value, mask)
        return results


# https://arxiv.org/pdf/2503.10622
class DyT(nn.Module):
    # 0.5 is good for non lmm
    # For LLM -> https://arxiv.org/pdf/2503.10622#page=11
    def __init__(self, C, init_α=0.8):
        super().__init__()
        #      self.α = nn.Parameter(torch.ones(1, device=DEVICE) * init_α)
        #       self.γ = nn.Parameter(torch.ones(C, device=DEVICE))
        #        self.β = nn.Parameter(torch.zeros(C, device=DEVICE))
        self.α = nn.Parameter(torch.ones(1) * init_α)
        self.γ = nn.Parameter(torch.ones(C))
        self.β = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        x = F.tanh(self.α * x)
        return self.γ * x + self.β
