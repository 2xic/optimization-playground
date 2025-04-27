import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_groups=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_groups = num_query_heads if num_groups is None else num_groups
        self.head_dim = embed_dim // num_query_heads

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
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=self.enable_gqa,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None


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
        q = self.q_proj(query).view(
            batch_size, seq_len, self.num_query_heads, self.head_dim
        )
        k = self.k_proj(key).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_groups, self.head_dim)

        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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


# https://arxiv.org/pdf/2503.10622
class DyT(nn.Module):
    def __init__(self, C, init_α=0.5):
        super().__init__()
        self.α = nn.Parameter(torch.ones(1, device=DEVICE) * init_α)
        self.γ = nn.Parameter(torch.ones(C, device=DEVICE))
        self.β = nn.Parameter(torch.zeros(C, device=DEVICE))

    def forward(self, x):
        x = F.tanh(self.α * x)
        return self.γ * x + self.β
