import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.enable_gqa = not (num_groups is None)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        q: torch.Tensor = self.q_proj(query).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k: torch.Tensor = self.k_proj(key).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v: torch.Tensor = self.v_proj(value).view(batch_size, seq_len, self.num_groups, self.head_dim)
        
        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
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
        assert num_query_heads % num_groups == 0, "num_query_heads must be divisible by num_groups"        

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_groups, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_groups, self.head_dim)
        
        # Transpose to [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use PyTorch's scaled_dot_product_attention with GQA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=True  # Enable Grouped Query Attention
        )
        
        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out), None

# https://arxiv.org/pdf/2503.10622
class DyT(nn.Module):
    def __init__(self, C, init_α = 0.5):
        super().__init__()
        self.α = nn.Parameter(torch.ones(1) * init_α)
        self.γ = nn.Parameter(torch.ones(C))
        self.β = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        x = F.tanh(self.α * x)
        return self.γ * x + self.β
