"""Pooling: reduce 25,600 Gaussians → 256 tokens for decoder cross-attention."""
import torch, torch.nn as nn, torch.nn.functional as F
from .gaussian_utils import farthest_point_sampling

class GaussianFPSPooling(nn.Module):
    def __init__(self, num_tokens=256, in_dims=128, out_dims=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(in_dims, out_dims)
    def forward(self, features, means):
        B, N, D = features.shape
        K = min(self.num_tokens, N)
        indices, _ = farthest_point_sampling(means, K)
        sampled = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1,-1,D))
        return self.proj(sampled)

class GaussianLearnedPooling(nn.Module):
    def __init__(self, num_tokens=256, in_dims=128, out_dims=256, num_heads=8):
        super().__init__()
        self.pool_queries = nn.Parameter(torch.randn(1, num_tokens, out_dims)*0.02)
        self.cross_attn = nn.MultiheadAttention(out_dims, num_heads, kdim=in_dims, vdim=in_dims, batch_first=True)
        self.norm = nn.LayerNorm(out_dims)
    def forward(self, features, means):
        B = features.shape[0]
        q = self.pool_queries.expand(B,-1,-1)
        out, _ = self.cross_attn(q, features, features)
        return self.norm(out + q)

def build_gaussian_pooling(method="fps", num_tokens=256, in_dims=128, out_dims=256, **kw):
    if method == "fps": return GaussianFPSPooling(num_tokens, in_dims, out_dims)
    elif method == "learned": return GaussianLearnedPooling(num_tokens, in_dims, out_dims)
    else: raise ValueError(f"Unknown pooling: {method}")
