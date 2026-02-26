"""3D Deformable Attention (DFA3D wrapper) and Sparse Conv block."""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

class DeformableAttention3D(nn.Module):
    """3D deformable attention for Gaussian refinement from camera+LiDAR features."""
    def __init__(self, embed_dims=128, num_heads=8, num_levels=4, num_points=8, num_3d_ref_points=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims; self.num_heads = num_heads
        self.num_levels = num_levels; self.num_3d_ref_points = num_3d_ref_points
        self.offset_3d = nn.Linear(embed_dims, num_3d_ref_points*3)
        self.attention_weights = nn.Linear(embed_dims, num_heads*num_levels*num_3d_ref_points*num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.offset_3d.weight); nn.init.zeros_(self.offset_3d.bias)

    def forward(self, query, gaussian_means, multi_scale_features, depth_maps, lidar2img, img_shape):
        B, N, C = query.shape; num_cams = lidar2img.shape[1]
        offsets = self.offset_3d(query).reshape(B, N, self.num_3d_ref_points, 3)
        ref_3d = gaussian_means.unsqueeze(2) + offsets
        ones = torch.ones(*ref_3d.shape[:-1], 1, device=query.device)
        ref_homo = torch.cat([ref_3d, ones], dim=-1)
        ref_flat = ref_homo.reshape(B, N*self.num_3d_ref_points, 4)
        projected = torch.einsum("bcij,bnj->bcni", lidar2img, ref_flat)
        depth = projected[...,2:3].clamp(min=1e-3)
        pixel = projected[...,:2]/depth
        H, W = img_shape
        pixel[...,0] = 2.0*pixel[...,0]/W - 1.0
        pixel[...,1] = 2.0*pixel[...,1]/H - 1.0
        valid = (pixel[...,0].abs()<=1)&(pixel[...,1].abs()<=1)&(depth.squeeze(-1)>0)
        sampled_feats = []
        for lvl, feat in enumerate(multi_scale_features):
            BV = B*num_cams
            ff = feat.reshape(BV, C, feat.shape[-2], feat.shape[-1])
            cf = pixel.reshape(BV, -1, 1, 2)
            s = F.grid_sample(ff, cf, mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.squeeze(-1).reshape(B, num_cams, C, N, self.num_3d_ref_points)
            sampled_feats.append(s)
        sampled = torch.stack(sampled_feats, dim=2)
        vm = valid.float().reshape(B, num_cams, N, self.num_3d_ref_points)
        vm = vm / vm.sum(1, keepdim=True).clamp(min=1)
        agg = (sampled * vm.unsqueeze(2).unsqueeze(3)).sum(1).mean(dim=[1,-1]).permute(0,2,1)
        return self.output_proj(self.dropout(agg))

class SparseConv3DBlock(nn.Module):
    """Sparse conv substitute: MLP-based local feature aggregation on Gaussian positions."""
    def __init__(self, embed_dims=128):
        super().__init__()
        self.local_mlp = nn.Sequential(nn.Linear(embed_dims+3, embed_dims), nn.BatchNorm1d(embed_dims), nn.ReLU(True), nn.Linear(embed_dims, embed_dims))
        self.norm = nn.LayerNorm(embed_dims)
    def forward(self, features, positions):
        B, N, D = features.shape
        out = self.local_mlp(torch.cat([features, positions], -1).reshape(B*N, D+3)).reshape(B, N, D)
        return self.norm(features + out)
