"""GaussianOccEncoder: Iterative refinement of 3D Gaussians via sparse conv + 3D deformable attention."""
import torch, torch.nn as nn
from typing import Tuple
from .deformable_attn_3d import DeformableAttention3D, SparseConv3DBlock
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.gaussian_utils import GaussianParameterHead

class GaussianRefinementBlock(nn.Module):
    def __init__(self, embed_dims=128, num_heads=8, num_levels=4, num_points=8, num_3d_ref_points=4, num_classes=17, dropout=0.1):
        super().__init__()
        self.sparse_conv = SparseConv3DBlock(embed_dims)
        self.deform_attn = DeformableAttention3D(embed_dims, num_heads, num_levels, num_points, num_3d_ref_points, dropout)
        self.attn_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, embed_dims*4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dims*4, embed_dims), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.prop_head = GaussianParameterHead(embed_dims, num_classes, True)

    def forward(self, query, gaussian_props, ms_feats, depth_maps, lidar2img, img_shape):
        means = gaussian_props[...,:3]
        query = self.sparse_conv(query, means)
        attn_out = self.deform_attn(query, means, ms_feats, depth_maps, lidar2img, img_shape)
        query = self.attn_norm(query + attn_out)
        query = self.ffn_norm(query + self.ffn(query))
        gaussian_props = gaussian_props + self.prop_head(query)
        return query, gaussian_props

class GaussianOccEncoder(nn.Module):
    def __init__(self, num_blocks=6, embed_dims=128, num_heads=8, num_levels=4, num_points=8, num_3d_ref_points=4, num_classes=17, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([GaussianRefinementBlock(embed_dims, num_heads, num_levels, num_points, num_3d_ref_points, num_classes, dropout) for _ in range(num_blocks)])

    def forward(self, query, gaussian_props, ms_feats, depth_maps, lidar2img, img_shape):
        intermediates = []
        for block in self.blocks:
            query, gaussian_props = block(query, gaussian_props, ms_feats, depth_maps, lidar2img, img_shape)
            intermediates.append(gaussian_props)
        return query, gaussian_props, intermediates
