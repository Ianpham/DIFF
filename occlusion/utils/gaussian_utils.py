"""Gaussian property utilities: covariance, splatting, quaternions, FPS."""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

def quaternion_to_rotation_matrix(q):
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)

def build_covariance_3d(scale, rotation):
    s = F.softplus(scale)
    S = torch.diag_embed(s)
    R = quaternion_to_rotation_matrix(rotation)
    return R @ S @ S.transpose(-1,-2) @ R.transpose(-1,-2)

def gaussian_properties_from_raw(raw, num_classes=17):
    mean=raw[...,:3]; scale=raw[...,3:6]; rotation=raw[...,6:10]
    opacity=torch.sigmoid(raw[...,10:11]); semantics=raw[...,11:11+num_classes]
    return dict(mean=mean, scale=scale, rotation=rotation, opacity=opacity,
                semantics=semantics, covariance=build_covariance_3d(scale,rotation))

class GaussianParameterHead(nn.Module):
    def __init__(self, embed_dims=128, num_classes=17, include_opacity=True):
        super().__init__()
        out_dim = 3+3+4+(1 if include_opacity else 0)+num_classes
        self.head = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, out_dim))
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1); nn.init.zeros_(m.bias)
    def forward(self, features):
        return self.head(features)

def farthest_point_sampling(points, num_samples):
    B, N, _ = points.shape; device = points.device
    indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float("inf"), device=device)
    batch_idx = torch.arange(B, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    for i in range(num_samples):
        indices[:, i] = farthest
        centroid = points[batch_idx, farthest].unsqueeze(1)
        dist = ((points - centroid)**2).sum(-1)
        distances = torch.min(distances, dist)
        farthest = distances.argmax(-1)
    sampled = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1,-1,3))
    return indices, sampled
