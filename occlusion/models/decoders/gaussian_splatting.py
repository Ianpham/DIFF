"""Gaussian-to-Voxel splatting for occupancy prediction."""
import torch, torch.nn as nn, torch.nn.functional as F

class GaussianSplattingDecoder(nn.Module):
    def __init__(self, occ_size=None, point_cloud_range=None, num_classes=17, embed_dims=128, neighbor_radius=3.0):
        super().__init__()
        if occ_size is None: occ_size = [200,200,16]
        if point_cloud_range is None: point_cloud_range = [-40,-40,-1,40,40,5.4]
        self.occ_size = occ_size; self.num_classes = num_classes; self.neighbor_radius = neighbor_radius
        xs = torch.linspace(point_cloud_range[0], point_cloud_range[3], occ_size[0])
        ys = torch.linspace(point_cloud_range[1], point_cloud_range[4], occ_size[1])
        zs = torch.linspace(point_cloud_range[2], point_cloud_range[5], occ_size[2])
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
        self.register_buffer("voxel_coords", grid.reshape(-1, 3))
        self.refine = nn.Sequential(nn.Linear(num_classes, num_classes*2), nn.ReLU(True), nn.Linear(num_classes*2, num_classes))

    def forward(self, gaussian_props, gaussian_feats=None):
        B, N, _ = gaussian_props.shape; V = self.voxel_coords.shape[0]
        X, Y, Z = self.occ_size; C = self.num_classes
        means = gaussian_props[...,:3]; scales = F.softplus(gaussian_props[...,3:6])
        opacity = torch.sigmoid(gaussian_props[...,10:11]); semantics = gaussian_props[...,11:11+C]
        chunk = min(V, 8000); parts = []
        for s in range(0, V, chunk):
            e = min(s+chunk, V); vox = self.voxel_coords[s:e]
            diff = vox.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
            sq = (diff**2).sum(-1); mask = sq < self.neighbor_radius**2
            inv_s = 1.0/(scales**2).clamp(min=1e-6)
            mahal = (diff**2 * inv_s.unsqueeze(1)).sum(-1)
            density = torch.exp(-0.5*mahal)
            w = density * opacity.squeeze(-1).unsqueeze(1) * mask.float()
            ws = w.sum(-1, keepdim=True).clamp(min=1e-6)
            parts.append(torch.bmm(w/ws, semantics))
        occ = torch.cat(parts, dim=1)
        return self.refine(occ).reshape(B, X, Y, Z, C)
