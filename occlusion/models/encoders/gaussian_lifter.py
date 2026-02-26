"""Voxel-to-Gaussian (V2G) Initialization from LiDAR point clouds."""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Tuple

class VoxelToGaussianLifter(nn.Module):
    def __init__(self, num_gaussians=25600, embed_dims=128, voxel_size=None,
                 point_cloud_range=None, max_points_per_voxel=10, num_classes=17):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        if voxel_size is None: voxel_size = [0.075, 0.075, 0.2]
        if point_cloud_range is None: point_cloud_range = [-40,-40,-1,40,40,5.4]
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.max_points = max_points_per_voxel
        self.grid_size = [int((point_cloud_range[3+i]-point_cloud_range[i])/voxel_size[i]) for i in range(3)]
        prop_dim = 11 + num_classes
        self.default_props = nn.Parameter(torch.zeros(1,1,prop_dim))
        nn.init.normal_(self.default_props[:,:,:3], std=0.1)
        self.default_props.data[:,:,6] = 1.0  # identity quaternion w
        nn.init.constant_(self.default_props[:,:,10], -2.0)
        self.default_features = nn.Parameter(torch.randn(1,1,embed_dims)*0.02)
        self.voxel_to_feat = nn.Sequential(nn.Linear(9, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, embed_dims))
        self.voxel_to_props = nn.Sequential(nn.Linear(9, 64), nn.ReLU(True), nn.Linear(64, prop_dim))

    def voxelize_points(self, points):
        device = points.device
        pc = torch.tensor(self.pc_range, device=device, dtype=torch.float32)
        vs = torch.tensor(self.voxel_size, device=device, dtype=torch.float32)
        mask = ((points[:,0]>=pc[0])&(points[:,0]<pc[3])&(points[:,1]>=pc[1])&
                (points[:,1]<pc[4])&(points[:,2]>=pc[2])&(points[:,2]<pc[5]))
        points = points[mask]
        if points.shape[0] == 0:
            return torch.zeros(0,9,device=device), torch.zeros(0,3,device=device), 0
        coords = ((points[:,:3]-pc[:3])/vs).long()
        gy, gz = self.grid_size[1], self.grid_size[2]
        voxel_idx = coords[:,0]*(gy*gz)+coords[:,1]*gz+coords[:,2]
        unique_idx, inverse = voxel_idx.unique(return_inverse=True)
        nv = unique_idx.shape[0]
        xyz = points[:,:3]
        intensity = points[:,3:4] if points.shape[1]>3 else torch.zeros_like(xyz[:,:1])
        vsum = torch.zeros(nv,3,device=device); vsum.scatter_add_(0,inverse.unsqueeze(-1).expand(-1,3),xyz)
        vcnt = torch.zeros(nv,1,device=device); vcnt.scatter_add_(0,inverse.unsqueeze(-1),torch.ones_like(intensity))
        vmean = vsum/vcnt.clamp(min=1)
        sq_diff = (xyz-vmean[inverse])**2
        vsq = torch.zeros(nv,3,device=device); vsq.scatter_add_(0,inverse.unsqueeze(-1).expand(-1,3),sq_diff)
        vstd = (vsq/vcnt.clamp(min=1)).sqrt()
        isum = torch.zeros(nv,1,device=device); isum.scatter_add_(0,inverse.unsqueeze(-1),intensity)
        imean = isum/vcnt.clamp(min=1)
        isq = torch.zeros(nv,1,device=device); isq.scatter_add_(0,inverse.unsqueeze(-1),(intensity-imean[inverse])**2)
        istd = (isq/vcnt.clamp(min=1)).sqrt()
        stats = torch.cat([vmean, vstd, vcnt.clamp(max=self.max_points)/self.max_points, imean, istd], dim=-1)
        return stats, vmean, nv

    def forward(self, points_batch):
        B = len(points_batch); device = points_batch[0].device
        N = self.num_gaussians; prop_dim = 11+self.num_classes
        all_p, all_f = [], []
        for b in range(B):
            stats, means, nv = self.voxelize_points(points_batch[b])
            if nv == 0:
                p = self.default_props.expand(1,N,-1).squeeze(0)
                f = self.default_features.expand(1,N,-1).squeeze(0)
            elif nv >= N:
                perm = torch.randperm(nv, device=device)[:N]
                p = self.voxel_to_props(stats[perm]); f = self.voxel_to_feat(stats[perm])
            else:
                pv = self.voxel_to_props(stats); fv = self.voxel_to_feat(stats)
                pad = N-nv
                p = torch.cat([pv, self.default_props.expand(1,pad,-1).squeeze(0)])
                f = torch.cat([fv, self.default_features.expand(1,pad,-1).squeeze(0)])
            all_p.append(p); all_f.append(f)
        return torch.stack(all_p), torch.stack(all_f)
