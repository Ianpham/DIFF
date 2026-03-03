"""Gaussian-to-Voxel splatting for occupancy prediction."""
# import torch, torch.nn as nn, torch.nn.functional as F

# class GaussianSplattingDecoder(nn.Module):
#     def __init__(self, occ_size=None, point_cloud_range=None, num_classes=17, embed_dims=128, neighbor_radius=3.0):
#         super().__init__()
#         if occ_size is None: occ_size = [200,200,16]
#         if point_cloud_range is None: point_cloud_range = [-40,-40,-1,40,40,5.4]
#         self.occ_size = occ_size; self.num_classes = num_classes; self.neighbor_radius = neighbor_radius
#         xs = torch.linspace(point_cloud_range[0], point_cloud_range[3], occ_size[0])
#         ys = torch.linspace(point_cloud_range[1], point_cloud_range[4], occ_size[1])
#         zs = torch.linspace(point_cloud_range[2], point_cloud_range[5], occ_size[2])
#         grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
#         self.register_buffer("voxel_coords", grid.reshape(-1, 3))
#         self.refine = nn.Sequential(nn.Linear(num_classes, num_classes*2), nn.ReLU(True), nn.Linear(num_classes*2, num_classes))

#     def forward(self, gaussian_props, gaussian_feats=None):
#         B, N, _ = gaussian_props.shape; V = self.voxel_coords.shape[0]
#         X, Y, Z = self.occ_size; C = self.num_classes
#         means = gaussian_props[...,:3]; scales = F.softplus(gaussian_props[...,3:6])
#         opacity = torch.sigmoid(gaussian_props[...,10:11]); semantics = gaussian_props[...,11:11+C]
#         chunk = min(V, 8000); parts = []
#         for s in range(0, V, chunk):
#             e = min(s+chunk, V); vox = self.voxel_coords[s:e]
#             diff = vox.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
#             sq = (diff**2).sum(-1); mask = sq < self.neighbor_radius**2
#             inv_s = 1.0/(scales**2).clamp(min=1e-6)
#             mahal = (diff**2 * inv_s.unsqueeze(1)).sum(-1)
#             density = torch.exp(-0.5*mahal)
#             w = density * opacity.squeeze(-1).unsqueeze(1) * mask.float()
#             ws = w.sum(-1, keepdim=True).clamp(min=1e-6)
#             parts.append(torch.bmm(w/ws, semantics))
#         occ = torch.cat(parts, dim=1)
#         return self.refine(occ).reshape(B, X, Y, Z, C)

"""
Implement Eq 1-3 from Gaussianformer3D (arxiv: 2505.10685):
Implements Eq 1-3 from GaussianFormer3D (arXiv:2505.10685):
    g(x; G) = sigma * exp(-0.5 * (x-m)^T Sigma^{-1} (x-m)) * c
    Sigma = R @ S @ S^T @ R^T
    o(x) = sum_{i in N_g(x)} g_i(x)

where N_g(x) is the set of neighboring Gaussians at voxel x,
determined by a spatial radius search for efficiency.

Also supports an optional fixed "empty Gaussian" covering the
entire scene for the empty class (from GaussianFormer-2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from utils.gaussian_utils import quaternion_to_rotation_matrix


class GaussianSplattingDecoder(nn.Module):
    """
    Gaussian-to-Voxel splatting.

    No learned parameters in the core splatting — this is a pure
    geometric projection from Gaussian properties to voxel logits.

    Args:
        occ_size: [X, Y, Z] voxel grid dimensions.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        num_classes: |C| semantic classes.
        embed_dims: Gaussian feature dimension (unused in pure splatting,
            kept for interface compatibility).
        neighbor_radius: spatial radius for neighbor search (meters).
            Only Gaussians within this distance contribute to a voxel.
        chunk_size: number of voxels to process at once (memory control).
        use_empty_gaussian: whether to add a fixed large Gaussian for
            the empty class.
        empty_class_idx: which class index is "empty" (default 0). 

    """
    def __init__(
            self,
            occ_size: Optional[List[int]] = None,
            point_cloud_range: Optional[List[float]] = None,
            num_classes: int = 17, #based on Openscene
            embed_dims: int = 128,
            neighbor_radius: float = 4.0,
            chunk_size: int = 4000,
            use_empty_gaussian: bool = True,
            empty_class_idx: int = 0,
    ):
        super().__init__()

        if occ_size is None:
            occ_size = [200, 200, 16]

        if point_cloud_range is None:
            point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        
        self.occ_size = occ_size
        self.num_classes = num_classes
        self.neighbor_radius = neighbor_radius
        self.neighbor_radius_sq = neighbor_radius ** 2
        self.chunk_size = chunk_size
        self.use_empty_gaussian = use_empty_gaussian
        self.empty_class_idx = empty_class_idx

        # precompute voxel center coordinates
        xs = torch.linspace(
            point_cloud_range[0], point_cloud_range[3], occ_size[0]
        )
        ys = torch.linspace(
            point_cloud_range[1], point_cloud_range[4], occ_size[1]
        )
        zs = torch.linspace(
            point_cloud_range[2], point_cloud_range[5], occ_size[2]
        )

        grid = torch.stack(
            torch.meshgrid(xs, ys, zs, indexing="ij"), dim = -1
        )

        self.register_buffer("voxel_coords", grid.reshape(-1, 3))
        # voxel_coords: (V, 3) where V = X * Y * Z

        # empty Gaussian: large scale covering the whole scene, low opacity,
        # semantics peaked at empty class

        if use_empty_gaussian:
            scene_center = torch.tensor([
                (point_cloud_range[0] + point_cloud_range[3]) / 2,
                (point_cloud_range[1] + point_cloud_range[4]) / 2,
                (point_cloud_range[2] + point_cloud_range[5]) / 2,
            ])
            scene_extent = torch.tensor([
                point_cloud_range[3] - point_cloud_range[0],
                point_cloud_range[4] - point_cloud_range[1],
                point_cloud_range[5] - point_cloud_range[2],
            ])
            self.register_buffer("empty_mean", scene_center)
            self.register_buffer("empty_scale", scene_extent / 2)
            # precompute inverse covariance for empty Gaussian
            # identity rotation, diagonal covariance = scale^2
            empty_cov_inv = torch.diag(1.0 / (self.empty_scale ** 2).clamp(min=1e-6))
            self.register_buffer("empty_cov_inv", empty_cov_inv)

            empty_semantics = torch.zeros(num_classes)
            empty_semantics[empty_class_idx] = 5.0  # strong prior for empty
            self.register_buffer("empty_semantics", empty_semantics)
            self.empty_opacity = 0.1  # low but nonzero

    def _parse_gaussian_props(
        self, gaussian_props: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract and activate Gaussian properties.

        Args:
            gaussian_props: (B, N, 11 + C) raw properties.

        Returns:
            means: (B, N, 3)
            cov_inv: (B, N, 3, 3) inverse covariance matrices
            opacity: (B, N, 1) in [0, 1]
            semantics: (B, N, C) raw logits
            det_sqrt: (B, N) sqrt of determinant of Sigma (for normalization)
        """
        means = gaussian_props[..., :3]
        scales_raw = gaussian_props[..., 3:6]
        quats = gaussian_props[..., 6:10]
        opacity = torch.sigmoid(gaussian_props[..., 10:11])
        semantics = gaussian_props[..., 11:11 + self.num_classes]

        # activate scales
        scales = F.softplus(scales_raw).clamp(min=1e-4)  # (B, N, 3)

        # build covariance: Sigma = R @ S @ S^T @ R^T
        # for Mahalanobis distance we need Sigma^{-1}
        # Sigma^{-1} = R @ S^{-2} @ R^T (since S is diagonal)
        R = quaternion_to_rotation_matrix(quats)  # (B, N, 3, 3)

        # S^{-2} as diagonal
        inv_scales_sq = 1.0 / (scales ** 2).clamp(min=1e-8)  # (B, N, 3)
        inv_S_sq = torch.diag_embed(inv_scales_sq)  # (B, N, 3, 3)

        # Sigma^{-1} = R @ diag(1/s^2) @ R^T
        cov_inv = R @ inv_S_sq @ R.transpose(-1, -2)  # (B, N, 3, 3)

        # sqrt(det(Sigma)) for optional normalization
        # det(Sigma) = prod(s_i^2) = (prod s_i)^2
        det_sqrt = scales.prod(dim=-1)  # (B, N)

        return means, cov_inv, opacity, semantics, det_sqrt

    def _evaluate_gaussians(
        self,
        voxel_pos: torch.Tensor,
        means: torch.Tensor,
        cov_inv: torch.Tensor,
        opacity: torch.Tensor,
        semantics: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Gaussian contributions at voxel positions.

        Implements Eq 1 + 3: weighted sum of Gaussian evaluations.

        Args:
            voxel_pos: (V_chunk, 3) voxel center positions.
            means: (B, N, 3) Gaussian means.
            cov_inv: (B, N, 3, 3) inverse covariances.
            opacity: (B, N, 1) opacities.
            semantics: (B, N, C) semantic logits.

        Returns:
            occ_chunk: (B, V_chunk, C) semantic occupancy logits.
        """
        B, N, _ = means.shape
        V = voxel_pos.shape[0]
        C = self.num_classes

        # diff: (B, V, N, 3) = voxel_pos[v] - means[n]
        # voxel_pos: (V, 3) -> (1, V, 1, 3)
        # means: (B, N, 3) -> (B, 1, N, 3)
        diff = voxel_pos.unsqueeze(0).unsqueeze(2) - means.unsqueeze(1)
        # diff: (B, V, N, 3)

        # neighbor mask: squared Euclidean distance for quick filtering
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, V, N)
        neighbor_mask = dist_sq < self.neighbor_radius_sq  # (B, V, N)

        # Mahalanobis distance: d^2 = diff^T @ Sigma^{-1} @ diff
        # diff: (B, V, N, 3) -> (B, V, N, 3, 1)
        # cov_inv: (B, N, 3, 3) -> (B, 1, N, 3, 3)
        diff_col = diff.unsqueeze(-1)  # (B, V, N, 3, 1)
        cov_inv_exp = cov_inv.unsqueeze(1)  # (B, 1, N, 3, 3)

        # (B, V, N, 1, 3) @ (B, 1, N, 3, 3) @ (B, V, N, 3, 1) -> (B, V, N, 1, 1)
        mahal = diff.unsqueeze(-2) @ cov_inv_exp @ diff_col
        mahal = mahal.squeeze(-1).squeeze(-1)  # (B, V, N)

        # Gaussian density (unnormalized, Eq 1 without semantic part)
        density = torch.exp(-0.5 * mahal)  # (B, V, N)

        # apply neighbor mask and opacity
        # weight = sigma_i * G(x; m_i, Sigma_i) for i in N_g(x)
        weight = density * opacity.squeeze(-1).unsqueeze(1) * neighbor_mask.float()
        # weight: (B, V, N)

        # weighted sum of semantics (Eq 3)
        # weight: (B, V, N) @ semantics: (B, N, C) -> (B, V, C)
        occ_chunk = torch.bmm(weight, semantics)

        return occ_chunk

    def _evaluate_empty_gaussian(
        self, voxel_pos: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """Evaluate the fixed empty-space Gaussian at voxel positions.

        Args:
            voxel_pos: (V_chunk, 3)
            B: batch size.

        Returns:
            empty_contrib: (B, V_chunk, C) contribution from empty Gaussian.
        """
        V = voxel_pos.shape[0]
        diff = voxel_pos - self.empty_mean.unsqueeze(0)  # (V, 3)
        mahal = (diff @ self.empty_cov_inv * diff).sum(dim=-1)  # (V,)
        density = torch.exp(-0.5 * mahal)  # (V,)
        weight = self.empty_opacity * density  # (V,)

        # (V, 1) * (1, C) -> (V, C)
        contrib = weight.unsqueeze(-1) * self.empty_semantics.unsqueeze(0)
        # expand to batch: (B, V, C)
        return contrib.unsqueeze(0).expand(B, -1, -1)
    def _evaluate_gaussians_indexed(
        self,
        voxel_pos: torch.Tensor,
        means: torch.Tensor,
        cov_inv: torch.Tensor,
        opacity: torch.Tensor,
        semantics: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Gaussians using pre-computed neighbor indices.

        Instead of checking all N Gaussians per voxel, only evaluates
        the K neighbors found by spatial hashing.

        Args:
            voxel_pos: (V_chunk, 3).
            means: (B, N, 3).
            cov_inv: (B, N, 3, 3).
            opacity: (B, N, 1).
            semantics: (B, N, C).
            neighbor_indices: (B, V_chunk, K) indices into N.
            neighbor_mask: (B, V_chunk, K) bool validity.

        Returns:
            occ_chunk: (B, V_chunk, C).
        """
        B, V, K = neighbor_indices.shape
        C = self.num_classes

        # gather neighbor properties: (B, V, K, ...)
        idx_exp_3 = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        idx_exp_33 = neighbor_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3, 3)
        idx_exp_1 = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 1)
        idx_exp_C = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, C)

        n_means = torch.gather(
            means.unsqueeze(1).expand(-1, V, -1, -1), 2, idx_exp_3
        )  # (B, V, K, 3)
        n_cov_inv = torch.gather(
            cov_inv.unsqueeze(1).expand(-1, V, -1, -1, -1), 2, idx_exp_33
        )  # (B, V, K, 3, 3)
        n_opacity = torch.gather(
            opacity.unsqueeze(1).expand(-1, V, -1, -1), 2, idx_exp_1
        )  # (B, V, K, 1)
        n_semantics = torch.gather(
            semantics.unsqueeze(1).expand(-1, V, -1, -1), 2, idx_exp_C
        )  # (B, V, K, C)

        # diff: (B, V, K, 3)
        diff = voxel_pos.unsqueeze(0).unsqueeze(2) - n_means

        # Mahalanobis: diff^T @ Sigma^{-1} @ diff
        diff_col = diff.unsqueeze(-1)  # (B, V, K, 3, 1)
        mahal = (diff.unsqueeze(-2) @ n_cov_inv @ diff_col).squeeze(-1).squeeze(-1)
        # (B, V, K)

        density = torch.exp(-0.5 * mahal)
        weight = density * n_opacity.squeeze(-1) * neighbor_mask.float()
        # (B, V, K)

        # weighted sum: (B, V, K) @ (B, V, K, C) -> (B, V, C)
        occ_chunk = (weight.unsqueeze(-1) * n_semantics).sum(dim=2)

        return occ_chunk
    # def forward(
    #         self,
    #         gaussian_props: torch.Tensor,
    #         gaussian_feats: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     """
    #     Splat Gaussians to voxel grid for occupancy predictions
    #     Args:
    #         gaussian_props: (B, N, 11 + C) Gaussian properties after
    #             iterative refinement. Layout:
    #             [mean(3), scale(3), quat(4), opacity(1), semantics(C)]
    #         gaussian_feats: (B, N, D) Gaussian query features.
    #             Not used in pure splatting (kept for interface compat).

    #     Returns:
    #         occ_pred: (B, X, Y, Z, C) semantic occupancy logits.
    #     """
    #     B = gaussian_props.shape[0]
    #     V = self.voxel_coords.shape[0]
    #     X, Y, Z = self.occ_size
    #     C = self.num_classes

    #     means, cov_inv, opacity, semantics, _ = self._parse_gaussian_props(
    #         gaussian_props
    #     )

    #     # process in chunks to control memory
    #     # full: V voxels x N gaussians x B batch is huge

    #     parts = []
    #     for start in range(0, V, self.chunk_size):
    #         end = min(start + self.chunk_size, V)
    #         vox = self.voxel_coords[start:end] # (chunk, 3)

    #         chunk_occ = self._evaluate_empty_gaussian(
    #             vox, means, cov_inv, opacity, semantics
    #         )

    #         if self.use_empty_gaussian:
    #             chunk_occ = chunk_occ + self._evaluate_empty_gaussian(vox, B)
            
    #         parts.append(chunk_occ)
        
    #     occ = torch.cat(parts, dim = 1) # (B, V, C)
    #     return occ.reshape(B, X, Y, Z, C)
    def forward(
        self,
        gaussian_props: torch.Tensor,
        gaussian_feats: Optional[torch.Tensor] = None,
        point_cloud_range: Optional[List[float]] = None,
    ) -> torch.Tensor:
        B = gaussian_props.shape[0]
        V = self.voxel_coords.shape[0]
        X, Y, Z = self.occ_size
        C = self.num_classes

        means, cov_inv, opacity, semantics, _ = self._parse_gaussian_props(
            gaussian_props
        )

        if point_cloud_range is None:
            point_cloud_range = [
                self.voxel_coords[:, 0].min().item(),
                self.voxel_coords[:, 1].min().item(),
                self.voxel_coords[:, 2].min().item(),
                self.voxel_coords[:, 0].max().item(),
                self.voxel_coords[:, 1].max().item(),
                self.voxel_coords[:, 2].max().item(),
            ]

        # build spatial hash from current Gaussian positions
        from utils.spatial_hash import build_spatial_hash
        grid = build_spatial_hash(
            means, self.neighbor_radius, point_cloud_range,
        )

        # process in chunks
        parts = []
        for start in range(0, V, self.chunk_size):
            end = min(start + self.chunk_size, V)
            vox = self.voxel_coords[start:end]

            # query neighbors for this chunk
            indices, mask, _ = grid.query_neighbors_vectorized(
                vox, self.neighbor_radius_sq, means,
                max_neighbors=256,
            )

            chunk_occ = self._evaluate_gaussians_indexed(
                vox, means, cov_inv, opacity, semantics,
                indices, mask,
            )

            if self.use_empty_gaussian:
                chunk_occ = chunk_occ + self._evaluate_empty_gaussian(vox, B)

            parts.append(chunk_occ)

        occ = torch.cat(parts, dim=1)
        return occ.reshape(B, X, Y, Z, C)
    def forward_multiresolution(
            self,
            gaussian_props: torch.Tensor,
            target_occ_size: List[int],
            target_pc_range: Optional[List[float]] = None,
            gaussian_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Splat to an arbitrary resolution grid (paper Fig 6)
        
        Same Gaussians, different voxel grid. No retraining needed.

        Args:
            gaussian_props: (B, N, 11 + C).
            target_occ_size: [X', Y', Z'] target grid.
            target_pc_range: optional different spatial range.
                Defaults to self' range.
            gaussian_feats: unused, interface compat.

        Returns:
            occ_pred: (B, X', Y', Z', C)
        """
        if target_pc_range is None:
            # use stored range but recompute voxel centers at new resolution
            pc = self.voxel_coords.new_tensor([
                self.voxel_coords[:, 0].min(),
                self.voxel_coords[:, 1].min(),
                self.voxel_coords[:, 2].min(), 
                self.voxel_coords[:, 0].max(),
                self.voxel_coords[:, 1].max(),
                self.voxel_coords[:, 2].max()
            ])

        else:
            pc = gaussian_props.new_tensor(target_pc_range)

        Xt, Yt, Zt = target_occ_size
        xs = torch.linspace(pc[0].item(), pc[3].item(), Xt, device=gaussian_props.device)
        ys = torch.linspace(pc[1].item(), pc[4].item(), Yt, device=gaussian_props.device)
        zs = torch.linspace(pc[2].item(), pc[5].item(), Zt, device=gaussian_props.device)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)
        target_voxels = grid.reshape(-1, 3)

        B = gaussian_props.shape[0]
        V = target_voxels.shape[0]
        C = self.num_classes

        means, cov_inv, opacity, semantics, _ = self._parse_gaussian_props(
            gaussian_props
        )

        parts = []
        for start in range(0, V, self.chunk_size):
            end = min(start + self.chunk_size, V)
            vox = target_voxels[start:end]
            chunk_occ = self._evaluate_gaussians(
                vox, means, cov_inv, opacity, semantics,
            )
            if self.use_empty_gaussian:
                chunk_occ = chunk_occ + self._evaluate_empty_gaussian(vox, B)
            parts.append(chunk_occ)

        occ = torch.cat(parts, dim=1)
        return occ.reshape(B, Xt, Yt, Zt, C)    
    


