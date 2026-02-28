"""Voxel-to-Gaussian (V2G) Initialization from LiDAR point clouds.

Follows GaussianFormer3D (arXiv:2505.10685), Section 3.2, Eq. 4,
with a full VoxelNet-style Voxel Feature Encoding (VFE) layer stack
instead of simple mean pooling.

VFE reference: Zhou & Tuzel, "VoxelNet: End-to-End Learning for
Point Cloud Based 3D Object Detection", CVPR 2018.

Per-voxel encoding pipeline:
  1. Gather raw points per voxel (padded to max_points_per_voxel T).
  2. Augment each point with centroid-relative offsets:
       [x, y, z, intensity, x-x̄, y-ȳ, z-z̄]   (7-dim)
  3. Pass through stacked VFE layers:
       Linear → BN → ReLU → concat(point_feat, max_pool) → repeat
  4. Final max-pool across points → voxel feature vector.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


# ======================================================================
# VFE Layer (VoxelNet building block)
# ======================================================================
class VFELayer(nn.Module):
    """Single Voxel Feature Encoding layer.

    For each voxel:
      point_feat = ReLU(BN(Linear(input)))        (N_v, T, c_out)
      voxel_feat = max_pool over T dim            (N_v, 1, c_out)
      output     = cat(point_feat, voxel_feat)    (N_v, T, 2*c_out)

    Args:
        c_in: input channel dim per point
        c_out: output channel dim per point (before concat → 2*c_out)
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (N_v, T, c_in) — per-point features, zero-padded
            mask: (N_v, T, 1)    — 1 for real points, 0 for padding

        Returns:
            out:  (N_v, T, 2 * c_out)
        """
        nv, T, _ = x.shape

        # shared MLP: Linear → BN → ReLU
        h = self.linear(x)                          # (N_v, T, c_out)
        # BN expects (N, C) or (N, C, L); reshape for batch norm
        h = h.reshape(-1, h.shape[-1])              # (N_v*T, c_out)
        h = self.bn(h)
        h = self.relu(h)
        h = h.reshape(nv, T, -1)                    # (N_v, T, c_out)

        h = h * mask                                # zero out padding

        # element-wise max pool across points → voxel-level feature
        # replace padded zeros with -inf before max
        pool_input = h.clone()
        pool_input[mask.squeeze(-1) == 0] = float("-inf")
        voxel_feat = pool_input.max(dim=1, keepdim=True).values  # (N_v, 1, c_out)
        voxel_feat = voxel_feat.expand_as(h)        # (N_v, T, c_out)

        # concatenate point-wise feat with locally-aggregated voxel feat
        out = torch.cat([h, voxel_feat], dim=-1)    # (N_v, T, 2*c_out)
        return out * mask


# ======================================================================
# Full Voxel Feature Encoder (stacked VFE layers + final max pool)
# ======================================================================
class VoxelFeatureEncoder(nn.Module):
    """Stack of VFE layers followed by a final max-pool.

    Default architecture (matching VoxelNet):
        VFE(7, 32) → VFE(64, 64) → Linear(128, out_channels) → max_pool

    The input per-point feature is 7-dim:
        [x, y, z, intensity, x - x̄, y - ȳ, z - z̄]

    Args:
        in_channels: raw point feature dim (default 7).
        vfe_channels: list of (c_in, c_out) for each VFE layer.
            After each VFE, output dim = 2 * c_out which feeds the next.
        out_channels: final voxel feature dimension.
        max_points_per_voxel: T — max number of points kept per voxel.
    """

    def __init__(
        self,
        in_channels: int = 7,
        vfe_channels: Optional[List[Tuple[int, int]]] = None,
        out_channels: int = 128,
        max_points_per_voxel: int = 10,
    ):
        super().__init__()
        self.max_points = max_points_per_voxel

        if vfe_channels is None:
            # VoxelNet default: two VFE layers
            # VFE1: 7 → 32  (output 64 after concat)
            # VFE2: 64 → 64 (output 128 after concat)
            vfe_channels = [(in_channels, 32), (64, 64)]

        self.vfe_layers = nn.ModuleList()
        for c_in, c_out in vfe_channels:
            self.vfe_layers.append(VFELayer(c_in, c_out))

        # final linear to project to desired out_channels
        last_vfe_out = vfe_channels[-1][1] * 2  # after concat
        self.final_linear = nn.Linear(last_vfe_out, out_channels, bias=False)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(
        self, points_per_voxel: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            points_per_voxel: (N_v, T, 7) — augmented point features
            mask:             (N_v, T, 1) — point validity mask

        Returns:
            voxel_features: (N_v, out_channels)
        """
        x = points_per_voxel
        for vfe in self.vfe_layers:
            x = vfe(x, mask)

        # final projection
        nv, T, _ = x.shape
        x = self.final_linear(x)                     # (N_v, T, out_channels)
        x = x.reshape(-1, x.shape[-1])
        x = self.final_bn(x)
        x = self.final_relu(x)
        x = x.reshape(nv, T, -1)
        x = x * mask

        # final max pool across points
        pool_input = x.clone()
        pool_input[mask.squeeze(-1) == 0] = float("-inf")
        voxel_features = pool_input.max(dim=1).values  # (N_v, out_channels)
        return voxel_features


# ======================================================================
# V2G Lifter with full VFE
# ======================================================================
class VoxelToGaussianLifter(nn.Module):
    """Voxel-to-Gaussian initialization with VoxelNet-style encoding.

    Args:
        num_gaussians: N_g. Default 25600.
        embed_dims:    Gaussian query feature dim (m). Default 128.
        voxel_size:    [vx, vy, vz] in meters.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        num_classes:   |C|. Default 17.
        num_sweeps:    N_f. Default 10.
        max_points_per_voxel: T — max points retained per voxel. Default 10.
        vfe_channels:  VFE layer sizes. None → VoxelNet default.
    """

    def __init__(
        self,
        num_gaussians: int = 25600,
        embed_dims: int = 128,
        voxel_size: Optional[List[float]] = None,
        point_cloud_range: Optional[List[float]] = None,
        num_classes: int = 17,
        num_sweeps: int = 10,
        max_points_per_voxel: int = 10,
        vfe_channels: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.num_sweeps = num_sweeps
        self.max_points_per_voxel = max_points_per_voxel

        if voxel_size is None:
            voxel_size = [0.075, 0.075, 0.2]
        if point_cloud_range is None:
            point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

        self.register_buffer(
            "voxel_size", torch.tensor(voxel_size, dtype=torch.float32)
        )
        self.register_buffer(
            "pc_range", torch.tensor(point_cloud_range, dtype=torch.float32)
        )
        self.grid_size = [
            int((point_cloud_range[3 + i] - point_cloud_range[i]) / voxel_size[i])
            for i in range(3)
        ]

        # --- VoxelNet-style feature encoder ---
        self.vfe = VoxelFeatureEncoder(
            in_channels=7,  # [x, y, z, intensity, dx, dy, dz]
            vfe_channels=vfe_channels,
            out_channels=embed_dims,
            max_points_per_voxel=max_points_per_voxel,
        )

        # Gaussian property dim: mean(3) + scale(3) + rotation(4) + opacity(1) + semantics(|C|)
        prop_dim = 11 + num_classes

        # MLP: voxel features → Gaussian properties (residual on top of geometric init)
        self.vfe_to_props = nn.Sequential(
            nn.Linear(embed_dims, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, prop_dim),
        )
        # Zero-init the last layer so initial output ≈ geometric defaults
        nn.init.zeros_(self.vfe_to_props[-1].weight)
        nn.init.zeros_(self.vfe_to_props[-1].bias)

        # Learned defaults for Gaussians without a matching voxel
        self.default_props = nn.Parameter(torch.zeros(1, prop_dim))
        nn.init.normal_(self.default_props[:, :3], std=0.1)
        self.default_props.data[:, 3:6] = 0.0      # scale (log-space)
        self.default_props.data[:, 6] = 1.0         # quaternion w
        self.default_props.data[:, 7:10] = 0.0      # quaternion xyz
        nn.init.constant_(self.default_props.data[:, 10], -2.0)  # opacity

        self.default_query = nn.Parameter(torch.randn(1, embed_dims) * 0.02)

    # ------------------------------------------------------------------
    # Sweep aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_sweeps(sweeps: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate N_f LiDAR sweeps into one combined point cloud."""
        return torch.cat(sweeps, dim=0)

    # ------------------------------------------------------------------
    # Hard voxelization + per-voxel point gathering
    # ------------------------------------------------------------------
    def voxelize(
        self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Assign points to voxels, gather per-voxel point groups.

        Args:
            points: (N, >=4) — [x, y, z, intensity, ...]

        Returns:
            points_per_voxel: (N_v, T, 7) — augmented features, zero-padded
            mask:             (N_v, T, 1) — 1 for real points, 0 for pad
            voxel_centers:    (N_v, 3)    — centroid of points per voxel
            n_voxels:         int
        """
        device = points.device
        pc = self.pc_range
        vs = self.voxel_size
        T = self.max_points_per_voxel

        # ---- range filter ----
        in_range = (
            (points[:, 0] >= pc[0]) & (points[:, 0] < pc[3])
            & (points[:, 1] >= pc[1]) & (points[:, 1] < pc[4])
            & (points[:, 2] >= pc[2]) & (points[:, 2] < pc[5])
        )
        points = points[in_range]

        if points.shape[0] == 0:
            return (
                torch.zeros(0, T, 7, device=device),
                torch.zeros(0, T, 1, device=device),
                torch.zeros(0, 3, device=device),
                0,
            )

        # ---- flat voxel index ----
        coords = ((points[:, :3] - pc[:3]) / vs).long()
        gy, gz = self.grid_size[1], self.grid_size[2]
        flat_idx = coords[:, 0] * (gy * gz) + coords[:, 1] * gz + coords[:, 2]

        unique_idx, inverse = flat_idx.unique(return_inverse=True)
        nv = unique_idx.shape[0]

        xyz = points[:, :3]
        intensity = (
            points[:, 3:4] if points.shape[1] > 3
            else torch.zeros(points.shape[0], 1, device=device)
        )

        # ---- per-voxel centroid (for augmentation + Gaussian mean init) ----
        sum_xyz = torch.zeros(nv, 3, device=device)
        sum_xyz.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), xyz)
        counts = torch.zeros(nv, 1, device=device)
        counts.scatter_add_(
            0, inverse.unsqueeze(-1),
            torch.ones(points.shape[0], 1, device=device),
        )
        counts = counts.clamp(min=1)
        voxel_centers = sum_xyz / counts  # (nv, 3)

        # ---- gather points into (N_v, T, 7) padded tensor ----
        # augmented feature: [x, y, z, intensity, x-x̄, y-ȳ, z-z̄]
        offsets = xyz - voxel_centers[inverse]  # (N_pts, 3)
        aug_feat = torch.cat([xyz, intensity, offsets], dim=-1)  # (N_pts, 7)

        # Scatter into (nv, T, 7) — keep at most T points per voxel.
        # We use a per-voxel counter to assign intra-voxel slot indices.
        slot_counts = torch.zeros(nv, dtype=torch.long, device=device)
        slot_idx = torch.zeros(points.shape[0], dtype=torch.long, device=device)

        # Sort by voxel index for efficient sequential slot assignment
        sorted_order = inverse.argsort()
        sorted_inverse = inverse[sorted_order]

        # Find boundaries of each voxel group
        boundaries = torch.cat([
            torch.tensor([0], device=device),
            (sorted_inverse[1:] != sorted_inverse[:-1]).nonzero(as_tuple=False).squeeze(-1) + 1,
            torch.tensor([sorted_inverse.shape[0]], device=device),
        ])

        points_per_voxel = torch.zeros(nv, T, 7, device=device)
        mask = torch.zeros(nv, T, 1, device=device)

        for v in range(nv):
            start = boundaries[v].item()
            end = boundaries[v + 1].item()
            n_pts = min(end - start, T)
            # Take first T points (could also random-sample for large voxels)
            idxs = sorted_order[start : start + n_pts]
            points_per_voxel[v, :n_pts] = aug_feat[idxs]
            mask[v, :n_pts] = 1.0

        return points_per_voxel, mask, voxel_centers, nv

    # ------------------------------------------------------------------
    # Build Gaussian props: geometric init + learned residual
    # ------------------------------------------------------------------
    def _init_props(
        self, voxel_features: torch.Tensor, voxel_centers: torch.Tensor,
        mean_intensity: torch.Tensor,
    ) -> torch.Tensor:
        """Combine geometric init (Eq. 4) with a learned residual from VFE.

        Args:
            voxel_features:  (n, embed_dims) — output of VFE encoder
            voxel_centers:   (n, 3) — voxel centroids → Gaussian mean
            mean_intensity:  (n, 1) — mean intensity → opacity seed

        Returns:
            props: (n, 11 + num_classes)
        """
        n = voxel_features.shape[0]
        device = voxel_features.device
        prop_dim = 11 + self.num_classes

        # geometric baseline (Eq. 4)
        base = torch.zeros(n, prop_dim, device=device)
        base[:, :3] = voxel_centers                 # mean
        base[:, 3:6] = 0.0                          # scale (log-space)
        base[:, 6] = 1.0                            # quaternion w
        base[:, 7:10] = 0.0                         # quaternion xyz
        base[:, 10] = mean_intensity.squeeze(-1)    # opacity seed

        # learned residual (zero-init so starts at base)
        residual = self.vfe_to_props(voxel_features)  # (n, prop_dim)

        return base + residual

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, points_batch: List[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points_batch: List[B] of List[N_f] sweep tensors, each (N_i, >=4).
                If sweeps are pre-aggregated, pass as single-element lists.

        Returns:
            all_props:   (B, N_g, 11+|C|) — initial Gaussian properties
            all_queries: (B, N_g, embed_dims) — initial Gaussian query features
        """
        B = len(points_batch)
        N = self.num_gaussians

        all_props = []
        all_queries = []

        for b in range(B):
            # 1) aggregate multi-sweep
            combined = self.aggregate_sweeps(points_batch[b])
            device = combined.device

            # 2) voxelize + gather per-voxel point groups
            ppv, mask, centers, nv = self.voxelize(combined)

            if nv == 0:
                props = self.default_props.expand(N, -1)
                queries = self.default_query.expand(N, -1)

            else:
                # 3) VFE encoding
                voxel_features = self.vfe(ppv, mask)  # (nv, embed_dims)

                # per-voxel mean intensity (for Eq. 4 opacity init)
                # extract from augmented features: column 3 is intensity
                int_sum = (ppv[:, :, 3:4] * mask).sum(dim=1)  # (nv, 1)
                int_cnt = mask.sum(dim=1).clamp(min=1)         # (nv, 1)
                mean_int = int_sum / int_cnt                   # (nv, 1)

                if nv >= N:
                    # more voxels than Gaussians → sample N
                    perm = torch.randperm(nv, device=device)[:N]
                    props = self._init_props(
                        voxel_features[perm], centers[perm], mean_int[perm]
                    )
                    queries = voxel_features[perm]
                else:
                    # fewer voxels → init nv, pad rest with defaults
                    props_v = self._init_props(
                        voxel_features, centers, mean_int
                    )
                    queries_v = voxel_features

                    pad = N - nv
                    props = torch.cat(
                        [props_v, self.default_props.expand(pad, -1)], dim=0
                    )
                    queries = torch.cat(
                        [queries_v, self.default_query.expand(pad, -1)], dim=0
                    )

            all_props.append(props)
            all_queries.append(queries)

        return torch.stack(all_props), torch.stack(all_queries)