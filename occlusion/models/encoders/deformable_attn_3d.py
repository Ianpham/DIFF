"""
LiDAR-Guided 3D Deformable Attention for GaussianFormer3D.

This module wraps the DFA3D CUDA kernel from:
    https://github.com/IDEA-Research/3D-deformable-attention

It implements Section 3.3 of GaussianFormer3D (arXiv:2505.10685):
    - Two-stage keypoint sampling (Eq. 5)
    - Multi-head weighted aggregation (Eq. 6)

The DFA3D CUDA kernel expects specific tensor layouts. This module handles
all the reshaping, flattening, and coordinate normalization so you can call
it with the natural GaussianFormer3D interface.

Install DFA3D first:
    cd 3D-deformable-attention/DFA3D && bash setup.sh 0
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
#  Try importing the DFA3D CUDA kernels; fall back to pure-PyTorch if absent
# ---------------------------------------------------------------------------
try:
    
    from dfa3D.ops.multi_scale_3D_deform_attn import (
        MultiScale3DDeformableAttnFunction,
        MultiScaleDepthScoreSampleFunction,
        WeightedMultiScaleDeformableAttnFunction,
    )
    HAS_DFA3D_CUDA = True
except ImportError:
    HAS_DFA3D_CUDA = False
    warnings.warn(
        "DFA3D CUDA kernel not found. Using pure-PyTorch fallback. "
        "Install from https://github.com/IDEA-Research/3D-deformable-attention"
    )


# ===========================================================================
#  Pure-PyTorch fallback for depth-weighted bilinear sampling
# ===========================================================================

def _depth_weighted_bilinear_sample_pytorch(
    img_feat: torch.Tensor,    # (BV, C, H, W)
    depth_dist: torch.Tensor,  # (BV, D, H, W)
    sample_uvd: torch.Tensor,  # (BV, P, 3)  u,v in [-1,1], d in [0,1]
) -> torch.Tensor:
    """Pure-PyTorch equivalent of the DFA3D CUDA kernel.

    Samples from the implicit volume F^{3D}(u,v,d) = F^d(u,v,d) · F^c(u,v)
    without materializing (H, W, D, C).

    Returns: (BV, C, P)
    """
    D = depth_dist.shape[1]
    uv = sample_uvd[..., :2]          # (BV, P, 2)
    d_norm = sample_uvd[..., 2]       # (BV, P) in [0, 1]

    grid = uv.unsqueeze(2)            # (BV, P, 1, 2)

    # 1) Bilinear-sample image features at (u, v)
    sampled_img = F.grid_sample(
        img_feat, grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).squeeze(-1)                     # (BV, C, P)

    # 2) Bilinear-sample depth distribution at (u, v) for all D bins
    sampled_depth = F.grid_sample(
        depth_dist, grid, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    ).squeeze(-1)                     # (BV, D, P)

    # 3) Linear interpolation along depth axis
    d_cont = d_norm * (D - 1)         # (BV, P) in [0, D-1]
    d_lo = d_cont.long().clamp(0, D - 2)
    d_hi = d_lo + 1
    w_hi = (d_cont - d_lo.float()).unsqueeze(1)  # (BV, 1, P)
    w_lo = 1.0 - w_hi

    score_lo = torch.gather(sampled_depth, 1, d_lo.unsqueeze(1))
    score_hi = torch.gather(sampled_depth, 1, d_hi.unsqueeze(1))
    depth_score = w_lo * score_lo + w_hi * score_hi  # (BV, 1, P)

    # 4) Outer-product sample
    return sampled_img * depth_score  # (BV, C, P)


# ===========================================================================
#  Helper: prepare flattened multi-scale tensors for DFA3D CUDA kernel
# ===========================================================================

def _prepare_dfa3d_inputs(
    multi_scale_cam_features: List[torch.Tensor],  # L x (B*Nc, C, H_l, W_l)
    multi_scale_depth_maps: List[torch.Tensor],     # L x (B*Nc, D, H_l, W_l)
    num_heads: int,
) -> dict:
    """Flatten multi-scale features into the layout DFA3D CUDA expects.

    DFA3D kernel expects:
        value:          (bs, spatial_size, num_heads, head_dim)
                        where spatial_size = sum(H_l * W_l) across levels
        value_dpt_dist: (bs, spatial_size, num_heads, D)
        spatial_shapes:    (num_levels, 2)  -- (H_l, W_l) per level
        spatial_shapes_3D: (num_levels, 3)  -- (H_l, W_l, D) per level
        level_start_index: (num_levels,)    -- cumsum offsets

    Note: DFA3D replicates depth across heads in the value_dpt_dist tensor.
    """
    device = multi_scale_cam_features[0].device
    BNc = multi_scale_cam_features[0].shape[0]
    C = multi_scale_cam_features[0].shape[1]
    D = multi_scale_depth_maps[0].shape[1]
    head_dim = C // num_heads
    L = len(multi_scale_cam_features)

    spatial_shapes_2d = []
    spatial_shapes_3d = []
    value_list = []
    dpt_list = []

    for l in range(L):
        feat_l = multi_scale_cam_features[l]   # (BNc, C, H_l, W_l)
        dmap_l = multi_scale_depth_maps[l]     # (BNc, D, H_l, W_l)
        H_l, W_l = feat_l.shape[2], feat_l.shape[3]

        spatial_shapes_2d.append([H_l, W_l])
        spatial_shapes_3d.append([H_l, W_l, D])

        # Reshape: (BNc, C, H_l, W_l) -> (BNc, H_l*W_l, num_heads, head_dim)
        feat_flat = feat_l.flatten(2).permute(0, 2, 1)  # (BNc, H_l*W_l, C)
        feat_flat = feat_flat.reshape(BNc, H_l * W_l, num_heads, head_dim)
        value_list.append(feat_flat)

        # Reshape: (BNc, D, H_l, W_l) -> (BNc, H_l*W_l, num_heads, D)
        # DFA3D replicates depth dist across all heads
        dmap_flat = dmap_l.flatten(2).permute(0, 2, 1)  # (BNc, H_l*W_l, D)
        dmap_flat = dmap_flat.unsqueeze(2).expand(-1, -1, num_heads, -1)
        # (BNc, H_l*W_l, num_heads, D)
        dpt_list.append(dmap_flat.contiguous())

    # Concatenate across levels
    value = torch.cat(value_list, dim=1)   # (BNc, total_spatial, num_heads, head_dim)
    value_dpt = torch.cat(dpt_list, dim=1) # (BNc, total_spatial, num_heads, D)

    spatial_shapes = torch.tensor(
        spatial_shapes_2d, dtype=torch.long, device=device
    )  # (L, 2)
    spatial_shapes_3D = torch.tensor(
        spatial_shapes_3d, dtype=torch.long, device=device
    )  # (L, 3)

    # level_start_index: cumulative sum of H_l * W_l
    areas = spatial_shapes[:, 0] * spatial_shapes[:, 1]
    level_start_index = torch.zeros(L, dtype=torch.long, device=device)
    level_start_index[1:] = torch.cumsum(areas[:-1], dim=0)

    return {
        "value": value.contiguous(),
        "value_dpt_dist": value_dpt.contiguous(),
        "spatial_shapes": spatial_shapes,
        "spatial_shapes_3D": spatial_shapes_3D,
        "level_start_index": level_start_index,
    }


# ===========================================================================
#  3D Deformable Attention Module
# ===========================================================================

class DeformableAttention3D(nn.Module):
    """LiDAR-Guided 3D Deformable Attention (GaussianFormer3D Sec 3.3).

    Uses the DFA3D CUDA kernel when available, otherwise falls back to
    a pure-PyTorch implementation.

    Notation:
        NR1 = num_3d_ref_points      (3D offsets around Gaussian mean)
        NR2 = num_sampling_points    (u,v,d offsets per projected ref point)
        M   = num_heads
        L   = num_levels (FPN)
        Nc  = number of camera views
        D   = num_depth_bins

    DFA3D kernel tensor shapes (for reference):
        value:             (bs, sum(H_l*W_l), M, head_dim)
        value_dpt_dist:    (bs, sum(H_l*W_l), M, D)
        sampling_locs_3D:  (bs, num_query, M, L, num_points, 3)  in [0, 1]
        sampling_locs_2D:  (bs, num_query, M, L, num_points, 2)  in [0, 1]
        attention_weights: (bs, num_query, M, L, num_points)      softmax-ed
    """

    def __init__(
        self,
        embed_dims: int = 128,
        num_heads: int = 8,
        num_levels: int = 4,
        num_3d_ref_points: int = 4,      # NR1
        num_sampling_points: int = 8,    # NR2
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
        dropout: float = 0.1,
        im2col_step: int = 64,
        use_cuda_kernel: Optional[bool] = None,  # None = auto-detect
    ):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.num_levels = num_levels
        self.NR1 = num_3d_ref_points
        self.NR2 = num_sampling_points
        self.num_depth_bins = num_depth_bins
        self.depth_min, self.depth_max = depth_range
        self.im2col_step = im2col_step

        # Total sampling points per query per head = NR1 * NR2
        # (flattened for the CUDA kernel which sees them as num_points)
        self.total_points = num_3d_ref_points * num_sampling_points

        # Decide backend
        if use_cuda_kernel is None:
            self.use_cuda = HAS_DFA3D_CUDA
        else:
            self.use_cuda = use_cuda_kernel
            if use_cuda_kernel and not HAS_DFA3D_CUDA:
                raise RuntimeError("DFA3D CUDA kernel requested but not installed")

        # ---- Learnable parameters ----

        # Stage 1: 3D offsets Δm in world coordinates  (B, N, NR1*3)
        self.offset_3d = nn.Linear(embed_dims, num_3d_ref_points * 3)

        # Stage 2: (Δu, Δv, Δd) offsets per head, level, NR1, NR2
        # Total: M * L * NR1 * NR2 * 3
        self.offset_uvd = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_3d_ref_points * num_sampling_points * 3,
        )

        # Attention weights: softmax over ALL sampling points per head
        # i.e. over L * NR1 * NR2 jointly (matching Deformable-DETR convention)
        self.attn_weights = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_3d_ref_points * num_sampling_points,
        )

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.offset_3d.weight)
        nn.init.zeros_(self.offset_3d.bias)
        nn.init.zeros_(self.offset_uvd.weight)
        nn.init.zeros_(self.offset_uvd.bias)
        nn.init.xavier_uniform_(self.attn_weights.weight)
        nn.init.zeros_(self.attn_weights.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # -------------------------------------------------------------------
    #  Projection: world 3D → normalised (u, v, d) per camera
    # -------------------------------------------------------------------

    def _project_to_uvd(
        self,
        ref_3d: torch.Tensor,       # (B, N, NR1, 3)
        lidar2img: torch.Tensor,     # (B, Nc, 4, 4)
        H_orig: int,
        W_orig: int,
    ):
        """Project 3D reference points to normalised (u, v, d) per camera.

        Returns:
            ref_uvd:  (B, Nc, N, NR1, 3)   with u,v in [0,1], d in [0,1]
            valid:    (B, Nc, N, NR1)       bool mask
        """
        B, N, NR1, _ = ref_3d.shape
        Nc = lidar2img.shape[1]

        # Homogeneous coords
        ones = ref_3d.new_ones(B, N, NR1, 1)
        ref_homo = torch.cat([ref_3d, ones], dim=-1)  # (B, N, NR1, 4)
        ref_homo = ref_homo.reshape(B, N * NR1, 4)

        # Project: (B, Nc, 4, 4) x (B, N*NR1, 4) -> (B, Nc, N*NR1, 4)
        proj = torch.einsum("bcij,bnj->bcni", lidar2img, ref_homo)

        depth = proj[..., 2:3].clamp(min=1e-5)                  # (B, Nc, N*NR1, 1)
        u_px = proj[..., 0:1] / depth                           # pixel coords
        v_px = proj[..., 1:2] / depth

        # Normalise to [0, 1] for DFA3D kernel (NOT [-1,1] like grid_sample)
        u_norm = u_px / W_orig                                   # (B, Nc, N*NR1, 1)
        v_norm = v_px / H_orig
        d_norm = (depth - self.depth_min) / (self.depth_max - self.depth_min)
        d_norm = d_norm.clamp(0, 1)

        uvd = torch.cat([u_norm, v_norm, d_norm], dim=-1)       # (B, Nc, N*NR1, 3)

        # Validity mask
        valid = (
            (u_norm[..., 0] >= 0) & (u_norm[..., 0] <= 1)
            & (v_norm[..., 0] >= 0) & (v_norm[..., 0] <= 1)
            & (depth[..., 0] > self.depth_min)
        )

        uvd = uvd.reshape(B, Nc, N, NR1, 3)
        valid = valid.reshape(B, Nc, N, NR1)
        return uvd, valid

    # -------------------------------------------------------------------
    #  Forward: CUDA kernel path
    # -------------------------------------------------------------------

    def _forward_cuda(
        self,
        query: torch.Tensor,                          # (B, N, C)
        gaussian_means: torch.Tensor,                  # (B, N, 3)
        multi_scale_cam_features: List[torch.Tensor],  # L x (B, Nc, C, H_l, W_l)
        multi_scale_depth_maps: List[torch.Tensor],    # L x (B, Nc, D, H_l, W_l)
        lidar2img: torch.Tensor,                       # (B, Nc, 4, 4)
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, C = query.shape
        Nc = lidar2img.shape[1]
        M = self.num_heads
        L = self.num_levels
        NR1, NR2 = self.NR1, self.NR2
        H_orig, W_orig = img_shape

        # ===== Stage 1: 3D offsets + projection =====
        offsets_3d = self.offset_3d(query).reshape(B, N, NR1, 3)
        ref_3d = gaussian_means.unsqueeze(2) + offsets_3d           # (B, N, NR1, 3)
        ref_uvd, valid = self._project_to_uvd(ref_3d, lidar2img, H_orig, W_orig)
        # ref_uvd: (B, Nc, N, NR1, 3) in [0, 1]
        # valid:   (B, Nc, N, NR1)

        # ===== Stage 2: u,v,d offsets =====
        delta_uvd = self.offset_uvd(query).reshape(B, N, M, L, NR1, NR2, 3)

        # ===== Attention weights: softmax over L * NR1 * NR2 jointly =====
        attn_w = self.attn_weights(query).reshape(B, N, M, L * NR1 * NR2)
        attn_w = attn_w.softmax(dim=-1)
        attn_w = attn_w.reshape(B, N, M, L, NR1, NR2)

        # ===== Accumulate across cameras =====
        # Process each camera view, call DFA3D kernel, then average
        output = query.new_zeros(B, N, C)

        for c in range(Nc):
            # Get validity mask for this camera: (B, N, NR1)
            v_mask_c = valid[:, c]  # (B, N, NR1)

            # Base reference: (B, N, NR1, 3) in [0, 1]
            base_c = ref_uvd[:, c]

            # Compute sampling locations:
            #   base_c:    (B, N, 1, 1, NR1, 1, 3)
            #   delta_uvd: (B, N, M, L, NR1, NR2, 3)
            # -> locs:     (B, N, M, L, NR1, NR2, 3)
            locs = base_c[:, :, None, None, :, None, :] + delta_uvd
            locs = locs.clamp(0, 1)

            # Flatten NR1 * NR2 -> total_points for DFA3D kernel
            # DFA3D expects: (bs, num_query, M, L, num_points, 3)
            locs_flat = locs.reshape(B, N, M, L, NR1 * NR2, 3)

            # sampling_locations_3D for depth score: (B, N, M, L, num_points, 3)
            sampling_locs_3D = locs_flat.contiguous()

            # sampling_locations_2D: just u,v: (B, N, M, L, num_points, 2)
            sampling_locs_2D = locs_flat[..., :2].contiguous()

            # Attention weights for this cam: (B, N, M, L, NR1*NR2)
            # Apply validity mask: zero out weights for invalid projections
            # v_mask_c: (B, N, NR1) -> expand to (B, N, 1, 1, NR1, 1)
            mask_expanded = v_mask_c[:, :, None, None, :, None].float()
            mask_expanded = mask_expanded.expand_as(attn_w)
            aw_masked = attn_w * mask_expanded
            aw_flat = aw_masked.reshape(B, N, M, L, NR1 * NR2).contiguous()

            # Prepare flattened multi-scale features for this camera
            cam_feats = [f[:, c] for f in multi_scale_cam_features]  # L x (B, C, H_l, W_l)
            cam_dmaps = [d[:, c] for d in multi_scale_depth_maps]    # L x (B, D, H_l, W_l)

            prepared = _prepare_dfa3d_inputs(cam_feats, cam_dmaps, M)

            # ----- Call DFA3D CUDA kernel (one-stage fused) -----
        # cam_result = MultiScale3DDeformableAttnFunction.apply(
        #         prepared["value"],               # (B, spatial_size, M, head_dim)
        #         prepared["value_dpt_dist"],       # (B, spatial_size, M, D)
        #         prepared["spatial_shapes_3D"],    # (L, 3)
        #         prepared["level_start_index"],    # (L,)
        #         sampling_locs_3D,                 # (B, N, M, L, num_points, 3)
        #         aw_flat,                          # (B, N, M, L, num_points)
        #         self.im2col_step,
        # )
        # print(f"Return type: {type(cam_result)}", flush=True)
        # print(f"Tuple length: {len(cam_result)}", flush=True)
        # for i, item in enumerate(cam_result):
        #     if isinstance(item, torch.Tensor):
        #         print(f"  [{i}] Tensor shape: {item.shape}, dtype: {item.dtype}", flush=True)
        #     else:
        #         print(f"  [{i}] {type(item)}: {item}", flush=True)
        cam_result = MultiScale3DDeformableAttnFunction.apply(
            prepared["value"],
            prepared["value_dpt_dist"],
            prepared["spatial_shapes_3D"],
            prepared["level_start_index"],
            sampling_locs_3D,
            aw_flat,
            self.im2col_step,
        )
        # DFA3D returns a tuple; first element is the attended features
        if isinstance(cam_result, tuple):
            cam_output = cam_result[0]
        else:
            cam_output = cam_result
        output = output + cam_output

        # Average over cameras
        output = output / Nc
        return self.output_proj(self.dropout(output))

    # -------------------------------------------------------------------
    #  Forward: pure-PyTorch fallback path
    # -------------------------------------------------------------------

    def _forward_pytorch(
        self,
        query: torch.Tensor,
        gaussian_means: torch.Tensor,
        multi_scale_cam_features: List[torch.Tensor],
        multi_scale_depth_maps: List[torch.Tensor],
        lidar2img: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        B, N, C = query.shape
        Nc = lidar2img.shape[1]
        M, hd = self.num_heads, self.head_dim
        NR1, NR2, L = self.NR1, self.NR2, self.num_levels
        D = self.num_depth_bins
        H_orig, W_orig = img_shape

        # Stage 1
        offsets_3d = self.offset_3d(query).reshape(B, N, NR1, 3)
        ref_3d = gaussian_means.unsqueeze(2) + offsets_3d

        # For PyTorch path, we need [-1, 1] for grid_sample
        # So recompute projection with [-1, 1] normalisation
        ref_uvd, valid = self._project_to_uvd_grid_sample(
            ref_3d, lidar2img, H_orig, W_orig
        )

        # Stage 2: offsets and weights
        delta_uvd = self.offset_uvd(query).reshape(B, N, M, L, NR1, NR2, 3)

        # Softmax over ALL sampling points jointly per head
        aw = self.attn_weights(query).reshape(B, N, M, L * NR1 * NR2).softmax(dim=-1)
        aw = aw.reshape(B, N, M, L, NR1, NR2)

        # Accumulate
        output = query.new_zeros(B, N, M, hd)

        for l in range(L):
            feat_l = multi_scale_cam_features[l]  # (B, Nc, C, H_l, W_l)
            dmap_l = multi_scale_depth_maps[l]    # (B, Nc, D, H_l, W_l)
            BNc = B * Nc
            H_l, W_l = feat_l.shape[-2], feat_l.shape[-1]

            # Reshape for per-head sampling
            feat_flat = feat_l.reshape(BNc, C, H_l, W_l)
            feat_heads = feat_flat.reshape(BNc, M, hd, H_l, W_l)
            dmap_flat = dmap_l.reshape(BNc, D, H_l, W_l)

            for r1 in range(NR1):
                base = ref_uvd[:, :, :, r1, :]       # (B, Nc, N, 3)
                v_mask = valid[:, :, :, r1].float()   # (B, Nc, N)
                delta = delta_uvd[:, :, :, l, r1]     # (B, N, M, NR2, 3)
                w = aw[:, :, :, l, r1]                # (B, N, M, NR2)

                # Sampling locs: (B, Nc, N, M, NR2, 3)
                locs = base.unsqueeze(3).unsqueeze(4) + delta.unsqueeze(1)
                locs[..., :2] = locs[..., :2].clamp(-1, 1)
                locs[..., 2] = locs[..., 2].clamp(0, 1)

                for h in range(M):
                    feat_h = feat_heads[:, h]   # (BNc, hd, H_l, W_l)
                    sl = locs[:, :, :, h]       # (B, Nc, N, NR2, 3)
                    sl = sl.reshape(BNc, N * NR2, 3)

                    sampled = _depth_weighted_bilinear_sample_pytorch(
                        feat_h, dmap_flat, sl
                    )
                    sampled = sampled.reshape(B, Nc, hd, N, NR2)

                    masked = sampled * v_mask.unsqueeze(2).unsqueeze(-1)
                    cam_avg = masked.mean(dim=1)   # (B, hd, N, NR2)

                    wh = w[:, :, h]                # (B, N, NR2)
                    agg = (cam_avg * wh.unsqueeze(1)).sum(dim=-1)  # (B, hd, N)
                    output[:, :, h, :] += agg.permute(0, 2, 1)

        output = output.reshape(B, N, C)
        return self.output_proj(self.dropout(output))

    def _project_to_uvd_grid_sample(self, ref_3d, lidar2img, H_orig, W_orig):
        """Project to (u, v, d) with u,v in [-1, 1] for grid_sample."""
        B, N, NR1, _ = ref_3d.shape
        ones = ref_3d.new_ones(B, N, NR1, 1)
        ref_homo = torch.cat([ref_3d, ones], dim=-1).reshape(B, N * NR1, 4)

        proj = torch.einsum("bcij,bnj->bcni", lidar2img, ref_homo)
        depth = proj[..., 2:3].clamp(min=1e-5)
        u_n = 2.0 * (proj[..., 0:1] / depth) / W_orig - 1.0
        v_n = 2.0 * (proj[..., 1:2] / depth) / H_orig - 1.0
        d_n = ((depth - self.depth_min) / (self.depth_max - self.depth_min)).clamp(0, 1)

        uvd = torch.cat([u_n, v_n, d_n], dim=-1)
        valid = (u_n[..., 0].abs() <= 1) & (v_n[..., 0].abs() <= 1) & (depth[..., 0] > 0)

        Nc = lidar2img.shape[1]
        return uvd.reshape(B, Nc, N, NR1, 3), valid.reshape(B, Nc, N, NR1)

    # -------------------------------------------------------------------
    #  Main forward dispatch
    # -------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,                          # (B, N, C)
        gaussian_means: torch.Tensor,                  # (B, N, 3)
        multi_scale_cam_features: List[torch.Tensor],  # L x (B, Nc, C, H_l, W_l)
        multi_scale_depth_maps: List[torch.Tensor],    # L x (B, Nc, D, H_l, W_l)
        lidar2img: torch.Tensor,                       # (B, Nc, 4, 4)
        img_shape: Tuple[int, int],                    # (H_orig, W_orig)
    ) -> torch.Tensor:
        """Returns ΔQ: (B, N, C)."""
        if self.use_cuda:
            return self._forward_cuda(
                query, gaussian_means,
                multi_scale_cam_features, multi_scale_depth_maps,
                lidar2img, img_shape,
            )
        else:
            return self._forward_pytorch(
                query, gaussian_means,
                multi_scale_cam_features, multi_scale_depth_maps,
                lidar2img, img_shape,
            )


# ===========================================================================
#  Sparse Conv Block (placeholder for spconv)
# ===========================================================================

class SparseConv3DBlock(nn.Module):
    """MLP substitute for 3D sparse convolution.

    Replace with spconv.SparseSequential for production.
    """

    def __init__(self, embed_dims: int = 128):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Linear(embed_dims + 3, embed_dims),
            nn.BatchNorm1d(embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        B, N, D = features.shape
        x = torch.cat([features, positions], dim=-1).reshape(B * N, D + 3)
        x = self.local_mlp(x).reshape(B, N, D)
        return self.norm(features + x)


# ===========================================================================
#  Full Gaussian Refinement Block
# ===========================================================================
# just for workability testing
class GaussianRefinementBlock(nn.Module):
    """One block of iterative Gaussian refinement (Sec 3.3).

    Pipeline: SparseConv → 3D DeformAttn → FFN.
    Stack multiple blocks for iterative update.
    """

    def __init__(
        self,
        embed_dims: int = 128,
        num_heads: int = 8,
        num_levels: int = 4,
        num_3d_ref_points: int = 4,
        num_sampling_points: int = 8,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sparse_conv = SparseConv3DBlock(embed_dims)
        self.deform_attn = DeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_3d_ref_points=num_3d_ref_points,
            num_sampling_points=num_sampling_points,
            num_depth_bins=num_depth_bins,
            depth_range=depth_range,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(True),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,                          # (B, N, C)
        gaussian_means: torch.Tensor,                  # (B, N, 3)
        multi_scale_cam_features: List[torch.Tensor],
        multi_scale_depth_maps: List[torch.Tensor],
        lidar2img: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        # 1) Sparse conv self-encoding
        query = self.sparse_conv(query, gaussian_means)

        # 2) Pre-norm + 3D deformable cross-attention + residual
        query = query + self.deform_attn(
            self.norm1(query), gaussian_means,
            multi_scale_cam_features, multi_scale_depth_maps,
            lidar2img, img_shape,
        )
        query = self.norm2(query)

        # 3) Pre-norm + FFN + residual
        query = query + self.ffn(self.norm3(query))
        return query, gaussian_means


# ===========================================================================
#  Quick sanity test
# ===========================================================================

# if __name__ == "__main__":
#     print(f"DFA3D CUDA available: {HAS_DFA3D_CUDA}")

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     B, N, C = 1, 100, 128
#     Nc, L, D = 6, 4, 64
#     M = 8

#     block = GaussianRefinementBlock(
#         embed_dims=C, num_heads=M, num_levels=L,
#         num_depth_bins=D,
#     ).to(device)

#     query = torch.randn(B, N, C, device=device)
#     means = torch.randn(B, N, 3, device=device) * 20
#     lidar2img = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, Nc, -1, -1).contiguous()

#     cam_feats, depth_maps = [], []
#     for l in range(L):
#         h, w = 32 // (2 ** l), 48 // (2 ** l)
#         h, w = max(h, 4), max(w, 4)
#         cam_feats.append(torch.randn(B, Nc, C, h, w, device=device))
#         depth_maps.append(torch.randn(B, Nc, D, h, w, device=device).softmax(dim=2))

#     out = block(query, means, cam_feats, depth_maps, lidar2img, (256, 384))
#     print(f"Output shape: {out.shape}")  # expect (1, 100, 128)
#     print("Sanity test passed!")

if __name__ == "__main__":
    import traceback
    try:
        print(f"DFA3D CUDA available: {HAS_DFA3D_CUDA}", flush=True)
        print(f"CUDA device available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        B, N, C = 1, 100, 128
        Nc, L, D = 6, 4, 64
        M = 8

        block = GaussianRefinementBlock(
            embed_dims=C, num_heads=M, num_levels=L,
            num_depth_bins=D,
        ).to(device)
        print("Block created successfully", flush=True)

        query = torch.randn(B, N, C, device=device)
        means = torch.randn(B, N, 3, device=device) * 20
        lidar2img = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, Nc, -1, -1).contiguous()

        cam_feats, depth_maps = [], []
        for l in range(L):
            h, w = 32 // (2 ** l), 48 // (2 ** l)
            h, w = max(h, 4), max(w, 4)
            cam_feats.append(torch.randn(B, Nc, C, h, w, device=device))
            depth_maps.append(torch.randn(B, Nc, D, h, w, device=device).softmax(dim=2))
        print("Inputs created, running forward pass...", flush=True)

        out = block(query, means, cam_feats, depth_maps, lidar2img, (256, 384))
        print(f"Output shape: {out.shape}", flush=True)
        print("Sanity test passed!", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"\nFailed with: {e}", flush=True)