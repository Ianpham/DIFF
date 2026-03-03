"""LiDAR-to-camera depth map generation for DFA3D.

Projects LiDAR points into camera views and builds discrete depth
distributions per pixel. Supports both hard (one-hot) and soft
(Gaussian-spread) assignment.

Reference:
    - DFA3D (Li et al., ICCV 2023): linear depth bins, depth-weighted attention
    - BEVDepth (Li et al., AAAI 2023): LiDAR-supervised depth estimation

Output format matches DeformableAttention3D expectations:
    multi_scale_depth_maps: List[Tensor]  # L x (B, Nc, D, H_l, W_l)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

def project_lidar_to_image(
        points: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: Tuple[int, int],
)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project Lidar points into a single camera view
    
    Args:
        points: (N, 3+) xyz in LiDAR frame.
        lidar2img: (4,4) projection matrix for 1 camera.
        img_shape: (H, W) imge dimensions.

    Returns:
        uv: (M, 2) pixel coordinates of valid projection (row, col).
        depths: (M,) depth values of vlaid projections.
        valid_mask: (M, ) bool mask indicating which input points are valid.
    """
    N = points.shape[0]
    H, W = img_shape

    # homogeneous coordinates
    ones = points.new_ones(N, 1)
    pts_homo = torch.cat([points[:, :3], ones], dim = -1) # (N, 4)

    # project: (4, 4) x (N, 4).T -> (4, N)
    proj = (lidar2img @ pts_homo.T).T # (N, 4)

    depth = proj[:, 2]

    # avoid division by zero
    valid_depth = depth > 1e-5
    depth_safe = depth.clamp(min = 1e-5)

    u = proj[:, 0] / depth_safe # pixel x (col)
    v = proj[:, 1] / depth_safe # pixel y (row)

    # bounds check
    valid = (
        valid_depth
        & (u >= 0) & (u < W)
        & (v >= 0) & (v < H)
    )

    uv = torch.stack([v[valid], u[valid]], dim = -1) # (M, 2) row, col
    depths_valid = depth[valid]

    return uv, depths_valid, valid

def build_depth_map_single(
        points: torch.Tensor,
        lidar2img: torch.Tensor,
        img_shape: Tuple[int, int],
        num_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
        mode: str = "hard",
        sigma: float = 1.0,
) -> torch.Tensor:
    """
    Build depth distribution map for one camera
    Args:
        points: (N, 3+) LiDAR
        lidar2img: (4, 4) simgle camera projection. #  later is 8 or 12 camera, base on situation pinhole or fisheye.
        img_shape: (H, W)
        num_bins: D - number of discrete depth bins.
        depth_range: (d_min, d_max)
        mode: "hard" for one-hot, "soft" for gaussian spread.
        sigma: spread width in bin units (only used if mode = "soft")

    Returns:
        depth_map: (D, H, W) depth distribution.
    """

    H, W = img_shape
    D = num_bins
    d_min, d_max = depth_range
    device = points.device

    # filter to depth range
    depth_map = torch.zeros(D, H, W, device=device)
    uv, depths, _ = project_lidar_to_image(points, lidar2img, img_shape)

    if uv.shape[0] == 0:
        return depth_map
    
    # pixel indices
    rows = uv[:, 0].long().clamp(0, H - 1)
    cols = uv[:, 1].long().clamp(0, W - 1)

    # depth to bin index (linear spacing)
    bin_float = (depths - d_min) / (d_max - d_min) * (D - 1)
    bin_float = bin_float.clamp(0, D - 1)

    if mode == "hard":
        bin_idx = bin_float.round().long()
        depth_map[bin_idx, rows, cols] = 1.0

    elif mode == "soft":
        # gaussian spread accorss neighboring bins
        bin_centers = torch.arange(D, device = device, dtype= torch.float32)

        # (M, D): distance from each point's depth to each bin center
        dist = (bin_float.unsqueeze(-1) - bin_centers.unsqueeze(0)) ** 2
        weights = torch.exp( -dist/(2*sigma**2)) # (M, D)
        weights = weights / weights.sum(dim = -1, keepdim= True).clamp(min = 1e-8)

        # scatter: for each point, distribute weight accross bins
        for i in range(uv.shape[0]):
            depth_map[:, rows[i], cols[i]] += weights[i]

    else:
        raise ValueError(f"Unknow mode: {mode}. Use 'hard' or 'soft'.")
    

    # handle mutiple points per pixel: normalize so each pixel sums to 1
    pixel_sum = depth_map.sum(dim = 0, keepdim= True).clamp(min = 1e-8)
    depth_map = depth_map / pixel_sum

    return depth_map


def build_depth_maps_batch(
        points_list: List[torch.tensor],
        lidar2img: torch.Tensor,
        img_shape: Tuple[int, int],
        num_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
        mode: str = "hard",
        sigma: float = 1.0,
)-> torch.Tensor:
    """Build depth maps for a batch, all cameras.

    Args:
        points_list: List[B] of (N_i, 3+) LiDAR point clouds.
        lidar2img: (B, Nc, 4, 4) projection matrices.
        img_shape: (H, W).
        num_bins: D.
        depth_range: (d_min, d_max).
        mode: "hard" or "soft".
        sigma: spread width for soft mode.

    Returns:
        depth_maps: (B, Nc, D, H, W) depth distributions at full resolution.
    """
    B = len(points_list)
    Nc = lidar2img.shape[1]
    D = num_bins
    H, W = img_shape
    device = lidar2img.device

    all_maps = torch.zeros(B, Nc, D, H, W, device =device)

    for b in range(B):
        pts = points_list[b]
        for c in range(Nc):
            all_maps[b, c] = build_depth_map_single(
                pts, lidar2img[b, c], img_shape, 
                num_bins, depth_range, mode, sigma
            )

    return all_maps

def downsample_depth_map(
    depth_map: torch.Tensor,
    target_shape: Tuple[int, int],
) -> torch.Tensor:
    """Downsample depth map to a target spatial resolution.

    Uses average pooling to preserve distribution properties.
    After pooling, re-normalizes so each pixel sums to 1.

    Args:
        depth_map: (..., D, H, W) — any leading batch dims.
        target_shape: (H_t, W_t).

    Returns:
        downsampled: (..., D, H_t, W_t).
    """
    leading = depth_map.shape[:-3]
    D, H, W = depth_map.shape[-3:]
    H_t, W_t = target_shape

    if H == H_t and W == W_t:
        return depth_map

    # flatten leading dims for interpolate
    flat = depth_map.reshape(-1, D, H, W)
    down = F.adaptive_avg_pool2d(flat, (H_t, W_t))

    # re-normalize per pixel
    pixel_sum = down.sum(dim=1, keepdim=True).clamp(min=1e-8)
    down = down / pixel_sum

    return down.reshape(*leading, D, H_t, W_t)

def create_multiscale_depth_maps(
        depth_maps_full: torch.Tensor,
        fpn_shapes: List[Tuple[int, int]],
) -> List[torch.Tensor]:
    """Create multi-scale depth maps matching FPN feature map sizes.
    
    Args:
        depth_map_full: (B, Nc, D, H, W) full resolution depth maps.
        fpn_shapes: List[L] of (H_l, W_l) per FPN level.

    Return:
        List[L] of(B, Nc, D, H_l, W_l) tensors.
    """
    ms_depth = []
    for h_l, w_l in fpn_shapes:
        # ms_depth.append(downsample_depth_map(depth_maps_full(depth_maps_full, (h_l, w_l))))
        ms_depth.append(downsample_depth_map(depth_maps_full, (h_l, w_l)))
    return ms_depth

def precompute_and_save(
        points: torch.Tensor,
        lidar2img: torch.Tensor, 
        img_shape: Tuple[int, int],
        save_path: str,
        num_bins: int = 64,
        depth_range: Tuple[float, float] = (1.0, 60.0),
        mode: str = "hard",
        sigma: float = 1.0,
):
    """Precompute depth maps for a single sample and save to disk.
    
    Saves at full resolution. Multi-scale downsampling is done at
    load time since FPN shapes depend on the model config.

    Args:
        points: (N, 3+) single sample LiDAR points.
        lidar2img: (Nc, 4, 4) camera projections.
        img_shape: (H, W).
        save_path: output .pt file path.
        num_bins, depth_range, mode, sigma: same as build_depth_map_single.
    
    """
    Nc = lidar2img.shape[0]
    D = num_bins
    H, W = img_shape

    depth_maps = torch.zeros(Nc, D, H, W)

    for c in range(Nc):
        depth_maps[c] = build_depth_map_single(
            points, lidar2img[c], img_shape,
            num_bins, depth_range, mode, sigma
        )
    # save as half precision to reduce disk usage
    # full res (Nc, D, H, W) at fp16: 8 cams * 64 * 448 * 800 * 2 bytes ≈ 275 MB
    # that's large — consider saving sparse or at reduced resolution
    torch.save(depth_maps.half(), save_path)


###  usage #####
# from utils.depth_map import build_depth_maps_batch, create_multiscale_depth_maps

# depth_maps_full = build_depth_maps_batch(
#     points, lidar2img, img_shape,
#     num_bins=64, depth_range=(1.0, 60.0), mode="hard",
# )
# fpn_shapes = [(f.shape[-2], f.shape[-1]) for f in ms_feats]
# depth_maps = create_multiscale_depth_maps(depth_maps_full, fpn_shapes)