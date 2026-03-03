"""Integration tests for the core GaussianFormer3D pipeline.

Tests each module individually, then chains them together to verify
the full path: V2G Lifter → Encoder → Splatting works end-to-end.

No backbones, no TransFuser, no diffusion decoder — just the core
Gaussian scene representation modules.

Run:
    cd occlusion
    python -m pytest tests/test_core_pipeline.py -v
    # or directly:
    python tests/test_core_pipeline.py
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.gaussian_utils import (
    quaternion_to_rotation_matrix,
    build_covariance_3d,
    gaussian_properties_from_raw,
    GaussianParameterHead,
    farthest_point_sampling,
)
from utils.depth_map import (
    project_lidar_to_image,
    build_depth_map_single,
    build_depth_maps_batch,
    downsample_depth_map,
    create_multiscale_depth_maps,
)
from utils.spatial_hash import SpatialHashGrid, build_spatial_hash
from utils.pooling import build_gaussian_pooling
from models.encoders.gaussian_lifter import VoxelToGaussianLifter
from models.decoders.gaussian_splatting import GaussianSplattingDecoder



# Config shared across tests


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
OCC_SIZE = [200, 200, 16]
NUM_CLASSES = 17
NUM_GAUSSIANS = 256  # small for testing, production uses 25600
EMBED_DIMS = 128
NUM_CAMERAS = 8
IMG_SHAPE = (448, 800)
DEPTH_BINS = 64
DEPTH_RANGE = (1.0, 60.0)
NEIGHBOR_RADIUS = 4.0


# def make_dummy_points(n=5000):
#     """Random LiDAR points within PC_RANGE."""
#     xyz = torch.zeros(n, 3)
#     xyz[:, 0] = torch.FloatTensor(n).uniform_(PC_RANGE[0], PC_RANGE[3])
#     xyz[:, 1] = torch.FloatTensor(n).uniform_(PC_RANGE[1], PC_RANGE[4])
#     xyz[:, 2] = torch.FloatTensor(n).uniform_(PC_RANGE[2], PC_RANGE[5])
#     intensity = torch.rand(n, 1)
#     timestamp = torch.zeros(n, 1)
#     return torch.cat([xyz, intensity, timestamp], dim=-1).to(DEVICE)


# def make_dummy_lidar2img(B=1, Nc=NUM_CAMERAS):
#     """Simple projection matrices. Not physically accurate but valid."""
#     l2i = torch.eye(4, device=DEVICE).unsqueeze(0).unsqueeze(0)
#     l2i = l2i.expand(B, Nc, -1, -1).contiguous()
#     # scale to put projections in image range
#     l2i[:, :, 0, 0] = 500.0  # fx
#     l2i[:, :, 1, 1] = 500.0  # fy
#     l2i[:, :, 0, 2] = IMG_SHAPE[1] / 2  # cx
#     l2i[:, :, 1, 2] = IMG_SHAPE[0] / 2  # cy
#     return l2i
def make_dummy_lidar2img(batch, num_cams):
    """Create a realistic projection matrix that actually projects
    points into the image."""
    # intrinsic: simple pinhole
    fx, fy = 500.0, 500.0
    cx, cy = IMG_SHAPE[1] / 2, IMG_SHAPE[0] / 2  # W/2, H/2
    
    intrinsic = torch.eye(4)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    
    # extrinsic: identity (camera = lidar frame)
    extrinsic = torch.eye(4)
    
    lidar2img = intrinsic @ extrinsic  # (4, 4)
    return lidar2img.unsqueeze(0).unsqueeze(0).expand(batch, num_cams, 4, 4).to(DEVICE)


def make_dummy_points(n, device="cuda"):
    """Points that will actually project into the image."""
    xyz = torch.randn(n, 3, device=device)
    xyz[:, 2] = xyz[:, 2].abs() + 5.0  # positive z, 5-15m range
    xyz[:, :2] *= 5.0  # spread in x, y
    return xyz

# 1. Gaussian Utils


def test_quaternion_to_rotation():
    """Identity quaternion → identity rotation."""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (1, 3, 3)
    err = (R - torch.eye(3, device=DEVICE)).abs().max()
    assert err < 1e-5, f"Identity quaternion gave non-identity R, err={err}"
    print("  [PASS] quaternion identity → R identity")


def test_quaternion_orthogonal():
    """Random quaternion → orthogonal R with det=+1."""
    q = torch.randn(4, 4, device=DEVICE)
    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (4, 3, 3)
    # R^T R ≈ I
    RtR = R.transpose(-1, -2) @ R
    err = (RtR - torch.eye(3, device=DEVICE)).abs().max()
    assert err < 1e-4, f"R not orthogonal, err={err}"
    # det ≈ +1
    det = torch.det(R)
    det_err = (det - 1.0).abs().max()
    assert det_err < 1e-4, f"det(R) != 1, err={det_err}"
    print("  [PASS] random quaternion → orthogonal R, det=+1")


def test_covariance():
    """Covariance is symmetric positive semi-definite."""
    scale = torch.randn(2, 3, device=DEVICE)
    rotation = torch.randn(2, 4, device=DEVICE)
    cov = build_covariance_3d(scale, rotation)
    assert cov.shape == (2, 3, 3)
    # symmetric
    sym_err = (cov - cov.transpose(-1, -2)).abs().max()
    assert sym_err < 1e-5, f"Covariance not symmetric, err={sym_err}"
    # PSD: eigenvalues >= 0
    eigvals = torch.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-5).all(), f"Negative eigenvalue: {eigvals.min()}"
    print("  [PASS] covariance symmetric PSD")


def test_property_parsing():
    """Raw tensor → parsed Gaussian properties."""
    raw = torch.randn(2, 10, 11 + NUM_CLASSES, device=DEVICE)
    props = gaussian_properties_from_raw(raw, NUM_CLASSES)
    assert props["mean"].shape == (2, 10, 3)
    assert props["scale"].shape == (2, 10, 3)
    assert props["rotation"].shape == (2, 10, 4)
    assert props["opacity"].shape == (2, 10, 1)
    assert props["semantics"].shape == (2, 10, NUM_CLASSES)
    assert props["covariance"].shape == (2, 10, 3, 3)
    # opacity in [0, 1]
    assert props["opacity"].min() >= 0 and props["opacity"].max() <= 1
    print("  [PASS] property parsing shapes + opacity range")


def test_parameter_head():
    """GaussianParameterHead: features → property residuals."""
    head = GaussianParameterHead(EMBED_DIMS, NUM_CLASSES).to(DEVICE)
    features = torch.randn(2, 10, EMBED_DIMS, device=DEVICE)
    out = head(features)
    expected_dim = 3 + 3 + 4 + 1 + NUM_CLASSES  # 28
    assert out.shape == (2, 10, expected_dim), f"Got {out.shape}"
    # check init: should be near-zero
    assert out.abs().max() < 1.0, "Head not zero-initialized"
    print("  [PASS] parameter head shape + zero-init")


def test_fps():
    """Farthest point sampling."""
    points = torch.randn(2, 100, 3, device=DEVICE)
    indices, sampled = farthest_point_sampling(points, 10)
    assert indices.shape == (2, 10)
    assert sampled.shape == (2, 10, 3)
    # indices should be unique per batch
    for b in range(2):
        assert indices[b].unique().shape[0] == 10, "FPS indices not unique"
    print("  [PASS] FPS unique indices, correct shapes")


def run_gaussian_utils_tests():
    print("\n=== Gaussian Utils ===")
    test_quaternion_to_rotation()
    test_quaternion_orthogonal()
    test_covariance()
    test_property_parsing()
    test_parameter_head()
    test_fps()



# 2. Depth Maps


def test_project_lidar_to_image():
    """LiDAR projection produces valid pixel coords."""
    points = make_dummy_points(1000)
    l2i = make_dummy_lidar2img(1, 1)[0, 0]  # (4, 4)

    uv, depths, valid = project_lidar_to_image(points, l2i, IMG_SHAPE)

    assert uv.ndim == 2 and uv.shape[-1] == 2
    assert depths.ndim == 1
    assert uv.shape[0] == depths.shape[0]
    assert uv.shape[0] > 0, "No valid projections at all"
    # check bounds
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < IMG_SHAPE[0]).all()
    assert (uv[:, 1] >= 0).all() and (uv[:, 1] < IMG_SHAPE[1]).all()
    assert (depths > 0).all()
    print(f"  [PASS] projection: {uv.shape[0]}/{points.shape[0]} valid points")


def test_depth_map_single_hard():
    """Single camera hard depth map."""
    points = make_dummy_points(2000)
    l2i = make_dummy_lidar2img(1, 1)[0, 0]

    dm = build_depth_map_single(
        points, l2i, IMG_SHAPE, DEPTH_BINS, DEPTH_RANGE, mode="hard"
    )
    assert dm.shape == (DEPTH_BINS, IMG_SHAPE[0], IMG_SHAPE[1])
    # should be non-negative
    assert (dm >= 0).all()
    # occupied pixels should sum to ~1 (normalized)
    pixel_sums = dm.sum(dim=0)
    nonzero_mask = pixel_sums > 0
    if nonzero_mask.any():
        max_err = (pixel_sums[nonzero_mask] - 1.0).abs().max()
        assert max_err < 1e-5, f"Pixel sum != 1, err={max_err}"
    # should be sparse
    sparsity = (dm == 0).float().mean()
    assert sparsity > 0.9, f"Depth map not sparse enough: {sparsity:.2%} zeros"
    print(f"  [PASS] hard depth map, sparsity={sparsity:.2%}")


def test_depth_map_single_soft():
    """Single camera soft depth map."""
    points = make_dummy_points(2000)
    l2i = make_dummy_lidar2img(1, 1)[0, 0]

    dm = build_depth_map_single(
        points, l2i, IMG_SHAPE, DEPTH_BINS, DEPTH_RANGE, mode="soft", sigma=1.0
    )
    assert dm.shape == (DEPTH_BINS, IMG_SHAPE[0], IMG_SHAPE[1])
    assert (dm >= 0).all()
    # soft should have more non-zero entries than hard
    nonzero_count = (dm > 0).sum()
    assert nonzero_count > 0, "Soft depth map is all zeros"
    print(f"  [PASS] soft depth map, {nonzero_count.item()} nonzero entries")


def test_depth_map_batch():
    """Batched multi-camera depth maps."""
    B, Nc = 2, NUM_CAMERAS
    points_list = [make_dummy_points(3000) for _ in range(B)]
    l2i = make_dummy_lidar2img(B, Nc)

    dm = build_depth_maps_batch(
        points_list, l2i, IMG_SHAPE, DEPTH_BINS, DEPTH_RANGE
    )
    assert dm.shape == (B, Nc, DEPTH_BINS, IMG_SHAPE[0], IMG_SHAPE[1])
    assert (dm >= 0).all()
    # at least some cameras should have projections
    per_cam_sum = dm.sum(dim=(2, 3, 4))  # (B, Nc)
    assert (per_cam_sum > 0).any(), "No camera got any depth"
    print(f"  [PASS] batch depth maps {dm.shape}")


def test_multiscale_depth():
    """Downsample to FPN levels."""
    B, Nc = 1, NUM_CAMERAS
    dm_full = torch.rand(B, Nc, DEPTH_BINS, 112, 200, device=DEVICE)
    dm_full = dm_full / dm_full.sum(dim=2, keepdim=True).clamp(min=1e-8)

    fpn_shapes = [(56, 100), (28, 50), (14, 25), (7, 13)]
    ms = create_multiscale_depth_maps(dm_full, fpn_shapes)

    assert len(ms) == 4
    for i, (h, w) in enumerate(fpn_shapes):
        assert ms[i].shape == (B, Nc, DEPTH_BINS, h, w), \
            f"Level {i}: expected (...,{h},{w}), got {ms[i].shape}"
        # should still be normalized per pixel
        psum = ms[i].sum(dim=2)
        nonzero = psum > 0
        if nonzero.any():
            err = (psum[nonzero] - 1.0).abs().max()
            assert err < 0.05, f"Level {i} normalization err={err}"
    print(f"  [PASS] multiscale depth: {[s for s in fpn_shapes]}")


def run_depth_map_tests():
    print("\n=== Depth Maps ===")
    test_project_lidar_to_image()
    test_depth_map_single_hard()
    test_depth_map_single_soft()
    test_depth_map_batch()
    test_multiscale_depth()



# 3. Spatial Hash


def test_spatial_hash_build():
    """Build spatial hash and verify structure."""
    B, N = 1, 500
    means = torch.zeros(B, N, 3, device=DEVICE)
    means[0, :, 0] = torch.linspace(PC_RANGE[0], PC_RANGE[3], N)
    means[0, :, 1] = torch.linspace(PC_RANGE[1], PC_RANGE[4], N)
    means[0, :, 2] = 0.0

    grid = build_spatial_hash(means, NEIGHBOR_RADIUS, PC_RANGE)

    assert grid.B == B
    assert grid.N == N
    # all points should be assigned
    total_assigned = grid.cell_counts.sum()
    assert total_assigned == N, f"Assigned {total_assigned}/{N}"
    print(f"  [PASS] spatial hash: {grid.num_cells} cells, all {N} points assigned")


def test_spatial_hash_query():
    """Query returns correct neighbors."""
    B, N = 1, 200
    # place points on a grid for predictable neighbors
    means = torch.zeros(B, N, 3, device=DEVICE)
    means[0, :, 0] = torch.linspace(-10, 10, N)
    means[0, :, 1] = 0.0
    means[0, :, 2] = 0.0

    radius = 2.0
    grid = build_spatial_hash(means, radius, PC_RANGE)

    # query at origin — should find points within [-2, 2] on x-axis
    query = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    indices, mask, dist_sq = grid.query_neighbors_vectorized(
        query, radius ** 2, means, max_neighbors=64
    )

    assert indices.shape[0] == B
    assert indices.shape[1] == 1  # one query point
    valid_count = mask[0, 0].sum().item()
    assert valid_count > 0, "No neighbors found"

    # verify all returned neighbors are actually within radius
    valid_idx = indices[0, 0][mask[0, 0]]
    neighbor_pos = means[0, valid_idx]
    dists = (neighbor_pos - query).norm(dim=-1)
    assert (dists <= radius + 1e-3).all(), f"Neighbor outside radius: max={dists.max()}"
    print(f"  [PASS] spatial hash query: {valid_count} neighbors, all within radius")


def test_spatial_hash_empty_region():
    """Query in empty region returns zero neighbors."""
    B, N = 1, 100
    means = torch.zeros(B, N, 3, device=DEVICE)
    means[0, :, 0] = torch.linspace(10, 20, N)  # all in [10, 20]

    grid = build_spatial_hash(means, NEIGHBOR_RADIUS, PC_RANGE)

    # query at x=-30 — far from all points
    query = torch.tensor([[-30.0, 0.0, 0.0]], device=DEVICE)
    _, mask, _ = grid.query_neighbors_vectorized(
        query, NEIGHBOR_RADIUS ** 2, means, max_neighbors=64
    )
    valid_count = mask[0, 0].sum().item()
    assert valid_count == 0, f"Expected 0 neighbors, got {valid_count}"
    print("  [PASS] empty region: 0 neighbors")


def run_spatial_hash_tests():
    print("\n=== Spatial Hash ===")
    test_spatial_hash_build()
    test_spatial_hash_query()
    test_spatial_hash_empty_region()



# 4. Pooling


def test_fps_pooling():
    """FPS pooling: 256 Gaussians → 32 tokens."""
    pool = build_gaussian_pooling("fps", num_tokens=32, in_dims=EMBED_DIMS, out_dims=256).to(DEVICE)
    features = torch.randn(2, NUM_GAUSSIANS, EMBED_DIMS, device=DEVICE)
    means = torch.randn(2, NUM_GAUSSIANS, 3, device=DEVICE)
    out = pool(features, means)
    assert out.shape == (2, 32, 256), f"Got {out.shape}"
    print(f"  [PASS] FPS pooling: {NUM_GAUSSIANS} → 32 tokens")


def test_learned_pooling():
    """Learned pooling via cross-attention."""
    pool = build_gaussian_pooling("learned", num_tokens=32, in_dims=EMBED_DIMS, out_dims=256).to(DEVICE)
    features = torch.randn(2, NUM_GAUSSIANS, EMBED_DIMS, device=DEVICE)
    means = torch.randn(2, NUM_GAUSSIANS, 3, device=DEVICE)
    out = pool(features, means)
    assert out.shape == (2, 32, 256), f"Got {out.shape}"
    # check gradient flows
    loss = out.sum()
    loss.backward()
    assert pool.pool_queries.grad is not None
    print(f"  [PASS] learned pooling: {NUM_GAUSSIANS} → 32 tokens, grad OK")


def run_pooling_tests():
    print("\n=== Pooling ===")
    test_fps_pooling()
    test_learned_pooling()



# 5. V2G Lifter


def test_lifter_basic():
    """Lifter produces correct shapes."""
    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # input: List[B] of List[sweeps] of (N_i, 5)
    points_batch = [
        [make_dummy_points(3000)],  # batch 0: 1 sweep
        [make_dummy_points(2000), make_dummy_points(1500)],  # batch 1: 2 sweeps
    ]

    props, queries = lifter(points_batch)
    assert props.shape == (2, NUM_GAUSSIANS, 11 + NUM_CLASSES)
    assert queries.shape == (2, NUM_GAUSSIANS, EMBED_DIMS)
    print(f"  [PASS] lifter: {props.shape}, {queries.shape}")


def test_lifter_empty():
    """Lifter handles empty point cloud gracefully."""
    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    # empty sweep — all points outside range
    empty_pts = torch.tensor([[999, 999, 999, 0, 0]], dtype=torch.float32, device=DEVICE)
    points_batch = [[empty_pts]]

    props, queries = lifter(points_batch)
    assert props.shape == (1, NUM_GAUSSIANS, 11 + NUM_CLASSES)
    assert queries.shape == (1, NUM_GAUSSIANS, EMBED_DIMS)
    print("  [PASS] lifter with empty point cloud: defaults used")


def test_lifter_gradient():
    """Gradient flows through lifter."""
    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    points_batch = [[make_dummy_points(3000)]]
    props, queries = lifter(points_batch)

    loss = props.sum() + queries.sum()
    loss.backward()

    has_grad = False
    for name, p in lifter.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients in lifter"
    print("  [PASS] lifter gradient flow OK")


def run_lifter_tests():
    print("\n=== V2G Lifter ===")
    test_lifter_basic()
    test_lifter_empty()
    test_lifter_gradient()



# 6. Gaussian Splatting


def test_splatting_shapes():
    """Splatting produces correct output shape."""
    small_occ = [20, 20, 4]  # small grid for testing
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=NEIGHBOR_RADIUS,
        chunk_size=500,
    ).to(DEVICE)

    # make valid Gaussian properties
    B, N = 1, 100
    props = torch.zeros(B, N, 11 + NUM_CLASSES, device=DEVICE)
    props[..., :3] = torch.randn(B, N, 3) * 10  # means
    props[..., 3:6] = 0.5  # scales (pre-softplus)
    props[..., 6] = 1.0  # quaternion w
    props[..., 10] = 0.0  # opacity (pre-sigmoid → 0.5)
    props[..., 11:] = torch.randn(B, N, NUM_CLASSES) * 0.1  # semantics

    occ = splat(props)
    assert occ.shape == (B, small_occ[0], small_occ[1], small_occ[2], NUM_CLASSES)
    assert torch.isfinite(occ).all(), "NaN/Inf in splatting output"
    print(f"  [PASS] splatting shape: {occ.shape}")


def test_splatting_localization():
    """Gaussian at a specific location should contribute most to nearby voxels."""
    small_occ = [10, 10, 4]
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=10.0,
        chunk_size=500,
        use_empty_gaussian=False,
    ).to(DEVICE)

    B, N = 1, 1
    props = torch.zeros(B, N, 11 + NUM_CLASSES, device=DEVICE)
    props[..., 0] = 0.0  # mean x = center
    props[..., 1] = 0.0  # mean y = center
    props[..., 2] = 2.0  # mean z
    props[..., 3:6] = 0.0  # small scale (softplus(0) ≈ 0.69)
    props[..., 6] = 1.0  # identity quaternion
    props[..., 10] = 5.0  # high opacity (sigmoid(5) ≈ 0.99)
    # class 1 has high logit
    props[..., 12] = 10.0

    occ = splat(props)

    # find the voxel closest to (0, 0, 2)
    voxels = splat.voxel_coords.reshape(
        small_occ[0], small_occ[1], small_occ[2], 3
    )
    center_x = small_occ[0] // 2
    center_y = small_occ[1] // 2
    # find z closest to 2.0
    z_coords = voxels[0, 0, :, 2]
    center_z = (z_coords - 2.0).abs().argmin().item()

    center_val = occ[0, center_x, center_y, center_z, 1]  # class 1
    corner_val = occ[0, 0, 0, 0, 1]  # far corner

    assert center_val > corner_val, \
        f"Center ({center_val:.4f}) should be > corner ({corner_val:.4f})"
    print(f"  [PASS] localization: center={center_val:.4f} > corner={corner_val:.4f}")


def test_splatting_gradient():
    """Gradient flows through splatting back to properties."""
    small_occ = [10, 10, 4]
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=10.0,
        chunk_size=500,
    ).to(DEVICE)

    B, N = 1, 50
    props = torch.randn(B, N, 11 + NUM_CLASSES, device=DEVICE, requires_grad=True)

    occ = splat(props)
    loss = occ.sum()
    loss.backward()

    assert props.grad is not None, "No gradient on props"
    assert props.grad.abs().sum() > 0, "Zero gradient"
    print("  [PASS] splatting gradient flow OK")


def test_splatting_empty_gaussian():
    """Empty Gaussian contributes empty-class logits everywhere."""
    small_occ = [5, 5, 2]
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=2.0,
        chunk_size=500,
        use_empty_gaussian=True,
    ).to(DEVICE)

    # zero Gaussians — only empty Gaussian contributes
    B, N = 1, 10
    props = torch.zeros(B, N, 11 + NUM_CLASSES, device=DEVICE)
    props[..., :3] = 999.0  # means far outside scene
    props[..., 6] = 1.0  # valid quaternion
    props[..., 10] = -10.0  # near-zero opacity

    occ = splat(props)

    # empty class (idx 0) should dominate
    empty_logit = occ[..., 0].mean()
    other_logit = occ[..., 1:].mean()
    assert empty_logit > other_logit, \
        f"Empty class ({empty_logit:.4f}) should dominate ({other_logit:.4f})"
    print(f"  [PASS] empty Gaussian: class0={empty_logit:.4f} > others={other_logit:.4f}")


def test_splatting_multiresolution():
    """Same Gaussians, different grid resolution."""
    small_occ = [10, 10, 4]
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=10.0,
        chunk_size=500,
        use_empty_gaussian=False,
    ).to(DEVICE)

    B, N = 1, 50
    props = torch.randn(B, N, 11 + NUM_CLASSES, device=DEVICE)
    props[..., :3] = torch.randn(B, N, 3) * 10
    props[..., 6] = 1.0

    occ_low = splat(props)
    occ_high = splat.forward_multiresolution(props, [20, 20, 8])

    assert occ_low.shape == (1, 10, 10, 4, NUM_CLASSES)
    assert occ_high.shape == (1, 20, 20, 8, NUM_CLASSES)
    print(f"  [PASS] multi-resolution: {occ_low.shape} and {occ_high.shape}")


def run_splatting_tests():
    print("\n=== Gaussian Splatting ===")
    test_splatting_shapes()
    test_splatting_localization()
    test_splatting_gradient()
    test_splatting_empty_gaussian()
    test_splatting_multiresolution()



# 7. End-to-End: Lifter → Splatting (skip encoder for now)


def test_lifter_to_splatting():
    """Chain lifter output directly to splatting."""
    small_occ = [20, 20, 4]

    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=NEIGHBOR_RADIUS,
        chunk_size=500,
    ).to(DEVICE)

    points_batch = [[make_dummy_points(5000)]]
    props, queries = lifter(points_batch)

    occ = splat(props)
    assert occ.shape == (1, small_occ[0], small_occ[1], small_occ[2], NUM_CLASSES)
    assert torch.isfinite(occ).all()

    # gradient through the whole chain
    loss = occ.sum()
    loss.backward()

    has_lifter_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lifter.parameters()
    )
    assert has_lifter_grad, "No gradient reaching lifter"
    print(f"  [PASS] lifter → splatting chain: {occ.shape}, grad flows to lifter")


def test_lifter_to_splatting_with_loss():
    """Full forward + occupancy loss."""
    from models.losses.occ_loss import OccupancyLoss

    small_occ = [20, 20, 4]

    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=NEIGHBOR_RADIUS,
        chunk_size=500,
    ).to(DEVICE)

    loss_fn = OccupancyLoss(NUM_CLASSES).to(DEVICE)

    points_batch = [[make_dummy_points(5000)]]
    props, queries = lifter(points_batch)
    occ_pred = splat(props)

    # dummy target
    target = torch.randint(0, NUM_CLASSES, small_occ, device=DEVICE).unsqueeze(0)

    losses = loss_fn(occ_pred, target)
    total = losses["total"]
    assert torch.isfinite(total), f"Loss not finite: {total}"
    assert total > 0, "Loss should be > 0 with random init"

    total.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in lifter.parameters()
    )
    assert has_grad, "No gradient from loss → lifter"
    print(f"  [PASS] lifter → splatting → CE+Lovász loss: {total.item():.4f}, grad OK")


def test_lifter_pooling_chain():
    """Lifter → pooling → decoder-ready tokens."""
    lifter = VoxelToGaussianLifter(
        num_gaussians=NUM_GAUSSIANS,
        embed_dims=EMBED_DIMS,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    pool = build_gaussian_pooling(
        "fps", num_tokens=32, in_dims=EMBED_DIMS, out_dims=256
    ).to(DEVICE)

    points_batch = [[make_dummy_points(5000)]]
    props, queries = lifter(points_batch)
    means = props[..., :3]
    tokens = pool(queries, means)

    assert tokens.shape == (1, 32, 256)
    print(f"  [PASS] lifter → FPS pool → 32 decoder tokens: {tokens.shape}")


def run_e2e_tests():
    print("\n=== End-to-End Chains ===")
    test_lifter_to_splatting()
    test_lifter_to_splatting_with_loss()
    test_lifter_pooling_chain()



# 8. Depth Maps + Spatial Hash Integration


def test_depth_to_multiscale_pipeline():
    """Full depth pipeline: points → full res → multi-scale."""
    B = 1
    points_list = [make_dummy_points(3000)]
    l2i = make_dummy_lidar2img(B, NUM_CAMERAS)

    # build at reduced resolution for speed
    reduced_shape = (112, 200)
    dm_full = build_depth_maps_batch(
        points_list, l2i, reduced_shape, DEPTH_BINS, DEPTH_RANGE
    )

    fpn_shapes = [(56, 100), (28, 50), (14, 25), (7, 13)]
    ms_depth = create_multiscale_depth_maps(dm_full, fpn_shapes)

    assert len(ms_depth) == 4
    for i, d in enumerate(ms_depth):
        assert d.shape[0] == B
        assert d.shape[1] == NUM_CAMERAS
        assert d.shape[2] == DEPTH_BINS
    print(f"  [PASS] full depth pipeline: points → batch → multiscale")


def test_spatial_hash_with_splatting():
    """Spatial hash accelerated splatting produces same-shape output."""
    small_occ = [10, 10, 4]
    splat = GaussianSplattingDecoder(
        occ_size=small_occ,
        point_cloud_range=PC_RANGE,
        num_classes=NUM_CLASSES,
        neighbor_radius=NEIGHBOR_RADIUS,
        chunk_size=200,
        use_empty_gaussian=False,
    ).to(DEVICE)

    B, N = 1, 100
    props = torch.randn(B, N, 11 + NUM_CLASSES, device=DEVICE)
    props[..., :3] = torch.randn(B, N, 3) * 10
    props[..., 6] = 1.0

    occ = splat(props, point_cloud_range=PC_RANGE)
    assert occ.shape == (1, 10, 10, 4, NUM_CLASSES)
    assert torch.isfinite(occ).all()
    print(f"  [PASS] spatial hash splatting: {occ.shape}, all finite")


def run_integration_tests():
    print("\n=== Integration ===")
    test_depth_to_multiscale_pipeline()
    test_spatial_hash_with_splatting()



# Main


def main():
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_gaussian_utils_tests()
    run_depth_map_tests()
    run_spatial_hash_tests()
    run_pooling_tests()
    run_lifter_tests()
    run_splatting_tests()
    run_e2e_tests()
    run_integration_tests()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()