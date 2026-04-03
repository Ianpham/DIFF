"""
Occupancy Label Generation Pipeline.

Generates 200×200×16 semantic occupancy labels from:
  - Multi-sweep LiDAR point clouds (accumulated via ego poses)
  - 3D bounding box annotations (dynamic objects)
  - HD map polygons (static surfaces)

This replicates the approach used by OpenScene to create occupancy
ground truth, adapted for arbitrary sensor configurations (any number
of cameras/LiDARs).

Usage:
    python -m data.generate_occ_labels \
        --data-root /data/new_dataset/ \
        --output-dir /data/new_dataset/occupancy/ \
        --split train \
        --num-workers 8

Pipeline per sample:
    1. Accumulate N seconds of LiDAR sweeps in ego reference frame
    2. Voxelize the accumulated point cloud → binary occupied mask
    3. Assign semantics from 3D bounding boxes (dynamic objects)
    4. Assign semantics from HD map (static surfaces: road, sidewalk, etc.)
    5. Ray-trace from sensor origin → mark free/observed voxels
    6. Save as compressed .npz: occ_label (X,Y,Z) + visibility (X,Y,Z)
"""

import os
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import json


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class OccLabelConfig:
    """Configuration for occupancy label generation."""

    # Voxel grid geometry
    point_cloud_range: List[float] = field(
        default_factory=lambda: [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    )
    occ_size: List[int] = field(default_factory=lambda: [200, 200, 16])

    # Multi-sweep accumulation
    accumulation_seconds: float = 20.0
    lidar_freq_hz: float = 10.0
    max_sweeps: int = 200

    # OpenScene 17-class taxonomy:
    #   0: empty       1: barrier       2: bicycle       3: bus
    #   4: car         5: construction  6: motorcycle    7: pedestrian
    #   8: traffic_cone 9: trailer     10: truck
    #  11: driveable   12: other_flat  13: sidewalk
    #  14: terrain     15: manmade     16: vegetation
    num_classes: int = 17
    empty_label: int = 0

    # YOUR DATASET'S box classes → OpenScene occupancy index
    # *** Modify this mapping for your dataset ***
    box_class_mapping: Dict[str, int] = field(default_factory=lambda: {
        "car": 4, "truck": 10, "bus": 3, "trailer": 9,
        "construction_vehicle": 5, "motorcycle": 6, "bicycle": 2,
        "pedestrian": 7, "traffic_cone": 8, "barrier": 1,
    })

    # YOUR DATASET'S map layer names → OpenScene occupancy index
    # *** Modify this mapping for your dataset ***
    map_class_mapping: Dict[str, int] = field(default_factory=lambda: {
        "driveable_area": 11, "road": 11, "lane": 11,
        "crosswalk": 12, "sidewalk": 13, "walkway": 13,
        "terrain": 14, "grass": 14,
        "building": 15, "wall": 15, "fence": 15,
        "vegetation": 16, "tree": 16,
    })

    # Ray tracing
    min_points_per_voxel: int = 1
    max_ray_trace_points: int = 30000

    # Ground-level z-range for map polygon assignment
    ground_z_min: float = -1.0
    ground_z_max: float = 0.5


# ======================================================================
# Coordinate utilities
# ======================================================================

def compute_voxel_size(pc_range, occ_size):
    """Derive voxel size from range and grid dimensions."""
    pc = np.array(pc_range, dtype=np.float32)
    return np.array([
        (pc[3] - pc[0]) / occ_size[0],
        (pc[4] - pc[1]) / occ_size[1],
        (pc[5] - pc[2]) / occ_size[2],
    ], dtype=np.float32)


def build_voxel_centers(pc_range, occ_size, voxel_size):
    """Precompute (X*Y*Z, 3) voxel center coordinates."""
    pc = np.array(pc_range, dtype=np.float32)
    X, Y, Z = occ_size
    xs = np.linspace(pc[0] + voxel_size[0] / 2, pc[3] - voxel_size[0] / 2, X)
    ys = np.linspace(pc[1] + voxel_size[1] / 2, pc[4] - voxel_size[1] / 2, Y)
    zs = np.linspace(pc[2] + voxel_size[2] / 2, pc[5] - voxel_size[2] / 2, Z)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def build_voxel_centers_2d(pc_range, occ_size, voxel_size):
    """Precompute (X*Y, 2) column center xy coordinates."""
    pc = np.array(pc_range, dtype=np.float32)
    X, Y = occ_size[0], occ_size[1]
    xs = np.linspace(pc[0] + voxel_size[0] / 2, pc[3] - voxel_size[0] / 2, X)
    ys = np.linspace(pc[1] + voxel_size[1] / 2, pc[4] - voxel_size[1] / 2, Y)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


# ======================================================================
# Step 1: Multi-sweep LiDAR accumulation
# ======================================================================

def accumulate_sweeps(
    sweep_paths: List[str],
    sweep_poses: List[np.ndarray],
    ego_pose_ref: np.ndarray,
    max_sweeps: int = 200,
    point_dim: int = 5,
) -> np.ndarray:
    """Accumulate multiple LiDAR sweeps into reference ego frame.

    Each sweep is transformed: sweep_ego → world → ref_ego.

    Args:
        sweep_paths:  paths to .bin / .npy / .npz point cloud files
        sweep_poses:  (4,4) world-frame ego pose per sweep
        ego_pose_ref: (4,4) reference frame ego pose
        max_sweeps:   cap on number of sweeps
        point_dim:    columns per point (x,y,z,intensity,ring,...)

    Returns:
        (N_total, point_dim) accumulated points in reference ego frame
    """
    ref_inv = np.linalg.inv(ego_pose_ref)
    all_points = []

    for i, (path, pose) in enumerate(zip(sweep_paths, sweep_poses)):
        if i >= max_sweeps:
            break

        pts = _load_points(path, point_dim)
        if pts is None or pts.shape[0] == 0:
            continue

        # Combined transform: ref_ego ← world ← sweep_ego
        transform = ref_inv @ pose
        xyz = pts[:, :3]
        ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
        xyz_ref = (transform @ np.concatenate([xyz, ones], axis=1).T).T[:, :3]

        if pts.shape[1] > 3:
            transformed = np.concatenate([xyz_ref, pts[:, 3:]], axis=1)
        else:
            transformed = xyz_ref

        all_points.append(transformed)

    if not all_points:
        return np.zeros((0, point_dim), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


def _load_points(path: str, point_dim: int = 5) -> Optional[np.ndarray]:
    """Load a point cloud file in common formats."""
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".bin"):
            return np.fromfile(path, dtype=np.float32).reshape(-1, point_dim)
        elif path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        elif path.endswith(".npz"):
            d = np.load(path)
            key = "points" if "points" in d else list(d.keys())[0]
            return d[key].astype(np.float32)
        elif path.endswith(".pcd"):
            # Basic ASCII PCD reader (for common formats)
            return _load_pcd_ascii(path)
    except Exception:
        return None
    return None


def _load_pcd_ascii(path: str) -> Optional[np.ndarray]:
    """Minimal ASCII PCD loader."""
    with open(path, "r") as f:
        lines = f.readlines()
    header_end = 0
    n_points = 0
    for i, line in enumerate(lines):
        if line.startswith("POINTS"):
            n_points = int(line.strip().split()[-1])
        if line.startswith("DATA"):
            header_end = i + 1
            break
    if n_points == 0:
        return None
    data_lines = lines[header_end:header_end + n_points]
    points = np.array(
        [list(map(float, l.strip().split())) for l in data_lines],
        dtype=np.float32,
    )
    return points


# ======================================================================
# Step 2: LiDAR voxelization → binary occupancy
# ======================================================================

def voxelize_points(
    points: np.ndarray,
    pc_range: np.ndarray,
    voxel_size: np.ndarray,
    occ_size: List[int],
    min_points: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize accumulated points into occupancy grid.

    Returns:
        occupied_mask: (X,Y,Z) bool — voxels with enough LiDAR points
        point_counts:  (X,Y,Z) int  — points per voxel (for diagnostics)
    """
    X, Y, Z = occ_size
    point_counts = np.zeros((X, Y, Z), dtype=np.int32)

    pts = points[:, :3]
    in_range = (
        (pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3]) &
        (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4]) &
        (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5])
    )
    pts = pts[in_range]

    if pts.shape[0] == 0:
        return np.zeros((X, Y, Z), dtype=bool), point_counts

    vi = np.clip(((pts[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32), 0, X - 1)
    vj = np.clip(((pts[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32), 0, Y - 1)
    vk = np.clip(((pts[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32), 0, Z - 1)

    np.add.at(point_counts, (vi, vj, vk), 1)
    return point_counts >= min_points, point_counts


# ======================================================================
# Step 3: Assign semantics from 3D bounding boxes
# ======================================================================

def assign_box_semantics(
    occ_grid: np.ndarray,
    voxel_centers: np.ndarray,
    boxes: List[Dict],
    class_mapping: Dict[str, int],
    ego_pose_ref: np.ndarray,
    occ_size: List[int],
    voxel_size: np.ndarray,
) -> np.ndarray:
    """Assign semantic class to voxels inside 3D bounding boxes.

    For each annotated box, tests which voxel centers fall inside
    the oriented 3D box and assigns the corresponding class.
    Box labels take priority over previous labels (overwrite).
    """
    ref_inv = np.linalg.inv(ego_pose_ref)
    X, Y, Z = occ_size

    for box in boxes:
        class_name = box.get("class_name", box.get("label", "unknown"))
        if class_name not in class_mapping:
            continue
        label = class_mapping[class_name]

        # Transform box center world → ego
        center_world = np.array(box["center"], dtype=np.float64)
        center_ego = (ref_inv @ np.append(center_world, 1.0))[:3]

        length, width, height = box["size"]
        heading = float(box.get("heading", 0.0))
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # Broad-phase: only test voxels near the box
        max_dim = max(length, width, height) / 2 + max(voxel_size) * 2
        near = np.all(np.abs(voxel_centers - center_ego) < max_dim, axis=1)
        candidate_idx = np.where(near)[0]
        if candidate_idx.shape[0] == 0:
            continue

        # Rotate to box-local frame and check half-extents
        delta = voxel_centers[candidate_idx] - center_ego
        local_x = delta[:, 0] * cos_h + delta[:, 1] * sin_h
        local_y = -delta[:, 0] * sin_h + delta[:, 1] * cos_h
        local_z = delta[:, 2]

        inside = (
            (np.abs(local_x) <= length / 2) &
            (np.abs(local_y) <= width / 2) &
            (np.abs(local_z) <= height / 2)
        )

        inside_idx = candidate_idx[inside]
        coords = np.unravel_index(inside_idx, (X, Y, Z))
        occ_grid[coords[0], coords[1], coords[2]] = label

    return occ_grid


# ======================================================================
# Step 4: Assign semantics from HD map polygons
# ======================================================================

def assign_map_semantics(
    occ_grid: np.ndarray,
    voxel_centers_2d: np.ndarray,
    map_polygons: List[Dict],
    class_mapping: Dict[str, int],
    ego_pose_ref: np.ndarray,
    occ_size: List[int],
    pc_range: np.ndarray,
    voxel_size: np.ndarray,
    ground_z_range: Tuple[float, float] = (-1.0, 0.5),
) -> np.ndarray:
    """Assign ground-level semantics from HD map polygons.

    Only fills voxels that are currently labeled empty (0),
    so box labels take priority.
    """
    X, Y, Z = occ_size
    ref_inv = np.linalg.inv(ego_pose_ref)

    z_start = max(0, int((ground_z_range[0] - pc_range[2]) / voxel_size[2]))
    z_end = min(Z, int((ground_z_range[1] - pc_range[2]) / voxel_size[2]) + 1)

    for poly_info in map_polygons:
        class_name = poly_info.get("class_name", poly_info.get("type", "unknown"))
        if class_name not in class_mapping:
            continue
        label = class_mapping[class_name]

        vertices = np.array(poly_info["vertices"], dtype=np.float64)
        if vertices.shape[0] < 3:
            continue

        # Transform polygon to ego frame (xy only)
        if vertices.shape[1] == 2:
            v3d = np.concatenate([
                vertices, np.zeros((vertices.shape[0], 1)),
                np.ones((vertices.shape[0], 1)),
            ], axis=1)
        else:
            v3d = np.concatenate([
                vertices[:, :3],
                np.ones((vertices.shape[0], 1)),
            ], axis=1)

        v_ego = (ref_inv @ v3d.T).T[:, :2]

        inside = _point_in_polygon_batch(voxel_centers_2d, v_ego)
        if not np.any(inside):
            continue

        inside_2d = inside.reshape(X, Y)
        for z in range(z_start, z_end):
            mask = inside_2d & (occ_grid[:, :, z] == 0)
            occ_grid[:, :, z][mask] = label

    return occ_grid


def _point_in_polygon_batch(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray-casting point-in-polygon test.

    Args:
        points:  (N, 2) test points
        polygon: (M, 2) polygon vertices

    Returns:
        (N,) boolean — True if inside
    """
    n_verts = polygon.shape[0]
    inside = np.zeros(points.shape[0], dtype=bool)
    px, py = points[:, 0], points[:, 1]

    j = n_verts - 1
    for i in range(n_verts):
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]

        cond = (yi > py) != (yj > py)
        if np.any(cond):
            x_intersect = xi + (xj - xi) / (yj - yi + 1e-12) * (py - yi)
            cross = cond & (px < x_intersect)
            inside[cross] = ~inside[cross]
        j = i

    return inside


# ======================================================================
# Step 5: Ray tracing for visibility / free space
# ======================================================================

def ray_trace_visibility(
    occ_grid: np.ndarray,
    lidar_points: np.ndarray,
    sensor_origin: np.ndarray,
    pc_range: np.ndarray,
    voxel_size: np.ndarray,
    occ_size: List[int],
    max_rays: int = 30000,
) -> np.ndarray:
    """Vectorized ray tracing to determine which voxels are observed.

    Traces rays from sensor origin through each LiDAR point.
    Voxels along the ray path are marked as "observed" (visible).

    Returns:
        visibility: (X,Y,Z) bool — True = observed (free or occupied)
    """
    X, Y, Z = occ_size
    visibility = np.zeros((X, Y, Z), dtype=bool)

    # All occupied voxels are visible by definition
    visibility[occ_grid > 0] = True

    n = lidar_points.shape[0]
    if n > max_rays:
        idx = np.random.choice(n, max_rays, replace=False)
        lidar_points = lidar_points[idx]

    directions = lidar_points[:, :3] - sensor_origin
    distances = np.linalg.norm(directions, axis=1)
    valid = distances > 0.5
    directions = directions[valid]
    distances = distances[valid]

    if directions.shape[0] == 0:
        return visibility

    dir_norm = directions / distances[:, None]
    step_size = float(voxel_size[0])  # Match voxel resolution
    max_steps = min(int(np.max(distances) / step_size) + 1, 500)

    for s in range(max_steps):
        t = s * step_size
        mask = t < distances
        if not np.any(mask):
            break

        ray_pts = sensor_origin + dir_norm[mask] * t
        vi = ((ray_pts[:, 0] - pc_range[0]) / voxel_size[0]).astype(np.int32)
        vj = ((ray_pts[:, 1] - pc_range[1]) / voxel_size[1]).astype(np.int32)
        vk = ((ray_pts[:, 2] - pc_range[2]) / voxel_size[2]).astype(np.int32)

        ok = (vi >= 0) & (vi < X) & (vj >= 0) & (vj < Y) & (vk >= 0) & (vk < Z)
        vi, vj, vk = vi[ok], vj[ok], vk[ok]
        visibility[vi, vj, vk] = True

    return visibility


# ======================================================================
# Main per-sample generation
# ======================================================================

def generate_single_sample(
    sample_info: Dict,
    cfg: OccLabelConfig,
    data_root: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Generate occupancy label for one sample.

    Args:
        sample_info: Standard dict (see DatasetAdapter docs)
        cfg: OccLabelConfig
        data_root: Root path for resolving relative file paths

    Returns:
        (token, occ_label, visibility_mask)
    """
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    occ_size = cfg.occ_size
    X, Y, Z = occ_size
    voxel_size = compute_voxel_size(pc_range, occ_size)

    token = sample_info["token"]
    ego_pose = np.array(sample_info["ego_pose"], dtype=np.float64)

    # Precompute grid geometry
    voxel_centers = build_voxel_centers(pc_range, occ_size, voxel_size)
    voxel_centers_2d = build_voxel_centers_2d(pc_range, occ_size, voxel_size)

    # ---- 1. Accumulate LiDAR ----
    sweep_paths, sweep_poses = [], []
    for sw in sample_info.get("lidar_sweeps", []):
        p = os.path.join(data_root, sw["path"])
        if os.path.exists(p):
            sweep_paths.append(p)
            sweep_poses.append(np.array(sw["ego_pose"], dtype=np.float64))

    pdim = sample_info.get("point_dim", 5)
    accumulated = (
        accumulate_sweeps(sweep_paths, sweep_poses, ego_pose, cfg.max_sweeps, pdim)
        if sweep_paths else np.zeros((0, pdim), dtype=np.float32)
    )

    # ---- 2. Voxelize → binary occupied ----
    occ_grid = np.zeros((X, Y, Z), dtype=np.int64)

    if accumulated.shape[0] > 0:
        occupied, _ = voxelize_points(
            accumulated, pc_range, voxel_size, occ_size, cfg.min_points_per_voxel,
        )
        # Default occupied label: manmade (15) — will be overwritten
        occ_grid[occupied] = 15

    # ---- 3. Box semantics (overwrites default) ----
    boxes = sample_info.get("boxes", [])
    if boxes:
        occ_grid = assign_box_semantics(
            occ_grid, voxel_centers, boxes,
            cfg.box_class_mapping, ego_pose, occ_size, voxel_size,
        )

    # ---- 4. Map semantics (fills empty ground voxels) ----
    map_polygons = sample_info.get("map_polygons", [])
    if map_polygons:
        occ_grid = assign_map_semantics(
            occ_grid, voxel_centers_2d, map_polygons,
            cfg.map_class_mapping, ego_pose, occ_size,
            pc_range, voxel_size,
            ground_z_range=(cfg.ground_z_min, cfg.ground_z_max),
        )

    # ---- 5. Visibility via ray tracing ----
    sensor_origin = np.array(
        sample_info.get("sensor_origin", [0.0, 0.0, 1.8]), dtype=np.float32,
    )
    # Use current-frame or subsampled accumulated for rays
    current_path = sample_info.get("current_lidar_path")
    if current_path:
        current_pts = _load_points(os.path.join(data_root, current_path), pdim)
    else:
        current_pts = None

    if current_pts is None or current_pts.shape[0] == 0:
        current_pts = accumulated[:min(50000, accumulated.shape[0])]

    visibility = (
        ray_trace_visibility(
            occ_grid, current_pts, sensor_origin,
            pc_range, voxel_size, occ_size, cfg.max_ray_trace_points,
        )
        if current_pts.shape[0] > 0
        else np.zeros((X, Y, Z), dtype=bool)
    )

    return token, occ_grid, visibility


# ======================================================================
# Dataset Adapter
# ======================================================================

class DatasetAdapter:
    """Base adapter — subclass for your specific dataset format.

    Implement load_sample_infos() returning List[Dict] where each dict has:
        - token: str
        - ego_pose: (4,4) array-like
        - lidar_sweeps: List[{path: str, ego_pose: (4,4)}]
        - boxes: List[{center:[x,y,z], size:[l,w,h],
                       heading:float, class_name:str}]
        - map_polygons: List[{vertices:[[x,y],...], class_name:str}]
        - current_lidar_path: str (optional)
        - sensor_origin: [x,y,z] (optional, default [0,0,1.8])
        - point_dim: int (optional, default 5)
    """

    def __init__(self, data_root: str, split: str = "train"):
        self.data_root = data_root
        self.split = split

    def load_sample_infos(self) -> List[Dict]:
        raise NotImplementedError


class NuPlanLikeAdapter(DatasetAdapter):
    """Adapter for nuPlan-style info pickle files.

    Expects: {data_root}/{split}_infos.pkl
    *** Modify _convert_info() field names for your dataset. ***
    """

    def load_sample_infos(self) -> List[Dict]:
        info_path = os.path.join(self.data_root, f"{self.split}_infos.pkl")
        if not os.path.exists(info_path):
            raise FileNotFoundError(
                f"Info file not found: {info_path}\n"
                f"Create it with your dataset's preprocessing script."
            )
        with open(info_path, "rb") as f:
            raw_infos = pickle.load(f)
        return [self._convert_info(r) for r in raw_infos]

    def _convert_info(self, raw: Dict) -> Dict:
        """Convert raw info dict to standard format.

        *** MODIFY THIS for your dataset's field names. ***
        The defaults below handle common nuPlan / nuScenes conventions.
        """
        info = {
            "token": raw.get("token", raw.get("scene_token", "unknown")),
            "ego_pose": raw.get("ego_pose", raw.get("ego2global", np.eye(4))),
            "point_dim": raw.get("point_dim", 5),
        }

        # Sweeps: current + historical
        sweeps = []
        if "lidar_path" in raw:
            sweeps.append({"path": raw["lidar_path"], "ego_pose": info["ego_pose"]})
            info["current_lidar_path"] = raw["lidar_path"]

        for s in raw.get("sweeps", raw.get("lidar_sweeps", [])):
            sweeps.append({
                "path": s.get("data_path", s.get("lidar_path", "")),
                "ego_pose": s.get("ego_pose", s.get("ego2global", np.eye(4))),
            })
        info["lidar_sweeps"] = sweeps

        # Boxes
        boxes = []
        for ann in raw.get("gt_boxes_list", raw.get("annotations", [])):
            if isinstance(ann, dict):
                boxes.append({
                    "center": ann.get("center", ann.get("translation", [0, 0, 0])),
                    "size": ann.get("size", [1, 1, 1]),
                    "heading": ann.get("heading", ann.get("rotation_yaw", 0)),
                    "class_name": ann.get("class_name", ann.get("category", "unknown")),
                })
            elif isinstance(ann, (list, np.ndarray)):
                a = np.array(ann)
                boxes.append({
                    "center": a[:3].tolist(),
                    "size": a[3:6].tolist(),
                    "heading": float(a[6]) if len(a) > 6 else 0.0,
                    "class_name": "unknown",
                })
        info["boxes"] = boxes

        info["map_polygons"] = raw.get("map_polygons", [])
        info["sensor_origin"] = raw.get("lidar_origin", [0.0, 0.0, 1.8])
        return info


# ======================================================================
# Batch processing
# ======================================================================

def _process_one(args):
    """Worker function for multiprocessing."""
    sample_info, cfg, data_root, output_dir = args
    try:
        token, occ_label, visibility = generate_single_sample(
            sample_info, cfg, data_root,
        )
        out_path = os.path.join(output_dir, f"{token}.npz")
        np.savez_compressed(
            out_path,
            occ_label=occ_label.astype(np.uint8),
            visibility=visibility,
        )
        return token, True, ""
    except Exception as e:
        return sample_info.get("token", "?"), False, str(e)


def run_generation(
    adapter: DatasetAdapter,
    cfg: OccLabelConfig,
    output_dir: str,
    num_workers: int = 8,
    skip_existing: bool = True,
):
    """Run label generation for all samples."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading sample infos...")
    infos = adapter.load_sample_infos()
    print(f"Found {len(infos)} samples.")

    if skip_existing:
        to_process = [
            info for info in infos
            if not os.path.exists(os.path.join(output_dir, f"{info['token']}.npz"))
        ]
        skipped = len(infos) - len(to_process)
        if skipped:
            print(f"Skipping {skipped} existing, processing {len(to_process)}.")
    else:
        to_process = infos

    if not to_process:
        print("Nothing to process.")
        return

    args_list = [(info, cfg, adapter.data_root, output_dir) for info in to_process]

    success, failed, errors = 0, 0, []

    if num_workers <= 1:
        for a in tqdm(args_list, desc="Generating labels"):
            tok, ok, err = _process_one(a)
            if ok:
                success += 1
            else:
                failed += 1
                errors.append((tok, err))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for tok, ok, err in tqdm(
                pool.map(_process_one, args_list),
                total=len(args_list), desc="Generating labels",
            ):
                if ok:
                    success += 1
                else:
                    failed += 1
                    errors.append((tok, err))

    print(f"\nDone. Success: {success}, Failed: {failed}")
    if errors:
        for tok, err in errors[:20]:
            print(f"  {tok}: {err}")

    # Save summary
    summary = {
        "total": len(infos), "processed": success + failed,
        "success": success, "failed": failed,
        "config": {
            "point_cloud_range": cfg.point_cloud_range,
            "occ_size": cfg.occ_size,
            "num_classes": cfg.num_classes,
            "accumulation_seconds": cfg.accumulation_seconds,
        },
    }
    with open(os.path.join(output_dir, "generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate occupancy labels from LiDAR + boxes + map"
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--pc-range", type=float, nargs=6, default=None)
    parser.add_argument("--occ-size", type=int, nargs=3, default=None)
    parser.add_argument("--accum-seconds", type=float, default=None)
    args = parser.parse_args()

    cfg = OccLabelConfig()
    if args.pc_range:
        cfg.point_cloud_range = args.pc_range
    if args.occ_size:
        cfg.occ_size = args.occ_size
    if args.accum_seconds:
        cfg.accumulation_seconds = args.accum_seconds

    adapter = NuPlanLikeAdapter(data_root=args.data_root, split=args.split)
    run_generation(adapter, cfg, args.output_dir, args.num_workers, not args.no_skip)


if __name__ == "__main__":
    main()