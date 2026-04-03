#!/usr/bin/env python3
"""
OpenScene Dataset Builder & Visualizer

Three modes:
  1. generate  — Merge per-log meta pkls into one openscene_infos_train.pkl
  2. verify    — Load a few samples through the updated dataset, check everything works
  3. visualize — Show camera images, LiDAR BEV, occupancy label, 3D boxes for N samples

Usage:
  # Step 1: Generate the combined pkl
  python build_dataset.py generate

  # Step 2: Verify dataset loading works
  python build_dataset.py verify

  # Step 3: Visualize samples
  python build_dataset.py visualize --num-samples 3
"""

import os
import sys
import pickle
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# PATHS — your machine
# ============================================================
META_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene_metadata_mini/openscene-v1.1/meta_datas/mini"
OCC_ROOT  = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"
SENSOR_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"
NAVSIM_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/"
MAPS_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps/"

# Output pkl
OUTPUT_PKL = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/openscene_infos_train.pkl"

# For the updated dataset loader
DATA_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"
OCC_LABEL_ROOT = OCC_ROOT

CAMERA_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

# OpenScene 17-class taxonomy
OCC_CLASSES = {
    0: "empty", 1: "barrier", 2: "bicycle", 3: "bus", 4: "car",
    5: "construction", 6: "motorcycle", 7: "pedestrian", 8: "traffic_cone",
    9: "trailer", 10: "truck", 11: "driveable", 12: "other_flat",
    13: "sidewalk", 14: "terrain", 15: "manmade", 16: "vegetation",
}

# Voxel grid config
POINT_CLOUD_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
OCC_SIZE = [200, 200, 16]


# ============================================================
# PART 1: Generate combined pkl
# ============================================================

def generate_combined_pkl():
    """Merge all per-log meta pkls into one combined pkl."""
    print("=" * 60)
    print("GENERATING openscene_infos_train.pkl")
    print("=" * 60)

    pkl_files = sorted(glob.glob(os.path.join(META_ROOT, "*.pkl")))
    if not pkl_files:
        print(f"ERROR: No pkl files in {META_ROOT}")
        return

    print(f"Found {len(pkl_files)} log pkl files")

    all_infos = []
    skipped_no_occ = 0

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            log_data = pickle.load(f)

        frames = log_data if isinstance(log_data, list) else []

        for frame in frames:
            # Adapt frame to our dataset loader format
            info = adapt_frame(frame)
            if info is not None:
                all_infos.append(info)
            else:
                skipped_no_occ += 1

    print(f"\nTotal frames: {len(all_infos)}")
    print(f"Skipped (no occ path): {skipped_no_occ}")

    # Stats
    n_with_occ = sum(1 for i in all_infos if i.get("_has_occ_on_disk", False))
    print(f"With occ label on disk: {n_with_occ}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(all_infos, f)

    size_mb = os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
    print(f"\nSaved to: {OUTPUT_PKL}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Samples: {len(all_infos)}")


def adapt_frame(frame: Dict) -> Optional[Dict]:
    """Convert OpenScene meta frame to our dataset loader format."""
    token = frame.get("token", "")
    if not token:
        return None

    # --- ego_status: list of 4 floats [speed, accel, yaw_rate, steering] ---
    ego_dyn = frame.get("ego_dynamic_state", [0, 0, 0, 0])
    if isinstance(ego_dyn, list):
        ego_status = {
            "speed": float(ego_dyn[0]) if len(ego_dyn) > 0 else 0.0,
            "acceleration": float(ego_dyn[1]) if len(ego_dyn) > 1 else 0.0,
            "yaw_rate": float(ego_dyn[2]) if len(ego_dyn) > 2 else 0.0,
            "steering_angle": float(ego_dyn[3]) if len(ego_dyn) > 3 else 0.0,
        }
    elif isinstance(ego_dyn, dict):
        ego_status = {
            "speed": float(ego_dyn.get("speed", 0)),
            "acceleration": float(ego_dyn.get("acceleration", 0)),
            "yaw_rate": float(ego_dyn.get("yaw_rate", 0)),
            "steering_angle": float(ego_dyn.get("steering_angle", 0)),
        }
    else:
        ego_status = {"speed": 0, "acceleration": 0, "yaw_rate": 0, "steering_angle": 0}

    # --- cameras: compute lidar2img from sensor2lidar ---
    cams_raw = frame.get("cams", {})
    cams = {}
    for cam_name in CAMERA_NAMES:
        cam_data = cams_raw.get(cam_name, {})
        data_path = cam_data.get("data_path", f"dummy_{cam_name}.jpg")

        intrinsic = np.array(cam_data.get("cam_intrinsic", np.eye(3)), dtype=np.float32)
        if intrinsic.shape == (3, 3):
            intrinsic = intrinsic.astype(np.float32)

        s2l_rot = np.array(cam_data.get("sensor2lidar_rotation", np.eye(3)), dtype=np.float32).reshape(3, 3)
        s2l_trans = np.array(cam_data.get("sensor2lidar_translation", np.zeros(3)), dtype=np.float32).reshape(3)

        # sensor2lidar: T_lidar_cam, so lidar2cam = inv(sensor2lidar)
        T_lidar_cam = np.eye(4, dtype=np.float32)
        T_lidar_cam[:3, :3] = s2l_rot
        T_lidar_cam[:3, 3] = s2l_trans
        T_cam_lidar = np.linalg.inv(T_lidar_cam)

        K_4x4 = np.eye(4, dtype=np.float32)
        K_4x4[:3, :3] = intrinsic.reshape(3, 3)
        lidar2img = (K_4x4 @ T_cam_lidar).astype(np.float32)

        cams[cam_name] = {
            "data_path": data_path,
            "cam_intrinsic": intrinsic,
            "lidar2img": lidar2img,
        }

    # --- occ label path: resolve to actual file ---
    occ_gt_path = frame.get("occ_gt_final_path", "")
    occ_resolved = resolve_occ_path(occ_gt_path)

    # --- flow label path ---
    flow_gt_path = frame.get("flow_gt_final_path", "")
    flow_resolved = resolve_occ_path(flow_gt_path) if flow_gt_path else None

    info = {
        "token": token,
        "lidar_path": frame.get("lidar_path", ""),
        "ego_status": ego_status,
        "cams": cams,
        # Extra fields for later phases
        "log_name": frame.get("log_name", ""),
        "scene_name": frame.get("scene_name", ""),
        "frame_idx": frame.get("frame_idx", 0),
        "timestamp": frame.get("timestamp", 0),
        "map_location": frame.get("map_location", ""),
        "ego2global": frame.get("ego2global", np.eye(4)),
        "lidar2ego": frame.get("lidar2ego", np.eye(4)),
        # Occ label
        "occ_gt_path": occ_resolved,
        "flow_gt_path": flow_resolved,
        "_has_occ_on_disk": occ_resolved is not None and os.path.exists(occ_resolved),
        # Annotations (3D boxes for phantom training later)
        "anns": frame.get("anns", {}),
        # Can bus (for trajectory)
        "can_bus": frame.get("can_bus", np.zeros(18)),
        "driving_command": frame.get("driving_command", np.zeros(4)),
    }

    return info


def resolve_occ_path(occ_gt_path: str) -> Optional[str]:
    """Resolve meta's occ_gt_final_path to actual file on disk.

    Meta format: 'dataset/openscene-v1.0/occupancy/mini/log-0001-scene-0001/occ_gt/000_occ_final.npy'
    Actual:      OCC_ROOT/log-0001-scene-0001/occ_gt/000_occ_final.npy
    """
    if not occ_gt_path:
        return None

    parts = occ_gt_path.replace("\\", "/").split("/")
    # Find 'mini' and take everything after it
    for i, p in enumerate(parts):
        if p == "mini":
            rel = "/".join(parts[i + 1:])
            full = os.path.join(OCC_ROOT, rel)
            return full

    return None


# ============================================================
# PART 2: Sparse occ → Dense grid conversion
# ============================================================

def sparse_occ_to_dense(sparse_arr: np.ndarray,
                        occ_size: List[int] = None,
                        num_classes: int = 17) -> np.ndarray:
    """Convert OpenScene sparse occ format to dense voxel grid.

    OpenScene stores occ as (N, 2) int32 where:
      col 0: flattened voxel index into (X*Y*Z) grid
      col 1: class label (0-16)

    Returns:
      dense: (X, Y, Z) int64 with class labels, 0 = empty
    """
    if occ_size is None:
        occ_size = OCC_SIZE

    X, Y, Z = occ_size
    dense = np.zeros(X * Y * Z, dtype=np.int64)  # all empty

    if sparse_arr.shape[0] == 0:
        return dense.reshape(X, Y, Z)

    voxel_indices = sparse_arr[:, 0].astype(np.int64)
    class_labels = sparse_arr[:, 1].astype(np.int64)

    # Filter valid indices
    valid = (voxel_indices >= 0) & (voxel_indices < X * Y * Z)
    voxel_indices = voxel_indices[valid]
    class_labels = class_labels[valid]

    # If indices exceed grid size, they might use a different grid
    # Check max index
    max_idx = voxel_indices.max() if len(voxel_indices) > 0 else 0

    if max_idx >= X * Y * Z:
        # Might be a different grid size, try to infer
        # OpenScene v1.0 uses 200x200x16 = 640000
        # But some versions use 512x512x40 or similar
        print(f"  WARNING: max voxel index {max_idx} >= {X*Y*Z}. "
              f"Trying to infer grid size...")

        # Try common grid sizes
        for test_size in [[200, 200, 16], [512, 512, 40], [256, 256, 32]]:
            if max_idx < test_size[0] * test_size[1] * test_size[2]:
                print(f"  Using inferred size: {test_size}")
                X, Y, Z = test_size
                dense = np.zeros(X * Y * Z, dtype=np.int64)
                break

    valid = (voxel_indices >= 0) & (voxel_indices < X * Y * Z)
    dense[voxel_indices[valid]] = class_labels[valid]

    return dense.reshape(X, Y, Z)


# ============================================================
# PART 3: Updated Dataset Loader
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader


class OpenSceneOccDatasetV2(Dataset):
    """Updated OpenScene dataset that works with your actual data layout.

    Key changes from v1:
      - Loads from combined pkl (generated by Part 1)
      - Camera images at: SENSOR_ROOT/{log_name}/{cam_name}/{hash}.jpg
      - LiDAR at: SENSOR_ROOT/{log_name}/MergedPointCloud/{token}.pcd
      - Occ labels: sparse .npy format → converted to dense (200,200,16)
      - ego_dynamic_state: list of 4 floats, not dict
    """

    CAMERA_NAMES = CAMERA_NAMES

    def __init__(self, info_file, sensor_root, occ_root,
                 img_size=(448, 800), point_cloud_range=None,
                 occ_size=None, num_classes=17,
                 trajectory_length=8, history_length=4,
                 load_occ=True, load_planning=True,
                 max_samples=None):

        self.sensor_root = sensor_root
        self.occ_root = occ_root
        self.img_size = img_size
        self.num_classes = num_classes
        self.traj_len = trajectory_length
        self.hist_len = history_length
        self.load_occ = load_occ
        self.load_planning = load_planning
        self.pc_range = np.array(point_cloud_range or POINT_CLOUD_RANGE)
        self.occ_size = occ_size or OCC_SIZE

        # Load infos
        with open(info_file, "rb") as f:
            self.infos = pickle.load(f)

        if max_samples and len(self.infos) > max_samples:
            self.infos = self.infos[:max_samples]

        print(f"Loaded {len(self.infos)} samples from {info_file}")

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        data = {}

        # --- Camera images ---
        imgs, l2i = [], []
        for cn in self.CAMERA_NAMES:
            ci = info["cams"][cn]
            # Path: sensor_root/log_name/cam_name/hash.jpg
            img_path = os.path.join(self.sensor_root, ci["data_path"])

            if os.path.exists(img_path):
                from PIL import Image
                pil_img = Image.open(img_path).resize(
                    (self.img_size[1], self.img_size[0])
                )
                img = np.array(pil_img).astype(np.float32)
            else:
                img = np.random.randint(
                    0, 256, (*self.img_size, 3), dtype=np.uint8
                ).astype(np.float32)

            # ImageNet normalization
            img -= np.array([103.530, 116.280, 123.675])
            imgs.append(img)
            l2i.append(ci["lidar2img"])

        data["images"] = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
        data["lidar2img"] = torch.from_numpy(np.stack(l2i)).float()

        # --- LiDAR points ---
        lidar_path = os.path.join(self.sensor_root, info["lidar_path"])
        if os.path.exists(lidar_path):
            if lidar_path.endswith(".pcd"):
                pts = self._load_pcd(lidar_path)
            else:
                pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        else:
            n = np.random.randint(30000, 60000)
            pts = np.concatenate([
                np.random.uniform(self.pc_range[:3], self.pc_range[3:], (n, 3)),
                np.random.uniform(0, 1, (n, 1)),
                np.zeros((n, 1))
            ], axis=1).astype(np.float32)

        data["points"] = torch.from_numpy(pts).float()

        # --- Occupancy label ---
        if self.load_occ:
            occ_path = info.get("occ_gt_path", "")
            if occ_path and os.path.exists(occ_path):
                sparse_occ = np.load(occ_path)
                dense_occ = sparse_occ_to_dense(sparse_occ, self.occ_size, self.num_classes)
            else:
                dense_occ = np.zeros(self.occ_size, dtype=np.int64)
            data["occ_label"] = torch.from_numpy(dense_occ).long()

        # --- Ego status ---
        e = info["ego_status"]
        data["ego_status"] = torch.tensor([
            e.get("speed", 0), e.get("acceleration", 0),
            e.get("yaw_rate", 0), e.get("steering_angle", 0),
            0, 0, 0
        ], dtype=torch.float32)

        # --- Planning ---
        if self.load_planning:
            gt = info.get("gt_trajectory", np.zeros((self.traj_len, 2), dtype=np.float32))
            ht = info.get("history_trajectory", np.zeros((self.hist_len, 2), dtype=np.float32))
            if isinstance(gt, list):
                gt = np.array(gt, dtype=np.float32)
            if isinstance(ht, list):
                ht = np.array(ht, dtype=np.float32)
            data["gt_trajectory"] = torch.from_numpy(gt).float()
            data["gt_actions"] = torch.from_numpy(
                np.diff(gt, axis=0, prepend=gt[:1])
            ).float()
            data["history_trajectory"] = torch.from_numpy(ht).float()

        data["token"] = info["token"]
        data["img_shape"] = torch.tensor(self.img_size)

        return data

    @staticmethod
    def _load_pcd(path: str) -> np.ndarray:
        """Load a .pcd file (binary format from nuPlan/OpenScene)."""
        with open(path, "rb") as f:
            # Read header
            header = {}
            while True:
                line = f.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("DATA"):
                    data_type = line.split()[-1]
                    break
                if " " in line:
                    key = line.split()[0]
                    val = " ".join(line.split()[1:])
                    header[key] = val

            n_points = int(header.get("POINTS", 0))
            fields = header.get("FIELDS", "x y z intensity").split()
            n_fields = len(fields)

            if data_type == "binary":
                # Build dtype from SIZE + TYPE per field
                # nuPlan PCD: x y z are F/4, intensity lidar_info ring are U/1
                sizes = header.get("SIZE", "4 " * n_fields).split()
                types = header.get("TYPE", "F " * n_fields).split()

                numpy_dtypes = []
                for fn, tp, sz in zip(fields, types, sizes):
                    sz = int(sz)
                    if tp == "F":
                        if sz == 4: numpy_dtypes.append((fn, np.float32))
                        elif sz == 8: numpy_dtypes.append((fn, np.float64))
                        else: numpy_dtypes.append((fn, np.float32))
                    elif tp == "U":
                        if sz == 1: numpy_dtypes.append((fn, np.uint8))
                        elif sz == 2: numpy_dtypes.append((fn, np.uint16))
                        elif sz == 4: numpy_dtypes.append((fn, np.uint32))
                        else: numpy_dtypes.append((fn, np.uint8))
                    elif tp == "I":
                        if sz == 1: numpy_dtypes.append((fn, np.int8))
                        elif sz == 2: numpy_dtypes.append((fn, np.int16))
                        elif sz == 4: numpy_dtypes.append((fn, np.int32))
                        else: numpy_dtypes.append((fn, np.int32))

                dt = np.dtype(numpy_dtypes)
                raw = np.frombuffer(f.read(), dtype=dt, count=n_points)

                # Convert to (N, 5): x, y, z, intensity, ring
                pts = np.zeros((n_points, 5), dtype=np.float32)
                for i, fn in enumerate(["x", "y", "z"]):
                    if fn in raw.dtype.names:
                        pts[:, i] = raw[fn].astype(np.float32)
                if "intensity" in raw.dtype.names:
                    pts[:, 3] = raw["intensity"].astype(np.float32) / 255.0
                elif "i" in raw.dtype.names:
                    pts[:, 3] = raw["i"].astype(np.float32) / 255.0
                if "ring" in raw.dtype.names:
                    pts[:, 4] = raw["ring"].astype(np.float32)

                return pts

            elif data_type == "binary_compressed":
                import struct, lzf
                # compressed size, then uncompressed size
                compressed_size = struct.unpack("I", f.read(4))[0]
                uncompressed_size = struct.unpack("I", f.read(4))[0]
                compressed_data = f.read(compressed_size)
                try:
                    raw_bytes = lzf.decompress(compressed_data, uncompressed_size)
                except Exception:
                    # Fallback: random points
                    n = 40000
                    return np.random.uniform(-40, 40, (n, 5)).astype(np.float32)

                sizes = [int(s) for s in header.get("SIZE", "4 " * n_fields).split()]
                pts = np.zeros((n_points, 5), dtype=np.float32)
                offset = 0
                for i, (fn, sz) in enumerate(zip(fields, sizes)):
                    col_data = np.frombuffer(raw_bytes[offset:offset + n_points * sz],
                                              dtype=np.float32, count=n_points)
                    if fn in ("x", "y", "z"):
                        idx = {"x": 0, "y": 1, "z": 2}[fn]
                        pts[:, idx] = col_data
                    elif fn in ("intensity", "i"):
                        pts[:, 3] = col_data
                    offset += n_points * sz

                return pts

            else:
                # ASCII
                lines_data = f.read().decode("utf-8", errors="ignore").strip().split("\n")
                pts = np.zeros((len(lines_data), 5), dtype=np.float32)
                for i, line in enumerate(lines_data):
                    vals = line.split()
                    for j in range(min(len(vals), 5)):
                        pts[i, j] = float(vals[j])
                return pts


# ============================================================
# PART 4: Verify
# ============================================================

def verify():
    """Load a few samples and check everything works."""
    print("=" * 60)
    print("VERIFYING DATASET LOADING")
    print("=" * 60)

    if not os.path.exists(OUTPUT_PKL):
        print(f"ERROR: {OUTPUT_PKL} not found. Run 'generate' first.")
        return

    ds = OpenSceneOccDatasetV2(
        info_file=OUTPUT_PKL,
        sensor_root=SENSOR_ROOT,
        occ_root=OCC_ROOT,
        load_occ=True,
        load_planning=True,
        max_samples=10,
    )

    print(f"\nDataset size: {len(ds)}")

    for i in range(min(3, len(ds))):
        print(f"\n--- Sample {i} ---")
        sample = ds[i]
        info = ds.infos[i]
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                if k == "occ_label":
                    unique, counts = v.unique(return_counts=True)
                    nonzero = (v > 0).sum().item()
                    total = v.numel()
                    print(f"    occupied voxels: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
                    print(f"    per-class breakdown:")
                    for cls_id, cnt in zip(unique.tolist(), counts.tolist()):
                        name = OCC_CLASSES.get(cls_id, f"cls{cls_id}")
                        if cnt > 0 and cls_id > 0:
                            print(f"      {cls_id:2d} ({name:15s}): {cnt:6d} voxels")
                if k == "points":
                    pts = v.numpy()
                    in_range = ((pts[:, 0] > -40) & (pts[:, 0] < 40) &
                                (pts[:, 1] > -40) & (pts[:, 1] < 40)).sum()
                    print(f"    in range [-40,40]: {in_range}/{len(pts)}")
                if k == "images":
                    print(f"    value range: [{v.min():.1f}, {v.max():.1f}]")
            elif isinstance(v, str):
                print(f"  {k}: '{v}'")

        # Check file existence
        lidar_path = os.path.join(SENSOR_ROOT, info["lidar_path"])
        cam_path = os.path.join(SENSOR_ROOT, info["cams"]["CAM_F0"]["data_path"])
        occ_path = info.get("occ_gt_path", "")
        print(f"  lidar exists: {os.path.exists(lidar_path)}")
        print(f"  cam_f0 exists: {os.path.exists(cam_path)}")
        print(f"  occ exists: {os.path.exists(occ_path) if occ_path else False}")
        print(f"  scene: {info['scene_name']}, map: {info.get('map_location','?')}")

    print("\n" + "=" * 60)
    print("NOTE: OpenScene v1.0 mini occ labels contain DYNAMIC objects only")
    print("(pedestrian, motorcycle, truck, etc). Static classes (driveable,")
    print("sidewalk, vegetation, manmade) are NOT in the sparse labels.")
    print("This is normal for this dataset version.")
    print("=" * 60)
    print("\nVerification PASSED")


# ============================================================
# PART 5: Visualize
# ============================================================

def visualize(num_samples=3, save_dir="vis_dataset"):
    """Visualize samples: cameras, LiDAR BEV, occupancy, 3D boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    print("=" * 60)
    print(f"VISUALIZING {num_samples} SAMPLES")
    print("=" * 60)

    if not os.path.exists(OUTPUT_PKL):
        print(f"ERROR: {OUTPUT_PKL} not found. Run 'generate' first.")
        return

    os.makedirs(save_dir, exist_ok=True)

    ds = OpenSceneOccDatasetV2(
        info_file=OUTPUT_PKL,
        sensor_root=SENSOR_ROOT,
        occ_root=OCC_ROOT,
        load_occ=True,
        load_planning=True,
        max_samples=num_samples * 50,  # pick from more samples
    )

    # Color map for occ classes
    occ_colors = np.array([
        [0, 0, 0],         # 0 empty
        [255, 120, 50],    # 1 barrier
        [255, 192, 203],   # 2 bicycle
        [255, 255, 0],     # 3 bus
        [0, 150, 245],     # 4 car
        [0, 255, 255],     # 5 construction
        [200, 180, 0],     # 6 motorcycle
        [255, 0, 0],       # 7 pedestrian
        [255, 240, 150],   # 8 traffic_cone
        [135, 60, 0],      # 9 trailer
        [160, 32, 240],    # 10 truck
        [255, 0, 255],     # 11 driveable
        [139, 137, 137],   # 12 other_flat
        [75, 0, 75],       # 13 sidewalk
        [150, 240, 80],    # 14 terrain
        [230, 230, 250],   # 15 manmade
        [0, 175, 0],       # 16 vegetation
    ], dtype=np.float32) / 255.0

    # Pick evenly spaced samples
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    for si, idx in enumerate(indices):
        print(f"\nSample {si+1}/{num_samples} (index {idx})")
        sample = ds[idx]
        info = ds.infos[idx]

        fig = plt.figure(figsize=(24, 16))

        # --- Row 1: 4 camera views ---
        cam_order = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
        for ci, cam_name in enumerate(cam_order):
            ax = fig.add_subplot(3, 4, ci + 1)
            # Undo normalization for display
            cam_idx = CAMERA_NAMES.index(cam_name)
            img = sample["images"][cam_idx].permute(1, 2, 0).numpy()
            img += np.array([103.530, 116.280, 123.675])
            img = np.clip(img / 255.0, 0, 1)
            ax.imshow(img)
            ax.set_title(cam_name, fontsize=10)
            ax.axis("off")

        # --- Row 2, Left: LiDAR BEV ---
        ax_bev = fig.add_subplot(3, 4, 5)
        pts = sample["points"].numpy()
        # Filter to range
        mask = (
            (pts[:, 0] > POINT_CLOUD_RANGE[0]) & (pts[:, 0] < POINT_CLOUD_RANGE[3]) &
            (pts[:, 1] > POINT_CLOUD_RANGE[1]) & (pts[:, 1] < POINT_CLOUD_RANGE[4])
        )
        pts_f = pts[mask]
        ax_bev.scatter(pts_f[:, 0], pts_f[:, 1], s=0.1, c=pts_f[:, 2],
                       cmap="viridis", vmin=-1, vmax=3)
        ax_bev.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
        ax_bev.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
        ax_bev.set_aspect("equal")
        ax_bev.set_title(f"LiDAR BEV ({len(pts)} pts)", fontsize=10)

        # --- Row 2, Middle: Occ BEV (top-down, class-colored) ---
        ax_occ_bev = fig.add_subplot(3, 4, 6)
        occ = sample["occ_label"].numpy()  # (X, Y, Z)
        # Top-down: take the dominant non-empty class along Z
        occ_bev = np.zeros((OCC_SIZE[0], OCC_SIZE[1]), dtype=np.int64)
        for z in range(OCC_SIZE[2]):
            mask_z = occ[:, :, z] > 0
            occ_bev[mask_z] = occ[:, :, z][mask_z]

        # Color it
        bev_rgb = occ_colors[occ_bev]
        ax_occ_bev.imshow(bev_rgb, origin="lower", extent=[
            POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3],
            POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4]
        ])
        ax_occ_bev.set_title("Occ BEV (top-down)", fontsize=10)
        ax_occ_bev.set_aspect("equal")

        # --- Row 2, Right: Occ side view (X-Z slice at Y=100) ---
        ax_occ_side = fig.add_subplot(3, 4, 7)
        y_mid = OCC_SIZE[1] // 2
        occ_side = occ[:, y_mid, :]  # (X, Z)
        side_rgb = occ_colors[occ_side]
        ax_occ_side.imshow(side_rgb.transpose(1, 0, 2), origin="lower",
                           aspect="auto",
                           extent=[POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3],
                                   POINT_CLOUD_RANGE[2], POINT_CLOUD_RANGE[5]])
        ax_occ_side.set_title(f"Occ Side View (Y={y_mid})", fontsize=10)

        # --- Row 2, Far Right: Occ class distribution ---
        ax_hist = fig.add_subplot(3, 4, 8)
        unique, counts = np.unique(occ[occ > 0], return_counts=True)
        if len(unique) > 0:
            names = [OCC_CLASSES.get(u, f"cls{u}") for u in unique]
            colors_bar = [occ_colors[u] for u in unique]
            bars = ax_hist.barh(range(len(unique)), counts, color=colors_bar)
            ax_hist.set_yticks(range(len(unique)))
            ax_hist.set_yticklabels(names, fontsize=7)
            ax_hist.set_title("Occ Class Distribution", fontsize=10)
        else:
            ax_hist.text(0.5, 0.5, "No occ labels", ha="center", va="center")

        # --- Row 3: 3D boxes on LiDAR BEV + Info ---
        ax_boxes = fig.add_subplot(3, 4, 9)
        ax_boxes.scatter(pts_f[:, 0], pts_f[:, 1], s=0.1, c="gray", alpha=0.3)

        anns = info.get("anns", {})
        gt_boxes = anns.get("gt_boxes", [])
        gt_names = anns.get("gt_names", [])
        box_colors = {"car": "blue", "truck": "purple", "bus": "yellow",
                      "pedestrian": "red", "bicycle": "pink", "motorcycle": "orange",
                      "barrier": "brown", "traffic_cone": "cyan"}

        for bi, box in enumerate(gt_boxes):
            if isinstance(box, (list, np.ndarray)) and len(box) >= 7:
                x, y, z, l, w, h, yaw = box[:7]
                color = box_colors.get(gt_names[bi] if bi < len(gt_names) else "", "green")
                # Draw rotated rectangle
                cos_a, sin_a = np.cos(yaw), np.sin(yaw)
                corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]])
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                corners_rot = corners @ rot.T + np.array([x, y])
                ax_boxes.plot(corners_rot[:, 0], corners_rot[:, 1], color=color, linewidth=0.5)

        ax_boxes.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
        ax_boxes.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
        ax_boxes.set_aspect("equal")
        n_boxes = len(gt_boxes)
        ax_boxes.set_title(f"3D Boxes ({n_boxes} objects)", fontsize=10)

        # --- Info text ---
        ax_info = fig.add_subplot(3, 4, 10)
        ax_info.axis("off")
        info_text = (
            f"Token: {info['token']}\n"
            f"Scene: {info['scene_name']}\n"
            f"Log: {info['log_name'][:30]}...\n"
            f"Map: {info.get('map_location', '?')}\n"
            f"Speed: {info['ego_status']['speed']:.1f} m/s\n"
            f"Occ on disk: {info.get('_has_occ_on_disk', False)}\n"
            f"LiDAR pts: {len(pts)}\n"
            f"Occ voxels: {(occ > 0).sum()}/{occ.size}\n"
            f"Boxes: {n_boxes}"
        )
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=9, verticalalignment="top", fontfamily="monospace")

        # --- Occ 3D scatter (lightweight) ---
        ax_3d = fig.add_subplot(3, 4, 11, projection="3d")
        # Sample occupied voxels for 3D view
        occupied = np.argwhere(occ > 0)
        if len(occupied) > 5000:
            idx_sub = np.random.choice(len(occupied), 5000, replace=False)
            occupied = occupied[idx_sub]
        if len(occupied) > 0:
            # Map voxel coords to world coords
            vx = POINT_CLOUD_RANGE[0] + occupied[:, 0] * (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / OCC_SIZE[0]
            vy = POINT_CLOUD_RANGE[1] + occupied[:, 1] * (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / OCC_SIZE[1]
            vz = POINT_CLOUD_RANGE[2] + occupied[:, 2] * (POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / OCC_SIZE[2]
            cls_ids = occ[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
            colors_3d = occ_colors[cls_ids]
            ax_3d.scatter(vx, vy, vz, c=colors_3d, s=0.5, alpha=0.5)
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Occ 3D", fontsize=10)

        # --- Legend ---
        ax_leg = fig.add_subplot(3, 4, 12)
        ax_leg.axis("off")
        for ci, (cls_id, cls_name) in enumerate(OCC_CLASSES.items()):
            if cls_id == 0:
                continue
            ax_leg.add_patch(Rectangle((0, 1 - ci * 0.06), 0.08, 0.04,
                                        facecolor=occ_colors[cls_id]))
            ax_leg.text(0.12, 1 - ci * 0.06 + 0.01, f"{cls_id}: {cls_name}",
                       fontsize=7, va="center")
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 1.05)
        ax_leg.set_title("Legend", fontsize=10)

        plt.suptitle(
            f"Sample {si+1}: {info['scene_name']} | {info.get('map_location', '?')} | "
            f"speed={info['ego_status']['speed']:.1f}m/s",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{si:03d}_{info['token'][:8]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\nAll visualizations saved to {save_dir}/")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "verify", "visualize", "all"],
                        help="What to do")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="vis_dataset")
    args = parser.parse_args()

    if args.mode == "generate" or args.mode == "all":
        generate_combined_pkl()

    if args.mode == "verify" or args.mode == "all":
        verify()

    if args.mode == "visualize" or args.mode == "all":
        visualize(args.num_samples, args.save_dir)