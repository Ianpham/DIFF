#!/usr/bin/env python3
"""
Paper-quality occupancy visualization — GaussianFormer3D style.

Generates figures like Fig. 5/6 in the GaussianFormer3D paper:
  Row per sample: Front Camera | LiDAR BEV | 3D Occ Render | Occ BEV GT

Also generates:
  - Per-sample detailed view (8 cameras + occ from multiple angles)
  - Dataset-level statistics

Usage:
  python visualize_paper.py --num-samples 4 --save-dir vis_paper
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
# CONFIG
# ============================================================
PKL = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/openscene_infos_train.pkl"
SENSOR_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"
OCC_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"

PC_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
OCC_SIZE = [200, 200, 16]
CAMERA_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

# OpenScene 17-class color palette (matching paper style)
OCC_PALETTE = np.array([
    [0,   0,   0],       # 0  empty (black)
    [255, 120, 50],      # 1  barrier (orange)
    [255, 192, 203],     # 2  bicycle (pink)
    [255, 255, 0],       # 3  bus (yellow)
    [0,   150, 245],     # 4  car (blue)
    [0,   255, 255],     # 5  construction (cyan)
    [200, 180, 0],       # 6  motorcycle (dark yellow)
    [255, 0,   0],       # 7  pedestrian (red)
    [255, 240, 150],     # 8  traffic_cone (light yellow)
    [135, 60,  0],       # 9  trailer (brown)
    [160, 32,  240],     # 10 truck (purple)
    [255, 0,   255],     # 11 driveable (magenta)
    [139, 137, 137],     # 12 other_flat (gray)
    [75,  0,   75],      # 13 sidewalk (dark purple)
    [150, 240, 80],      # 14 terrain (green)
    [230, 230, 250],     # 15 manmade (lavender)
    [0,   175, 0],       # 16 vegetation (dark green)
], dtype=np.float32) / 255.0

OCC_NAMES = {
    0: "empty", 1: "barrier", 2: "bicycle", 3: "bus", 4: "car",
    5: "construction", 6: "motorcycle", 7: "pedestrian", 8: "traffic_cone",
    9: "trailer", 10: "truck", 11: "driveable", 12: "other_flat",
    13: "sidewalk", 14: "terrain", 15: "manmade", 16: "vegetation",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_infos():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def load_image(path):
    from PIL import Image
    if os.path.exists(path):
        return np.array(Image.open(path))
    return np.random.randint(0, 256, (540, 960, 3), dtype=np.uint8)


def load_pcd(path):
    """Load nuPlan PCD binary."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("DATA"):
                break
            parts = line.split(None, 1)
            if len(parts) == 2:
                header[parts[0]] = parts[1]

        n_points = int(header.get("POINTS", "0"))
        fields = header.get("FIELDS", "x y z").split()
        sizes = header.get("SIZE", "").split()
        types = header.get("TYPE", "").split()

        numpy_dtypes = []
        for fn, tp, sz in zip(fields, types, sizes):
            sz = int(sz)
            if tp == "F":
                numpy_dtypes.append((fn, np.float32 if sz == 4 else np.float64))
            elif tp == "U":
                numpy_dtypes.append((fn, {1: np.uint8, 2: np.uint16, 4: np.uint32}.get(sz, np.uint8)))
            elif tp == "I":
                numpy_dtypes.append((fn, {1: np.int8, 2: np.int16, 4: np.int32}.get(sz, np.int32)))

        dt = np.dtype(numpy_dtypes)
        raw = np.frombuffer(f.read(), dtype=dt, count=n_points)

        pts = np.zeros((n_points, 5), dtype=np.float32)
        for i, fn in enumerate(["x", "y", "z"]):
            if fn in raw.dtype.names:
                pts[:, i] = raw[fn].astype(np.float32)
        if "intensity" in raw.dtype.names:
            pts[:, 3] = raw["intensity"].astype(np.float32) / 255.0
        return pts


def load_occ_sparse(path):
    """Load sparse occ and return (N, 3) world coords + (N,) labels."""
    sparse = np.load(path)
    indices = sparse[:, 0].astype(np.int64)
    labels = sparse[:, 1].astype(np.int64)

    X, Y, Z = OCC_SIZE
    voxel_size = np.array([
        (PC_RANGE[3] - PC_RANGE[0]) / X,
        (PC_RANGE[4] - PC_RANGE[1]) / Y,
        (PC_RANGE[5] - PC_RANGE[2]) / Z,
    ])

    # C-order unravel: flat = x * (Y*Z) + y * Z + z
    xi = indices // (Y * Z)
    yi = (indices % (Y * Z)) // Z
    zi = indices % Z

    world_x = PC_RANGE[0] + (xi + 0.5) * voxel_size[0]
    world_y = PC_RANGE[1] + (yi + 0.5) * voxel_size[1]
    world_z = PC_RANGE[2] + (zi + 0.5) * voxel_size[2]

    coords = np.stack([world_x, world_y, world_z], axis=1)
    return coords, labels


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def render_occ_bev(coords, labels, ax, title="Occupancy GT",
                   show_legend=False, pc_range=None):
    """Render occupancy as colored BEV — paper style."""
    if pc_range is None:
        pc_range = PC_RANGE

    ax.set_facecolor("black")

    # Only non-empty
    mask = labels > 0
    if mask.sum() == 0:
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    c = coords[mask]
    l = labels[mask]

    # Sort by class so smaller objects (pedestrians) render on top
    priority = {7: 100, 8: 90, 2: 80, 6: 70, 1: 60, 4: 50, 10: 40, 3: 30, 9: 20}
    sort_key = np.array([priority.get(li, 0) for li in l])
    order = np.argsort(sort_key)
    c, l = c[order], l[order]

    colors = OCC_PALETTE[l]

    ax.scatter(c[:, 1], c[:, 0], c=colors, s=0.3, marker="s",
               edgecolors="none", rasterized=True)

    # Ego car marker
    ax.plot(0, 0, marker="^", color="white", markersize=8, markeredgecolor="gray")

    ax.set_xlim(pc_range[1], pc_range[4])
    ax.set_ylim(pc_range[0], pc_range[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Y (m)", fontsize=8, color="white")
    ax.set_ylabel("X (m)", fontsize=8, color="white")
    for spine in ax.spines.values():
        spine.set_color("gray")


def render_occ_3d(coords, labels, ax, title="3D Occupancy",
                  elev=45, azim=-60, max_points=8000):
    """Render occupancy as 3D colored voxel scatter — paper style."""
    mask = labels > 0
    if mask.sum() == 0:
        return

    c = coords[mask]
    l = labels[mask]

    # Subsample if too many
    if len(c) > max_points:
        idx = np.random.choice(len(c), max_points, replace=False)
        c, l = c[idx], l[idx]

    colors = OCC_PALETTE[l]

    ax.scatter(c[:, 0], c[:, 1], c[:, 2], c=colors, s=0.8,
               marker="s", alpha=0.7, edgecolors="none", depthshade=True)

    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.set_xlim(PC_RANGE[0], PC_RANGE[3])
    ax.set_ylim(PC_RANGE[1], PC_RANGE[4])
    ax.set_zlim(PC_RANGE[2], PC_RANGE[5])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_facecolor("black")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def render_lidar_bev(pts, ax, title="LiDAR Point Cloud",
                     pc_range=None, with_boxes=None, box_names=None):
    """Render LiDAR BEV with optional 3D box overlays."""
    if pc_range is None:
        pc_range = PC_RANGE

    ax.set_facecolor("black")

    mask = (
        (pts[:, 0] > pc_range[0]) & (pts[:, 0] < pc_range[3]) &
        (pts[:, 1] > pc_range[1]) & (pts[:, 1] < pc_range[4])
    )
    p = pts[mask]

    # Color by height
    z_norm = np.clip((p[:, 2] - pc_range[2]) / (pc_range[5] - pc_range[2]), 0, 1)
    ax.scatter(p[:, 1], p[:, 0], s=0.05, c=z_norm, cmap="plasma",
               alpha=0.8, rasterized=True)

    # Draw 3D boxes
    if with_boxes is not None and len(with_boxes) > 0:
        box_colors = {"car": "#0096F5", "truck": "#A020F0", "bus": "#FFFF00",
                      "pedestrian": "#FF0000", "bicycle": "#FFC0CB",
                      "motorcycle": "#C8B400", "barrier": "#FF7832",
                      "traffic_cone": "#FFF096", "trailer": "#873C00",
                      "construction_vehicle": "#00FFFF"}
        for bi, box in enumerate(with_boxes):
            if len(box) < 7:
                continue
            x, y, z, l, w, h, yaw = box[:7]
            name = box_names[bi] if (box_names is not None and len(box_names) > bi) else ""
            color = box_colors.get(name, "#00FF00")
            cos_a, sin_a = np.cos(yaw), np.sin(yaw)
            corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2],
                                [-l/2, w/2], [-l/2, -w/2]])
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners_rot = corners @ rot.T + np.array([x, y])
            ax.plot(corners_rot[:, 1], corners_rot[:, 0],
                    color=color, linewidth=0.4, alpha=0.8)

    ax.plot(0, 0, marker="^", color="lime", markersize=6, markeredgecolor="white")

    ax.set_xlim(pc_range[1], pc_range[4])
    ax.set_ylim(pc_range[0], pc_range[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("gray")


def make_legend(ax, classes_present=None):
    """Draw color legend for occ classes."""
    ax.axis("off")
    all_classes = list(range(1, 17))  # skip empty
    if classes_present is not None:
        all_classes = sorted(set(all_classes) & set(classes_present))

    y = 0.95
    for cls_id in all_classes:
        color = OCC_PALETTE[cls_id]
        name = OCC_NAMES.get(cls_id, f"class {cls_id}")
        ax.add_patch(Rectangle((0.02, y - 0.03), 0.12, 0.025,
                                facecolor=color, edgecolor="gray", linewidth=0.5))
        ax.text(0.18, y - 0.017, f"{cls_id}: {name}", fontsize=7,
                va="center", fontfamily="monospace")
        y -= 0.055

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Legend", fontsize=10, fontweight="bold")


# ============================================================
# FIGURE 1: Paper-style comparison grid (Fig 5/6 style)
# ============================================================

def fig_paper_grid(infos, num_samples=4, save_dir="vis_paper"):
    """
    Generate paper-style figure:
    Each row = one sample
    Columns: Front Camera | LiDAR BEV (with boxes) | Occ BEV | Occ 3D
    """
    os.makedirs(save_dir, exist_ok=True)

    # Pick diverse samples (different scenes)
    seen_scenes = set()
    selected = []
    for info in infos:
        scene = info.get("scene_name", "")
        occ_path = info.get("occ_gt_path", "")
        if scene not in seen_scenes and occ_path and os.path.exists(occ_path):
            seen_scenes.add(scene)
            selected.append(info)
        if len(selected) >= num_samples:
            break

    n = len(selected)
    fig = plt.figure(figsize=(24, 5.5 * n), facecolor="black")
    gs = gridspec.GridSpec(n, 5, width_ratios=[1.2, 1, 1, 1, 0.4],
                           wspace=0.08, hspace=0.15)

    for row, info in enumerate(selected):
        print(f"  Processing sample {row+1}/{n}: {info['scene_name']}")

        # Load data
        cam_path = os.path.join(SENSOR_ROOT, info["cams"]["CAM_F0"]["data_path"])
        lidar_path = os.path.join(SENSOR_ROOT, info["lidar_path"])
        occ_path = info["occ_gt_path"]

        cam_img = load_image(cam_path)
        pts = load_pcd(lidar_path) if os.path.exists(lidar_path) else np.zeros((100, 5))
        occ_coords, occ_labels = load_occ_sparse(occ_path)

        anns = info.get("anns", {})
        gt_boxes = anns.get("gt_boxes", [])
        gt_names = anns.get("gt_names", [])

        # Col 0: Front Camera
        ax_cam = fig.add_subplot(gs[row, 0])
        ax_cam.imshow(cam_img)
        ax_cam.axis("off")
        if row == 0:
            ax_cam.set_title("Front Camera", fontsize=13, fontweight="bold",
                             color="white", pad=10)
        # Scene info overlay
        speed = info["ego_status"].get("speed", 0)
        ax_cam.text(10, cam_img.shape[0] - 10,
                    f"{info['scene_name']} | {info.get('map_location', '?')} | {speed:.1f}m/s",
                    fontsize=7, color="white", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

        # Col 1: LiDAR BEV with boxes
        ax_lidar = fig.add_subplot(gs[row, 1])
        render_lidar_bev(pts, ax_lidar,
                         title="LiDAR + 3D Boxes" if row == 0 else "",
                         with_boxes=gt_boxes, box_names=gt_names)

        # Col 2: Occ BEV
        ax_occ_bev = fig.add_subplot(gs[row, 2])
        render_occ_bev(occ_coords, occ_labels, ax_occ_bev,
                       title="Occupancy GT (BEV)" if row == 0 else "")

        # Col 3: Occ 3D
        ax_occ_3d = fig.add_subplot(gs[row, 3], projection="3d")
        render_occ_3d(occ_coords, occ_labels, ax_occ_3d,
                      title="Occupancy GT (3D)" if row == 0 else "",
                      elev=35, azim=-45)

        # Col 4: Legend (only first row)
        if row == 0:
            ax_leg = fig.add_subplot(gs[row, 4])
            classes_present = np.unique(occ_labels[occ_labels > 0]).tolist()
            make_legend(ax_leg, classes_present)
        else:
            ax_leg = fig.add_subplot(gs[row, 4])
            ax_leg.axis("off")

    save_path = os.path.join(save_dir, "paper_grid.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 2: Detailed per-sample (all 8 cameras + multi-view occ)
# ============================================================

def fig_detailed(infos, num_samples=2, save_dir="vis_paper"):
    """
    Detailed per-sample visualization:
    Row 1: 8 camera views
    Row 2: LiDAR BEV | Occ BEV | Occ 3D (front) | Occ 3D (side)
    Row 3: Occ class distribution + metadata
    """
    os.makedirs(save_dir, exist_ok=True)

    seen_scenes = set()
    selected = []
    for info in infos:
        scene = info.get("scene_name", "")
        occ_path = info.get("occ_gt_path", "")
        if scene not in seen_scenes and occ_path and os.path.exists(occ_path):
            seen_scenes.add(scene)
            selected.append(info)
        if len(selected) >= num_samples:
            break

    for si, info in enumerate(selected):
        print(f"  Detailed sample {si+1}/{len(selected)}: {info['scene_name']}")

        fig = plt.figure(figsize=(28, 18), facecolor="black")

        # Load data
        occ_path = info["occ_gt_path"]
        lidar_path = os.path.join(SENSOR_ROOT, info["lidar_path"])
        occ_coords, occ_labels = load_occ_sparse(occ_path)
        pts = load_pcd(lidar_path) if os.path.exists(lidar_path) else np.zeros((100, 5))
        anns = info.get("anns", {})
        gt_boxes = anns.get("gt_boxes", [])
        gt_names = anns.get("gt_names", [])

        # Row 1: 8 cameras
        for ci, cam_name in enumerate(CAMERA_NAMES):
            ax = fig.add_subplot(3, 8, ci + 1)
            cam_path = os.path.join(SENSOR_ROOT, info["cams"][cam_name]["data_path"])
            img = load_image(cam_path)
            ax.imshow(img)
            ax.set_title(cam_name, fontsize=9, color="white", fontweight="bold")
            ax.axis("off")

        # Row 2: LiDAR | Occ BEV | Occ 3D front | Occ 3D side
        ax_lidar = fig.add_subplot(3, 4, 5)
        render_lidar_bev(pts, ax_lidar, title=f"LiDAR ({len(pts):,} pts)",
                         with_boxes=gt_boxes, box_names=gt_names)

        ax_occ_bev = fig.add_subplot(3, 4, 6)
        render_occ_bev(occ_coords, occ_labels, ax_occ_bev,
                       title="Occupancy GT (BEV)")

        ax_3d_front = fig.add_subplot(3, 4, 7, projection="3d")
        render_occ_3d(occ_coords, occ_labels, ax_3d_front,
                      title="Occ 3D (front)", elev=25, azim=-90)

        ax_3d_side = fig.add_subplot(3, 4, 8, projection="3d")
        render_occ_3d(occ_coords, occ_labels, ax_3d_side,
                      title="Occ 3D (perspective)", elev=45, azim=-45)

        # Row 3: Class histogram | Occ side view | Info | Legend
        ax_hist = fig.add_subplot(3, 4, 9)
        ax_hist.set_facecolor("black")
        unique, counts = np.unique(occ_labels[occ_labels > 0], return_counts=True)
        if len(unique) > 0:
            names = [OCC_NAMES.get(u, f"cls{u}") for u in unique]
            colors = [OCC_PALETTE[u] for u in unique]
            bars = ax_hist.barh(range(len(unique)), counts, color=colors,
                                edgecolor="gray", linewidth=0.5)
            ax_hist.set_yticks(range(len(unique)))
            ax_hist.set_yticklabels(names, fontsize=8, color="white")
            ax_hist.tick_params(colors="white")
            ax_hist.set_title("Class Distribution", fontsize=11,
                              fontweight="bold", color="white")
            for spine in ax_hist.spines.values():
                spine.set_color("gray")

        # Occ X-Z side view (at Y=middle)
        ax_side = fig.add_subplot(3, 4, 10)
        ax_side.set_facecolor("black")
        mask = occ_labels > 0
        if mask.sum() > 0:
            c = occ_coords[mask]
            l = occ_labels[mask]
            # Filter near Y=0 (±5m)
            y_mask = np.abs(c[:, 1]) < 5
            if y_mask.sum() > 0:
                ax_side.scatter(c[y_mask, 0], c[y_mask, 2], c=OCC_PALETTE[l[y_mask]],
                                s=1.5, marker="s", edgecolors="none", rasterized=True)
        ax_side.set_xlim(PC_RANGE[0], PC_RANGE[3])
        ax_side.set_ylim(PC_RANGE[2], PC_RANGE[5])
        ax_side.set_xlabel("X (m)", fontsize=8, color="white")
        ax_side.set_ylabel("Z (m)", fontsize=8, color="white")
        ax_side.set_title("Side View (|Y|<5m)", fontsize=11,
                          fontweight="bold", color="white")
        ax_side.tick_params(colors="white")
        for spine in ax_side.spines.values():
            spine.set_color("gray")

        # Info panel
        ax_info = fig.add_subplot(3, 4, 11)
        ax_info.axis("off")
        speed = info["ego_status"].get("speed", 0)
        n_occ = (occ_labels > 0).sum()
        info_text = (
            f"Token: {info['token']}\n"
            f"Scene: {info['scene_name']}\n"
            f"Log: {info['log_name'][:35]}...\n"
            f"Map: {info.get('map_location', '?')}\n"
            f"Speed: {speed:.1f} m/s\n"
            f"LiDAR pts: {len(pts):,}\n"
            f"Occ voxels: {n_occ:,} / {np.prod(OCC_SIZE):,}\n"
            f"3D Boxes: {len(gt_boxes)}\n"
            f"Occupancy: {100*n_occ/np.prod(OCC_SIZE):.1f}% occupied"
        )
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=10, va="top", color="white",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a1a",
                               edgecolor="gray"))

        # Legend
        ax_leg = fig.add_subplot(3, 4, 12)
        classes_present = np.unique(occ_labels[occ_labels > 0]).tolist()
        make_legend(ax_leg, classes_present)

        plt.suptitle(
            f"{info['scene_name']}  |  {info.get('map_location', '?')}  |  "
            f"speed={speed:.1f}m/s  |  {len(gt_boxes)} objects",
            fontsize=14, fontweight="bold", color="white", y=0.98
        )

        save_path = os.path.join(save_dir, f"detailed_{si:03d}_{info['token'][:8]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="black", edgecolor="none")
        plt.close()
        print(f"  Saved: {save_path}")


# ============================================================
# FIGURE 3: Dataset statistics
# ============================================================

def fig_statistics(infos, save_dir="vis_paper", max_scan=500):
    """Dataset-level statistics: class distribution, speed, points, etc."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Scanning {min(max_scan, len(infos))} samples for statistics...")

    all_classes = []
    all_speeds = []
    all_n_pts = []
    all_n_occ = []
    all_maps = []

    for info in infos[:max_scan]:
        speed = info["ego_status"].get("speed", 0)
        all_speeds.append(speed)
        all_maps.append(info.get("map_location", "unknown"))

        occ_path = info.get("occ_gt_path", "")
        if occ_path and os.path.exists(occ_path):
            sparse = np.load(occ_path)
            labels = sparse[:, 1]
            for l in labels[labels > 0]:
                all_classes.append(l)
            all_n_occ.append((labels > 0).sum())

        lidar_path = os.path.join(SENSOR_ROOT, info["lidar_path"])
        if os.path.exists(lidar_path):
            fsize = os.path.getsize(lidar_path)
            # Approximate point count from file size (15 bytes per point)
            all_n_pts.append(fsize // 15)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="black")

    # Class distribution
    ax = axes[0, 0]
    ax.set_facecolor("#1a1a1a")
    if all_classes:
        unique, counts = np.unique(all_classes, return_counts=True)
        names = [OCC_NAMES.get(u, f"cls{u}") for u in unique]
        colors = [OCC_PALETTE[u] for u in unique]
        ax.barh(range(len(unique)), counts, color=colors, edgecolor="gray")
        ax.set_yticks(range(len(unique)))
        ax.set_yticklabels(names, fontsize=9, color="white")
    ax.set_title("Occupancy Class Distribution (aggregated)", fontsize=12,
                 fontweight="bold", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    # Speed histogram
    ax = axes[0, 1]
    ax.set_facecolor("#1a1a1a")
    ax.hist(all_speeds, bins=30, color="#0096F5", edgecolor="gray", alpha=0.8)
    ax.set_title("Ego Speed Distribution", fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Speed (m/s)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    # Points per frame
    ax = axes[1, 0]
    ax.set_facecolor("#1a1a1a")
    if all_n_pts:
        ax.hist(all_n_pts, bins=30, color="#A020F0", edgecolor="gray", alpha=0.8)
    ax.set_title("LiDAR Points per Frame", fontsize=12, fontweight="bold", color="white")
    ax.set_xlabel("Points", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    # Map distribution
    ax = axes[1, 1]
    ax.set_facecolor("#1a1a1a")
    from collections import Counter
    map_counts = Counter(all_maps)
    if map_counts:
        names = list(map_counts.keys())
        counts = list(map_counts.values())
        ax.barh(range(len(names)), counts, color="#00B300", edgecolor="gray")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace("us-", "").replace("-", " ") for n in names],
                           fontsize=9, color="white")
    ax.set_title("Map Location Distribution", fontsize=12, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("gray")

    plt.suptitle(f"Dataset Statistics ({min(max_scan, len(infos))} samples)",
                 fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "dataset_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="vis_paper")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-detailed", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    print("Loading dataset...")
    infos = load_infos()
    print(f"Total frames: {len(infos)}")

    if not args.skip_grid:
        print(f"\n[1/3] Paper-style grid ({args.num_samples} samples)...")
        fig_paper_grid(infos, args.num_samples, args.save_dir)

    if not args.skip_detailed:
        print(f"\n[2/3] Detailed views ({min(2, args.num_samples)} samples)...")
        fig_detailed(infos, min(2, args.num_samples), args.save_dir)

    if not args.skip_stats:
        print(f"\n[3/3] Dataset statistics...")
        fig_statistics(infos, args.save_dir)

    print(f"\nAll visualizations saved to {args.save_dir}/")


if __name__ == "__main__":
    main()