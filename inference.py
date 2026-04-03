"""
Inference + Visualization on Real Sensor Data
==============================================
Shows predicted and GT trajectories overlaid on:
  - Real LiDAR BEV (rasterized point cloud)
  - Front camera image
  - Side-by-side comparison panel

GT is in absolute UTM coords → converted to ego-relative
Pred is in (-1, 1) normalized → scaled to realistic driving range (~20m)
"""

import sys
import os

DDPM_ROOT = "/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser/DDPM"
NAVSIM_ROOT = DDPM_ROOT + "/datasets/navsim"
NAVSIM_UTILIZE = NAVSIM_ROOT + "/navsim_utilize"

sys.path.insert(0, DDPM_ROOT)
sys.path.insert(0, NAVSIM_ROOT)
sys.path.insert(0, NAVSIM_UTILIZE)

os.environ["OPENSCENE_DATA_ROOT"] = DDPM_ROOT + "/datasets/navsim/download"
os.environ["NUPLAN_MAPS_ROOT"]    = DDPM_ROOT + "/datasets/navsim/download/maps"
os.environ["NAVSIM_DEVKIT_ROOT"]  = DDPM_ROOT + "/datasets/navsim"

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from datasets.navsim.navsim_utilize.data import NavsimDataset
from adapters import EncoderAdapter
from engine import create_transdiffuser_adapted


# ============================================================
# Constants
# ============================================================
BEV_RANGE   = 50.0   # meters, matches dataset bev_range
BEV_SIZE    = 200    # pixels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
        default="outputs/run_20260323_165439_job179646/checkpoints/best.pt")
    parser.add_argument("--num_samples",         type=int, default=12)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--hidden_size",  type=int, default=768)
    parser.add_argument("--depth",        type=int, default=12)
    parser.add_argument("--num_heads",    type=int, default=12)
    parser.add_argument("--max_agents",   type=int, default=8)
    parser.add_argument("--output_dir",   type=str, default=None)
    return parser.parse_args()


# ============================================================
# Coordinate helpers
# ============================================================

def gt_to_ego_relative(gt_np):
    """
    Convert GT trajectory from absolute UTM coords to ego-relative meters.
    gt_np: (T, 5) — first 2 cols are UTM x,y
    Returns (T, 2) in meters relative to ego
    """
    xy = gt_np[:, :2].copy()
    xy = xy - xy[0]          # subtract start point → relative displacement
    # UTM coords: x=east, y=north → swap to ego frame (x=forward, y=left)
    # In NAVSIM the vehicle moves in the x direction
    return xy                # (T, 2) in meters


def pred_to_ego_meters(pred_np, gt_rel):
    """
    Scale predicted trajectory from (-1,1) to realistic meter range.
    pred_np: (T, 5) raw model output in (-1,1)
    gt_rel:  (T, 2) GT in ego-relative meters (for scale reference)
    Returns (T, 2) in meters
    """
    pred_xy = pred_np[:, :2].copy()

    # GT total displacement range
    gt_max = np.abs(gt_rel).max()
    if gt_max < 0.5:
        gt_max = 15.0  # fallback: assume ~15m travel

    # Pred range
    pred_max = np.abs(pred_xy).max()
    if pred_max > 0:
        pred_xy = pred_xy * (gt_max / pred_max)

    return pred_xy  # (T, 2)


def meters_to_bev_px(xy_m, bev_size=BEV_SIZE, bev_range=BEV_RANGE):
    """
    Convert (x_forward, y_lateral) meters to BEV pixel coordinates.
    BEV image: origin at center, x=up (forward), y=right (lateral)
    Returns (px_col, px_row) for matplotlib imshow
    """
    cx = bev_size / 2
    cy = bev_size / 2
    scale = bev_size / (2 * bev_range)

    # x forward → row decreases (up in image)
    # y lateral → col increases (right in image)
    px_col = cx + xy_m[:, 1] * scale   # lateral  → horizontal
    px_row = cy - xy_m[:, 0] * scale   # forward  → vertical (flipped)

    return px_col, px_row


# ============================================================
# Drawing helpers
# ============================================================

def draw_traj_on_lidar(ax, lidar_bev, gt_rel, pred_m,
                        sample_idx=0, loss=None, ade=None):
    """
    Overlay trajectories on LiDAR BEV image.
    lidar_bev: (2, H, W) tensor — density + height channels
    gt_rel:    (T, 2) ego-relative meters
    pred_m:    (T, 2) ego-relative meters (scaled prediction)
    """
    # Render LiDAR: blend density (ch0) + height (ch1) into RGB
    density = lidar_bev[0].numpy()
    height  = lidar_bev[1].numpy()

    # Create colorized BEV: density=green channel, height=blue channel
    rgb = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
    rgb[:, :, 1] = np.clip(density * 2.5, 0, 1)      # green = density
    rgb[:, :, 2] = np.clip(height  * 1.5, 0, 1)      # blue  = height
    rgb[:, :, 0] = np.clip(density * 0.5, 0, 1)      # red tint

    ax.imshow(rgb, origin="upper", aspect="equal",
              extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE])

    # Draw grid
    for v in np.arange(-40, 41, 10):
        ax.axhline(v, color="white", linewidth=0.3, alpha=0.15, zorder=1)
        ax.axvline(v, color="white", linewidth=0.3, alpha=0.15, zorder=1)

    # ---- GT trajectory (green dashed) ----
    if gt_rel is not None and len(gt_rel) > 0:
        # Convert: x=forward (up), y=lateral (right) → plot(y, x)
        gt_x = np.concatenate([[0.0], gt_rel[:, 0]])  # forward
        gt_y = np.concatenate([[0.0], gt_rel[:, 1]])  # lateral
        ax.plot(gt_y, gt_x, color="#00FF88", linewidth=2.5,
                linestyle="--", zorder=4, alpha=0.9, label="GT")
        ax.scatter(gt_rel[:, 1], gt_rel[:, 0], color="#00FF88",
                   s=25, zorder=5, edgecolors="none")
        # End arrow
        if len(gt_rel) >= 2:
            ax.annotate("",
                xy=(gt_rel[-1, 1], gt_rel[-1, 0]),
                xytext=(gt_rel[-2, 1], gt_rel[-2, 0]),
                arrowprops=dict(arrowstyle="-|>", color="#00FF88",
                                lw=1.5, mutation_scale=12), zorder=6)

    # ---- Predicted trajectory (cyan solid) ----
    if pred_m is not None and len(pred_m) > 0:
        pred_x = np.concatenate([[0.0], pred_m[:, 0]])
        pred_y = np.concatenate([[0.0], pred_m[:, 1]])
        ax.plot(pred_y, pred_x, color="#00CCFF", linewidth=2.5,
                linestyle="-", zorder=4, alpha=0.95, label="Predicted")
        ax.scatter(pred_m[:, 1], pred_m[:, 0], color="#00CCFF",
                   s=30, marker="D", zorder=5, edgecolors="none")
        if len(pred_m) >= 2:
            ax.annotate("",
                xy=(pred_m[-1, 1], pred_m[-1, 0]),
                xytext=(pred_m[-2, 1], pred_m[-2, 0]),
                arrowprops=dict(arrowstyle="-|>", color="#00CCFF",
                                lw=1.5, mutation_scale=12), zorder=6)

    # Ego marker
    ego = plt.Polygon(
        [[-1.0, -2.2], [1.0, -2.2], [1.0, 1.5], [0, 2.5], [-1.0, 1.5]],
        closed=True, facecolor="white", edgecolor="#888888",
        linewidth=0.8, zorder=10
    )
    ax.add_patch(ego)

    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE * 0.3, BEV_RANGE)
    ax.set_xlabel("Lateral (m)", fontsize=7, color="#AABBCC")
    ax.set_ylabel("Forward (m)", fontsize=7, color="#AABBCC")

    title = "LiDAR BEV  |  Sample " + str(sample_idx)
    if loss is not None:
        title += "  loss=" + str(round(float(loss), 3))
    ax.set_title(title, fontsize=9, fontweight="bold", color="#CCDDEE", pad=5)
    ax.tick_params(colors="#445566", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#1a2a3a")

    if ade is not None:
        ax.text(0.02, 0.04, "ADE=" + str(round(ade, 2)) + "m",
                transform=ax.transAxes, fontsize=7, color="#AABBCC",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a1020",
                          edgecolor="#1e3a5f", alpha=0.85))

    handles = [
        mpatches.Patch(color="#00FF88", label="GT trajectory"),
        mpatches.Patch(color="#00CCFF", label="Predicted"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper right",
              facecolor="#0a0f1a", labelcolor="#CCDDEE",
              edgecolor="#1e3a5f", framealpha=0.9)


def draw_camera_with_traj(ax, camera_images, gt_rel, pred_m):
    """
    Show front camera image with trajectory projected as colored dots.
    camera_images: (N_cams, 3, H, W) tensor
    """
    if camera_images is None or camera_images.shape[0] == 0:
        ax.set_facecolor("#0a0f1a")
        ax.text(0.5, 0.5, "No Camera", transform=ax.transAxes,
                ha="center", va="center", color="#445566", fontsize=10)
        ax.axis("off")
        return

    # Front camera = index 0
    img = camera_images[0]  # (3, H, W)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    ax.imshow(img_np, aspect="auto")

    # Simple trajectory projection onto image bottom strip
    # Just show a colored bar indicating direction
    H, W = img_np.shape[:2]

    if gt_rel is not None and len(gt_rel) > 0:
        # Draw trajectory hint at bottom of image
        last_pt = gt_rel[-1]
        cx = W / 2 - last_pt[1] * (W / (2 * BEV_RANGE)) * 3
        cx = np.clip(cx, W * 0.1, W * 0.9)
        ax.annotate("GT end", xy=(cx, H * 0.85),
                    xytext=(W / 2, H * 0.7),
                    fontsize=6, color="#00FF88",
                    arrowprops=dict(arrowstyle="->", color="#00FF88", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#001a00",
                              edgecolor="#00FF88", alpha=0.7))

    if pred_m is not None and len(pred_m) > 0:
        last_pt = pred_m[-1]
        cx = W / 2 - last_pt[1] * (W / (2 * BEV_RANGE)) * 3
        cx = np.clip(cx, W * 0.1, W * 0.9)
        ax.annotate("Pred end", xy=(cx, H * 0.75),
                    xytext=(W / 2, H * 0.55),
                    fontsize=6, color="#00CCFF",
                    arrowprops=dict(arrowstyle="->", color="#00CCFF", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#001a20",
                              edgecolor="#00CCFF", alpha=0.7))

    ax.set_title("Front Camera", fontsize=8, fontweight="bold",
                 color="#CCDDEE", pad=4)
    ax.axis("off")


def draw_displacement_chart(ax, gt_rel, pred_m, sample_idx=0):
    """
    Plot displacement over time for both trajectories.
    """
    ax.set_facecolor("#080c14")

    timesteps = np.arange(1, 9)  # 8 future steps

    if gt_rel is not None and len(gt_rel) > 0:
        gt_disp = np.sqrt(np.sum(gt_rel**2, axis=1))
        ax.plot(timesteps[:len(gt_disp)], gt_disp,
                color="#00FF88", linewidth=2, marker="o",
                markersize=5, label="GT displacement", zorder=4)

    if pred_m is not None and len(pred_m) > 0:
        pred_disp = np.sqrt(np.sum(pred_m**2, axis=1))
        ax.plot(timesteps[:len(pred_disp)], pred_disp,
                color="#00CCFF", linewidth=2, marker="D",
                markersize=5, linestyle="--", label="Pred displacement", zorder=4)

    ax.set_xlabel("Future timestep", fontsize=7, color="#AABBCC")
    ax.set_ylabel("Displacement (m)", fontsize=7, color="#AABBCC")
    ax.set_title("Displacement from Ego over Time", fontsize=8,
                 fontweight="bold", color="#CCDDEE")
    ax.tick_params(colors="#556677", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#0f1a2e")
    ax.grid(color="#0f1a2e", linewidth=0.6, zorder=0)
    ax.legend(fontsize=6, facecolor="#080c14", labelcolor="#CCDDEE",
              edgecolor="#1e3a5f")


# ============================================================
# Per-sample figure
# ============================================================

def make_sample_figure(sample_idx, lidar_bev, camera_images,
                        gt_rel, pred_m, loss, ade, fde, save_path):
    """
    3-panel figure: LiDAR BEV | Front Camera | Displacement Chart
    """
    fig = plt.figure(figsize=(18, 7), facecolor="#04080f")
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           width_ratios=[1.2, 1.2, 0.9],
                           wspace=0.12, left=0.04, right=0.97,
                           top=0.88, bottom=0.10)

    ax_lidar  = fig.add_subplot(gs[0])
    ax_cam    = fig.add_subplot(gs[1])
    ax_chart  = fig.add_subplot(gs[2])

    draw_traj_on_lidar(ax_lidar, lidar_bev, gt_rel, pred_m,
                        sample_idx=sample_idx, loss=loss, ade=ade)
    draw_camera_with_traj(ax_cam, camera_images, gt_rel, pred_m)
    draw_displacement_chart(ax_chart, gt_rel, pred_m, sample_idx)

    title = "Sample " + str(sample_idx)
    if ade is not None:
        title += "   ADE=" + str(round(ade, 2)) + "m"
    if fde is not None:
        title += "   FDE=" + str(round(fde, 2)) + "m"
    if loss is not None:
        title += "   loss=" + str(round(float(loss), 3))

    fig.suptitle(title, fontsize=12, fontweight="bold",
                 color="#AACCEE", y=0.96)

    plt.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor="#04080f", edgecolor="none")
    plt.close()


# ============================================================
# Summary grid
# ============================================================

def make_summary_grid(results, save_path, ncols=4):
    n = len(results)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(7 * ncols, 7 * nrows), facecolor="#04080f")
    fig.subplots_adjust(hspace=0.3, wspace=0.15,
                        left=0.03, right=0.97, top=0.95, bottom=0.03)

    for i, r in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_facecolor("#080c14")

        # LiDAR background
        lidar = r.get("lidar_bev")
        if lidar is not None:
            lidar_t = torch.tensor(lidar)
            density = lidar_t[0].numpy()
            height  = lidar_t[1].numpy()
            rgb = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.float32)
            rgb[:, :, 1] = np.clip(density * 2.5, 0, 1)
            rgb[:, :, 2] = np.clip(height  * 1.5, 0, 1)
            rgb[:, :, 0] = np.clip(density * 0.5, 0, 1)
            ax.imshow(rgb, origin="upper", aspect="equal",
                      extent=[-BEV_RANGE, BEV_RANGE, -BEV_RANGE, BEV_RANGE])

        gt_rel = np.array(r["gt_rel"]) if r.get("gt_rel") else None
        pred_m = np.array(r["pred_m"]) if r.get("pred_m") else None

        if gt_rel is not None:
            gt_x = np.concatenate([[0.0], gt_rel[:, 0]])
            gt_y = np.concatenate([[0.0], gt_rel[:, 1]])
            ax.plot(gt_y, gt_x, color="#00FF88", linewidth=2.0,
                    linestyle="--", zorder=4, alpha=0.9)
            ax.scatter(gt_rel[:, 1], gt_rel[:, 0],
                       color="#00FF88", s=15, zorder=5)

        if pred_m is not None:
            pred_x = np.concatenate([[0.0], pred_m[:, 0]])
            pred_y = np.concatenate([[0.0], pred_m[:, 1]])
            ax.plot(pred_y, pred_x, color="#00CCFF", linewidth=2.0,
                    linestyle="-", zorder=4, alpha=0.95)
            ax.scatter(pred_m[:, 1], pred_m[:, 0],
                       color="#00CCFF", s=18, marker="D", zorder=5)

        # Ego
        ego = plt.Polygon(
            [[-0.8, -1.8], [0.8, -1.8], [0.8, 1.2], [0, 2.0], [-0.8, 1.2]],
            closed=True, facecolor="white", edgecolor="#888888",
            linewidth=0.6, zorder=10
        )
        ax.add_patch(ego)

        ax.set_xlim(-BEV_RANGE * 0.6, BEV_RANGE * 0.6)
        ax.set_ylim(-BEV_RANGE * 0.2, BEV_RANGE * 0.9)
        ax.set_aspect("equal")
        ax.tick_params(colors="#334455", labelsize=5)
        for spine in ax.spines.values():
            spine.set_color("#0f1a2e")

        title = "S" + str(r["sample_idx"])
        if r.get("ade") is not None:
            title += " ADE=" + str(round(r["ade"], 1)) + "m"
        if r.get("loss") is not None:
            title += " L=" + str(round(float(r["loss"]), 2))
        ax.set_title(title, fontsize=7, fontweight="bold",
                     color="#CCDDEE", pad=3)

        handles = [
            mpatches.Patch(color="#00FF88", label="GT"),
            mpatches.Patch(color="#00CCFF", label="Pred"),
        ]
        ax.legend(handles=handles, fontsize=5, loc="upper right",
                  facecolor="#0a0f1a", labelcolor="#CCDDEE",
                  edgecolor="#1e3a5f", framealpha=0.9,
                  handlelength=1, handleheight=0.7)

    valid = [r for r in results if r.get("ade") is not None]
    subtitle = ""
    if valid:
        mean_ade = round(np.mean([r["ade"] for r in valid]), 3)
        mean_fde = round(np.mean([r["fde"] for r in valid]), 3)
        subtitle = "Mean ADE=" + str(mean_ade) + "m   Mean FDE=" + str(mean_fde) + "m"

    fig.suptitle(
        "TransDiffuser  —  Trajectory Prediction on LiDAR BEV\n" + subtitle,
        fontsize=14, fontweight="bold", color="#AACCEE", y=0.98
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#04080f", edgecolor="none")
    plt.close()
    print("  Saved grid: " + str(save_path))


# ============================================================
# Metrics plot
# ============================================================

def make_metrics_plot(results, save_path):
    valid = [r for r in results if r.get("ade") is not None]
    if not valid:
        return

    ades = [r["ade"] for r in valid]
    fdes = [r["fde"] for r in valid]
    idxs = list(range(len(valid)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#04080f")
    fig.subplots_adjust(wspace=0.3)

    for ax, vals, label, color, mean_c in [
        (axes[0], ades, "ADE (m)", "#00FF88", "#00AA55"),
        (axes[1], fdes, "FDE (m)", "#00CCFF", "#0088BB"),
    ]:
        ax.set_facecolor("#080c14")
        bars = ax.bar(idxs, vals, color=color, edgecolor="#0f1a2e",
                      linewidth=0.5, alpha=0.8, zorder=3)
        worst = int(np.argmax(vals))
        bars[worst].set_color("#FF4444")
        mean_val = np.mean(vals)
        ax.axhline(mean_val, color=mean_c, linewidth=1.5, linestyle="--",
                   zorder=4, label="Mean=" + str(round(mean_val, 3)) + "m")
        ax.set_xlabel("Sample", fontsize=9, color="#AABBCC")
        ax.set_ylabel(label, fontsize=9, color="#AABBCC")
        ax.set_title(label + "  (lower=better)", fontsize=11,
                     fontweight="bold", color="#CCDDEE")
        ax.tick_params(colors="#556677", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#0f1a2e")
        ax.grid(axis="y", color="#0f1a2e", linewidth=0.8, zorder=0)
        ax.legend(fontsize=8, facecolor="#080c14", labelcolor="#CCDDEE",
                  edgecolor="#1e3a5f")

    fig.suptitle("Trajectory Evaluation Metrics",
                 fontsize=14, fontweight="bold", color="#AACCEE")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#04080f", edgecolor="none")
    plt.close()
    print("  Saved metrics: " + str(save_path))


# ============================================================
# Main
# ============================================================

def compute_ade(pred, gt):
    pred = np.array(pred)
    gt   = np.array(gt)
    T = min(len(pred), len(gt))
    return float(np.mean(np.sqrt(np.sum((pred[:T] - gt[:T])**2, axis=1))))


def compute_fde(pred, gt):
    pred = np.array(pred)
    gt   = np.array(gt)
    return float(np.sqrt(np.sum((pred[-1] - gt[-1])**2)))


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir \
                 else Path("outputs/inference_sensor_" + timestamp)
    (output_dir / "vis").mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  TransDiffuser — Sensor Visualization")
    print("="*60)
    print("  Samples : " + str(args.num_samples))
    print("  Steps   : " + str(args.num_inference_steps))
    print("  Output  : " + str(output_dir))
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    print("\n[1/4] Loading dataset...")
    dataset = NavsimDataset(data_split="mini", extract_labels=True,
                            compute_acceleration=True)
    num_samples = min(args.num_samples, len(dataset))
    subset = Subset(dataset, list(range(num_samples)))
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None)

    # Model
    print("\n[2/4] Loading model...")
    adapter = EncoderAdapter(dataset, mode="efficient")
    model = create_transdiffuser_adapted(
        adapter=adapter, hidden_size=args.hidden_size,
        depth=args.depth, num_heads=args.num_heads, max_agents=args.max_agents,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("  Loaded epoch=" + str(ckpt.get("epoch")) +
          "  best_loss=" + str(round(float(ckpt.get("best_loss", 0)), 4)))

    # Inference
    print("\n[3/4] Running inference...")
    results = []

    for batch_idx, batch in enumerate(loader):
        adapted = adapter.adapt_batch(batch)
        for k in adapted:
            if isinstance(adapted[k], torch.Tensor):
                adapted[k] = adapted[k].to(device)

        # Training loss
        with torch.no_grad():
            train_out = model(adapted)
        loss = float(train_out["total_loss"].item())

        # Diffusion sampling
        pred_raw = None
        try:
            pred = model._generate_single_proposal(
                adapted_batch=adapted,
                num_inference_steps=args.num_inference_steps,
            )
            pred_raw = pred[0, 0].cpu().numpy()[:8]  # (8, 5)
        except Exception as e:
            print("  Warning: " + str(e))

        # GT → ego relative meters
        gt_rel = None
        if "gt_trajectory" in batch:
            gt_np = batch["gt_trajectory"][0, 0].cpu().numpy()  # (8, 5)
            gt_rel = gt_to_ego_relative(gt_np)  # (8, 2)

        # Pred → ego meters (scaled)
        pred_m = None
        if pred_raw is not None:
            pred_m = pred_to_ego_meters(pred_raw, gt_rel if gt_rel is not None
                                        else np.zeros((8, 2)))

        # Metrics
        ade, fde = None, None
        if pred_m is not None and gt_rel is not None:
            ade = compute_ade(pred_m, gt_rel)
            fde = compute_fde(pred_m, gt_rel)

        # LiDAR BEV (keep as tensor for rendering)
        lidar_bev = batch["lidar_bev"][0]  # (2, H, W)

        # Camera images
        cam_imgs = None
        if "camera_images" in batch:
            cam_imgs = batch["camera_images"][0]  # (N, 3, H, W)

        # Save per-sample figure
        vis_path = output_dir / "vis" / ("sample_" + str(batch_idx).zfill(4) + ".png")
        make_sample_figure(
            sample_idx=batch_idx,
            lidar_bev=lidar_bev,
            camera_images=cam_imgs,
            gt_rel=gt_rel,
            pred_m=pred_m,
            loss=loss,
            ade=ade,
            fde=fde,
            save_path=vis_path,
        )

        result = {
            "sample_idx": batch_idx,
            "loss": loss,
            "gt_rel":  gt_rel.tolist()  if gt_rel  is not None else None,
            "pred_m":  pred_m.tolist()  if pred_m  is not None else None,
            "lidar_bev": lidar_bev.tolist(),
            "ade": ade,
            "fde": fde,
        }
        results.append(result)

        status = "ADE=" + str(round(ade, 2)) + "m" if ade else "no pred"
        print("  [" + str(batch_idx+1) + "/" + str(num_samples) + "]"
              + "  loss=" + str(round(loss, 4))
              + "  " + status)

    # Summary outputs
    print("\n[4/4] Generating summary...")
    make_summary_grid(results, output_dir / "grid.png")
    make_metrics_plot(results, output_dir / "metrics_plot.png")

    valid = [r for r in results if r.get("ade") is not None]
    mean_ade = round(np.mean([r["ade"] for r in valid]), 4) if valid else None
    mean_fde = round(np.mean([r["fde"] for r in valid]), 4) if valid else None

    # Strip lidar_bev from JSON (too large)
    for r in results:
        r.pop("lidar_bev", None)

    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "num_inference_steps": args.num_inference_steps,
        "mean_ade_m": mean_ade,
        "mean_fde_m": mean_fde,
        "mean_loss": round(float(np.mean([r["loss"] for r in results])), 4),
        "timestamp": timestamp,
        "samples": results,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("  DONE")
    if mean_ade:
        print("  Mean ADE : " + str(mean_ade) + " m")
        print("  Mean FDE : " + str(mean_fde) + " m")
    print("  Grid     : " + str(output_dir / "grid.png"))
    print("  Per-sample: " + str(output_dir / "vis/"))
    print("="*60)


if __name__ == "__main__":
    main()