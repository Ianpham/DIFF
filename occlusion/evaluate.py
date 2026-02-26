"""
Evaluation script for NavSim PDMS metrics.

Runs inference on navtest split, generates trajectory predictions,
and computes PDMS sub-metrics (NC, DAC, EP, TTC, C).

Usage:
  python evaluate.py --config configs/base.py --ckpt work_dirs/phase3/best.pth
"""

import os
import argparse
import importlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict

from models.gaussian_transdiffuser import GaussianTransDiffuser
from data.openscene_dataset import OpenSceneOccDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="NavSim Evaluation")
    parser.add_argument("--config", type=str, default="configs/base.py")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="eval_results/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vis-samples", type=int, default=0,
                        help="Number of samples to visualize (0=none)")
    return parser.parse_args()


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


@torch.no_grad()
def evaluate(model, dataloader, device, output_dir):
    """Run inference and collect predictions.

    Returns predictions in NavSim submission format.
    """
    model.eval()
    all_predictions = {}

    for batch_idx, batch in enumerate(dataloader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        if "points" in batch:
            batch["points"] = [p.to(device) for p in batch["points"]]

        out = model.forward_inference(batch)

        # Collect per-sample predictions
        top1 = out["top1_trajectory"].cpu().numpy()   # (B, T, 2)
        all_cands = out["trajectories"].cpu().numpy()  # (B, K, T, 2)

        tokens = batch["token"]
        for i, token in enumerate(tokens):
            all_predictions[token] = {
                "trajectory": top1[i].tolist(),          # Selected trajectory
                "candidates": all_cands[i].tolist(),     # All candidates
            }

        if (batch_idx + 1) % 10 == 0:
            print(f"  Evaluated {(batch_idx+1) * dataloader.batch_size} samples...")

    return all_predictions


def compute_metrics_placeholder(predictions: Dict) -> Dict[str, float]:
    """Placeholder for PDMS metric computation.

    In production, use NavSim's official evaluation toolkit:
        from navsim.evaluate import evaluate_predictions
        results = evaluate_predictions(predictions, gt_annotations)

    This placeholder computes simple trajectory statistics.
    """
    all_trajs = []
    for token, pred in predictions.items():
        traj = np.array(pred["trajectory"])
        all_trajs.append(traj)

    all_trajs = np.stack(all_trajs)  # (N, T, 2)

    # Trajectory statistics
    displacements = np.linalg.norm(np.diff(all_trajs, axis=1), axis=-1)
    total_distance = displacements.sum(axis=1)
    avg_speed = total_distance / (all_trajs.shape[1] * 0.5)  # 2Hz → 0.5s per step

    # Smoothness: average jerk
    velocities = np.diff(all_trajs, axis=1)
    accelerations = np.diff(velocities, axis=1)
    jerk = np.linalg.norm(np.diff(accelerations, axis=1), axis=-1).mean()

    metrics = {
        "num_samples": len(predictions),
        "avg_distance_m": float(total_distance.mean()),
        "avg_speed_ms": float(avg_speed.mean()),
        "smoothness_jerk": float(jerk),
        # Placeholder PDMS sub-metrics (replace with official eval)
        "NC": "N/A (use NavSim toolkit)",
        "DAC": "N/A (use NavSim toolkit)",
        "EP": "N/A (use NavSim toolkit)",
        "TTC": "N/A (use NavSim toolkit)",
        "C": "N/A (use NavSim toolkit)",
        "PDMS": "N/A (use NavSim toolkit)",
    }

    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = GaussianTransDiffuser(
        num_gaussians=cfg.num_gaussians,
        embed_dims=cfg.embed_dims,
        num_classes=cfg.num_classes,
        num_encoder_blocks=cfg.num_encoder_blocks,
        num_gaussian_tokens=cfg.num_gaussian_tokens,
        pooling_method=cfg.pooling_method,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        decoder_num_layers=cfg.decoder_num_layers,
        trajectory_length=cfg.trajectory_length,
        action_dim=cfg.action_dim,
        num_diffusion_steps=cfg.num_diffusion_steps,
        num_candidates=cfg.num_trajectory_candidates,
        point_cloud_range=cfg.point_cloud_range,
        occ_size=cfg.occ_size,
    )

    # Load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')}")

    # Build dataset
    dataset = OpenSceneOccDataset(
        data_root=cfg.data_root,
        occ_label_root=cfg.occ_label_root,
        split=args.split,
        point_cloud_range=cfg.point_cloud_range,
        occ_size=cfg.occ_size,
        num_classes=cfg.num_classes,
        trajectory_length=cfg.trajectory_length,
        load_occ=False,
        load_planning=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"\nEvaluating {len(dataset)} samples on {args.split} split...")
    predictions = evaluate(model, dataloader, device, args.output_dir)

    # Compute metrics
    metrics = compute_metrics_placeholder(predictions)

    # Save
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    metric_path = os.path.join(args.output_dir, "metrics.json")
    with open(metric_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nPredictions saved to {pred_path}")
    print(f"Metrics saved to {metric_path}")
    print(f"\nFor official PDMS evaluation, run:")
    print(f"  python -m navsim.evaluate --predictions {pred_path}")


if __name__ == "__main__":
    main()
