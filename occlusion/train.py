"""
Main training script for GaussianFormer3D × TransDiffuser.

Supports 3 training phases:
  Phase 1: Occupancy-only (GaussianFormer3D)
  Phase 2: Planning with frozen Gaussians
  Phase 3: Joint fine-tuning

Usage:
  python train.py --phase 1 --config configs/base.py
  python train.py --phase 2 --config configs/base.py --occ-ckpt work_dirs/phase1/best.pth
  python train.py --phase 3 --config configs/base.py --ckpt work_dirs/phase2/best.pth
"""

import os
import sys
import time
import argparse
import importlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict

from models.gaussian_transdiffuser import GaussianTransDiffuser
from data.openscene_dataset import OpenSceneOccDataset, collate_fn


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GaussianFormer3D × TransDiffuser Training")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3],
                        help="Training phase: 1=occ, 2=frozen+plan, 3=joint")
    parser.add_argument("--config", type=str, default="configs/base.py")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--occ-ckpt", type=str, default=None,
                        help="Phase 1 checkpoint for Phase 2 init")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_config(path: str):
    """Load Python config file as module."""
    spec = importlib.util.spec_from_file_location("config", path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


# ============================================================
# Training loop
# ============================================================

class Trainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.phase = args.phase

        # Work directory
        if args.work_dir:
            self.work_dir = args.work_dir
        else:
            self.work_dir = os.path.join(cfg.work_dir, f"phase{self.phase}")
        os.makedirs(self.work_dir, exist_ok=True)

        # Device
        self.distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
        if self.distributed:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(args.local_rank)
            self.device = torch.device(f"cuda:{args.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Phase config
        phase_cfg = getattr(cfg, f"phase{self.phase}")
        self.epochs = phase_cfg["epochs"]
        self.lr = phase_cfg["lr"]
        self.batch_size = phase_cfg["batch_size"]
        self.grad_clip = phase_cfg.get("grad_clip", 35.0)
        self.loss_weights = phase_cfg.get("loss_weights", {})

        self._build_model()
        self._build_data()
        self._build_optimizer()

    def _build_model(self):
        """Build model and handle phase-specific init/freezing."""
        cfg = self.cfg

        self.model = GaussianTransDiffuser(
            num_gaussians=cfg.num_gaussians,
            embed_dims=cfg.embed_dims,
            num_classes=cfg.num_classes,
            num_encoder_blocks=cfg.num_encoder_blocks,
            num_gaussian_tokens=cfg.num_gaussian_tokens,
            pooling_method=cfg.pooling_method,
            decoder_embed_dim=cfg.decoder_embed_dim,
            decoder_num_heads=cfg.decoder_num_heads,
            decoder_num_layers=cfg.decoder_num_layers,
            decoder_ff_dim=cfg.decoder_ff_dim,
            trajectory_length=cfg.trajectory_length,
            action_dim=cfg.action_dim,
            num_diffusion_steps=cfg.num_diffusion_steps,
            num_candidates=cfg.num_trajectory_candidates,
            point_cloud_range=cfg.point_cloud_range,
            occ_size=cfg.occ_size,
            loss_weights=self.loss_weights,
        )

        # Phase-specific setup
        if self.phase == 2:
            # Load Phase 1 occupancy checkpoint
            if self.args.occ_ckpt:
                state = torch.load(self.args.occ_ckpt, map_location="cpu")
                missing, unexpected = self.model.load_state_dict(
                    state["model"], strict=False,
                )
                if self.rank == 0:
                    print(f"Loaded occ checkpoint. Missing: {len(missing)}, "
                          f"Unexpected: {len(unexpected)}")
            self.model.freeze_gaussian_branch()

        elif self.phase == 3:
            # Load Phase 2 checkpoint
            if self.args.ckpt:
                state = torch.load(self.args.ckpt, map_location="cpu")
                self.model.load_state_dict(state["model"], strict=False)
                if self.rank == 0:
                    print(f"Loaded Phase 2 checkpoint for joint fine-tuning.")
            self.model.unfreeze_gaussian_branch()

        # Resume
        self.start_epoch = 0
        if self.args.ckpt and self.phase != 3:
            state = torch.load(self.args.ckpt, map_location="cpu")
            self.model.load_state_dict(state["model"])
            self.start_epoch = state.get("epoch", 0)

        self.model = self.model.to(self.device)

        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[self.args.local_rank],
                find_unused_parameters=True,
            )

        if self.rank == 0:
            counts = (self.model.module if self.distributed else self.model).count_parameters()
            print("\n=== Parameter Counts ===")
            for name, info in counts.items():
                if name == "TOTAL":
                    print(f"  TOTAL: {info['total']:,}")
                else:
                    print(f"  {name}: {info['total']:,} (trainable: {info['trainable']:,})")
            print()

    def _build_data(self):
        cfg = self.cfg
        load_planning = self.phase >= 2

        self.train_dataset = OpenSceneOccDataset(
            data_root=cfg.data_root,
            occ_label_root=cfg.occ_label_root,
            split="train",
            point_cloud_range=cfg.point_cloud_range,
            occ_size=cfg.occ_size,
            num_classes=cfg.num_classes,
            trajectory_length=cfg.trajectory_length,
            load_occ=(self.phase in (1, 3)),
            load_planning=load_planning,
        )

        sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def _build_optimizer(self):
        model = self.model.module if self.distributed else self.model
        param_groups = model.get_param_groups(phase=self.phase)

        # Apply LR scaling
        for group in param_groups:
            group["lr"] = self.lr * group.pop("lr_scale", 1.0)

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=0.01,
        )

        # Scheduler
        phase_cfg = getattr(self.cfg, f"phase{self.phase}")
        sched_type = phase_cfg.get("scheduler", "cosine")

        if sched_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6,
            )
        elif sched_type == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.lr,
                epochs=self.epochs,
                steps_per_epoch=len(self.train_loader),
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.epochs // 3, gamma=0.1,
            )
        self.sched_type = sched_type

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        total_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            if "points" in batch:
                batch["points"] = [p.to(self.device) for p in batch["points"]]

            # Forward (phase-specific)
            model = self.model.module if self.distributed else self.model
            if self.phase == 1:
                out = model.forward_phase1(batch)
            elif self.phase == 2:
                out = model.forward_phase2(batch)
            else:
                out = model.forward_phase3(batch)

            loss = out["losses"]["total"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip,
                )

            self.optimizer.step()

            if self.sched_type == "onecycle":
                self.scheduler.step()

            # Accumulate metrics
            for k, v in out["losses"].items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1

            # Logging
            if self.rank == 0 and (batch_idx + 1) % 20 == 0:
                avg_loss = total_losses["total"] / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch+1}][{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss: {avg_loss:.4f} lr: {lr:.2e}"
                )

        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
        return avg_losses

    def save_checkpoint(self, epoch: int, losses: Dict[str, float]):
        if self.rank != 0:
            return

        model = self.model.module if self.distributed else self.model
        state = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "losses": losses,
            "phase": self.phase,
        }

        path = os.path.join(self.work_dir, f"epoch_{epoch+1}.pth")
        torch.save(state, path)

        # Also save as latest
        latest = os.path.join(self.work_dir, "latest.pth")
        torch.save(state, latest)
        print(f"  Saved checkpoint: {path}")

    def train(self):
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"  Phase {self.phase} Training")
            print(f"  Epochs: {self.epochs}, LR: {self.lr}")
            print(f"  Batch size: {self.batch_size} × {self.world_size} GPUs")
            print(f"  Work dir: {self.work_dir}")
            print(f"{'='*60}\n")

        best_loss = float("inf")

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            losses = self.train_one_epoch(epoch)
            t1 = time.time()

            if self.sched_type != "onecycle":
                self.scheduler.step()

            if self.rank == 0:
                loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
                print(f"Epoch {epoch+1}/{self.epochs} ({t1-t0:.1f}s) — {loss_str}")

                # Save periodically and best
                if (epoch + 1) % 5 == 0 or (epoch + 1) == self.epochs:
                    self.save_checkpoint(epoch, losses)

                if losses["total"] < best_loss:
                    best_loss = losses["total"]
                    state = {
                        "model": (self.model.module if self.distributed
                                  else self.model).state_dict(),
                        "epoch": epoch,
                        "losses": losses,
                    }
                    torch.save(state, os.path.join(self.work_dir, "best.pth"))
                    print(f"  ★ New best: {best_loss:.4f}")

        if self.rank == 0:
            print(f"\nPhase {self.phase} training complete. Best loss: {best_loss:.4f}")


# ============================================================
# Entry point
# ============================================================

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args, cfg)
    trainer.train()

    if trainer.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
