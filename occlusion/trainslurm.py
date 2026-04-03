#!/usr/bin/env python3
"""
TransDiffuser Training Script
=============================
Supports: single-GPU, multi-GPU (DDP via torchrun)
All outputs (checkpoints, logs, metrics, TensorBoard) → outputs/

Usage:
    Single GPU:   python trainslurm.py --config config_1xA100.py
    Multi GPU:    torchrun --nproc_per_node=4 trainslurm.py --config config_4xA100.py
    Resume:       python trainslurm.py --config config_1xA100.py --resume outputs/run_XXXX/checkpoints/latest.pt
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# 1. Utilities
# ============================================================

def is_main_process():
    """True on rank 0 or single-GPU."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def setup_distributed():
    """Initialize DDP if launched with torchrun/srun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"[DDP] Initialized: {world_size} GPUs, backend=nccl")
        dist.barrier()
        return local_rank
    else:
        print("[Single GPU] No DDP environment detected")
        return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path):
    """Load a .py config file as a module."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """Format seconds to HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def reduce_tensor(tensor):
    """Average tensor across all DDP processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


# ============================================================
# 2. Output Directory Setup
# ============================================================

class OutputManager:
    """
    Manages all output paths under:
        <project_root>/outputs/run_YYYYMMDD_HHMMSS/
            ├── checkpoints/     # model weights
            ├── logs/            # text logs
            ├── tensorboard/     # TensorBoard events
            ├── metrics/         # JSON metrics per epoch
            ├── config.py        # copy of config used
            └── summary.json     # final training summary
    """

    def __init__(self, project_root, config_path, resume_run=None):
        self.project_root = Path(project_root)
        outputs_base = self.project_root / "outputs"

        if resume_run:
            # Resume into existing run directory
            self.run_dir = Path(resume_run).parent.parent
            assert self.run_dir.exists(), f"Resume dir not found: {self.run_dir}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slurm_id = os.environ.get("SLURM_JOB_ID", "local")
            self.run_dir = outputs_base / f"run_{timestamp}_job{slurm_id}"

        # Create subdirectories (only on rank 0)
        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "checkpoints").mkdir(exist_ok=True)
            (self.run_dir / "logs").mkdir(exist_ok=True)
            (self.run_dir / "tensorboard").mkdir(exist_ok=True)
            (self.run_dir / "metrics").mkdir(exist_ok=True)

            # Copy config
            import shutil
            shutil.copy2(config_path, self.run_dir / "config.py")

        # Wait for rank 0 to create dirs
        if dist.is_initialized():
            dist.barrier()

    @property
    def checkpoint_dir(self):
        return self.run_dir / "checkpoints"

    @property
    def log_dir(self):
        return self.run_dir / "logs"

    @property
    def tb_dir(self):
        return self.run_dir / "tensorboard"

    @property
    def metrics_dir(self):
        return self.run_dir / "metrics"


# ============================================================
# 3. Logger Setup
# ============================================================

def setup_logger(output_manager, name="transdiffuser"):
    """Setup logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear

    formatter = logging.Formatter(
        "[%(asctime)s] [Rank %(rank)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = get_rank()
            return True

    # Console handler (all ranks)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    ch.setFormatter(formatter)
    ch.addFilter(RankFilter())
    logger.addHandler(ch)

    # File handler (rank 0 only)
    if is_main_process():
        fh = logging.FileHandler(output_manager.log_dir / "train.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        fh.addFilter(RankFilter())
        logger.addHandler(fh)

    return logger


# ============================================================
# 4. Metrics Tracker
# ============================================================

class MetricsTracker:
    """Track and save metrics per epoch + TensorBoard logging."""

    def __init__(self, output_manager, enabled=True):
        self.output_manager = output_manager
        self.enabled = enabled and is_main_process()
        self.history = {"phases": {}}
        self.writer = None

        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(output_manager.tb_dir))

    def log_epoch(self, phase, epoch, metrics, lr, epoch_time):
        """Log one epoch's metrics."""
        if not self.enabled:
            return

        phase_key = f"phase{phase}"
        if phase_key not in self.history["phases"]:
            self.history["phases"][phase_key] = []

        record = {
            "epoch": epoch,
            "lr": lr,
            "epoch_time_sec": round(epoch_time, 1),
            "timestamp": datetime.now().isoformat(),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self.history["phases"][phase_key].append(record)

        # TensorBoard
        global_step = epoch
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"phase{phase}/{k}", v, global_step)
        self.writer.add_scalar(f"phase{phase}/lr", lr, global_step)
        self.writer.add_scalar(f"phase{phase}/epoch_time", epoch_time, global_step)
        self.writer.flush()

        # Save per-epoch JSON
        epoch_file = self.output_manager.metrics_dir / f"phase{phase}_epoch{epoch:04d}.json"
        with open(epoch_file, "w") as f:
            json.dump(record, f, indent=2)

    def log_gpu_stats(self, phase, epoch):
        """Log GPU memory usage."""
        if not self.enabled:
            return
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_alloc = torch.cuda.max_memory_allocated() / 1e9
        self.writer.add_scalar(f"phase{phase}/gpu_mem_allocated_gb", allocated, epoch)
        self.writer.add_scalar(f"phase{phase}/gpu_mem_reserved_gb", reserved, epoch)
        self.writer.add_scalar(f"phase{phase}/gpu_mem_peak_gb", max_alloc, epoch)

    def save_summary(self, total_time):
        """Save final training summary."""
        if not self.enabled:
            return
        self.history["total_time_sec"] = round(total_time, 1)
        self.history["total_time_human"] = format_time(total_time)
        self.history["finished_at"] = datetime.now().isoformat()
        self.history["num_gpus"] = get_world_size()
        self.history["gpu_name"] = torch.cuda.get_device_name(0)

        summary_path = self.output_manager.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()


# ============================================================
# 5. Checkpoint Manager
# ============================================================

class CheckpointManager:
    """Save/load checkpoints with best-model tracking."""

    def __init__(self, output_manager, keep_last_n=3):
        self.ckpt_dir = output_manager.checkpoint_dir
        self.keep_last_n = keep_last_n
        self.best_metric = float("inf")
        self.saved_ckpts = []

    def save(self, model, optimizer, scaler, phase, epoch, metrics, is_best=False):
        """Save checkpoint (rank 0 only)."""
        if not is_main_process():
            return

        # Unwrap DDP
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        state = {
            "phase": phase,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "best_metric": self.best_metric,
            "timestamp": datetime.now().isoformat(),
        }

        # Save latest (always overwritten)
        latest_path = self.ckpt_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save numbered checkpoint
        ckpt_path = self.ckpt_dir / f"phase{phase}_epoch{epoch:04d}.pt"
        torch.save(state, ckpt_path)
        self.saved_ckpts.append(ckpt_path)

        # Clean old checkpoints (keep last N)
        while len(self.saved_ckpts) > self.keep_last_n:
            old = self.saved_ckpts.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink()

        # Save best
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)

    def load(self, path, model, optimizer=None, scaler=None):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle DDP wrapped model
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        self.best_metric = ckpt.get("best_metric", float("inf"))
        return ckpt["phase"], ckpt["epoch"]


# ============================================================
# 6. Training Loop
# ============================================================

def train_one_epoch(model, dataloader, optimizer, scaler, scheduler, phase_cfg,
                    device, epoch, logger, use_amp=True):
    """Train one epoch with AMP + gradient clipping."""
    model.train()
    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            batch = [v.to(device, non_blocking=True) if torch.is_tensor(v) else v for v in batch]

        optimizer.zero_grad(set_to_none=True)

        # Forward with AMP
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            # ============================================
            # TODO: Replace with your actual model forward
            # output = model(batch)
            # loss = compute_loss(output, batch, phase_cfg["loss_weights"])
            # ============================================

            # --- PLACEHOLDER: remove this block and use your model ---
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_losses = {"placeholder": 0.0}
            # --- END PLACEHOLDER ---

        # Backward with AMP
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if phase_cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), phase_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if phase_cfg.get("grad_clip", 0) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), phase_cfg["grad_clip"])
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accumulate metrics
        total_loss += loss.item()
        for k, v in batch_losses.items():
            loss_components[k] = loss_components.get(k, 0.0) + (v if isinstance(v, float) else v.item())
        num_batches += 1

        # Log every N batches
        if batch_idx % 50 == 0 and is_main_process():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"  [Phase {phase_cfg.get('_phase_num', '?')}] "
                f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | GPU Mem: {mem_gb:.1f}GB"
            )

    # Average losses
    avg_loss = total_loss / max(num_batches, 1)
    avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}

    # Reduce across GPUs
    avg_loss_tensor = reduce_tensor(torch.tensor(avg_loss, device=device))

    metrics = {"loss": avg_loss_tensor.item(), **avg_components}
    return metrics


def build_scheduler(optimizer, phase_cfg, steps_per_epoch):
    """Build LR scheduler based on config."""
    sched_type = phase_cfg.get("scheduler", "cosine")
    epochs = phase_cfg["epochs"]
    warmup_epochs = phase_cfg.get("warmup_epochs", 0)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if sched_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
    elif sched_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=phase_cfg["lr"],
            total_steps=total_steps,
            pct_start=max(warmup_epochs / epochs, 0.05),
            anneal_strategy="cos",
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1.0)

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )
    return main_scheduler


def run_phase(phase_num, phase_cfg, model, dataset, device, output_mgr,
              ckpt_mgr, metrics_tracker, logger, use_amp=True, resume_epoch=0):
    """Run one training phase (phase1, phase2, or phase3)."""
    phase_cfg["_phase_num"] = phase_num
    epochs = phase_cfg["epochs"]
    is_distributed = dist.is_initialized()

    logger.info(f"{'='*60}")
    logger.info(f"PHASE {phase_num} — {epochs} epochs, bs={phase_cfg['batch_size']}, lr={phase_cfg['lr']}")
    logger.info(f"Loss weights: {phase_cfg.get('loss_weights', {})}")
    if is_distributed:
        eff_bs = phase_cfg["batch_size"] * get_world_size()
        logger.info(f"Effective batch size: {phase_cfg['batch_size']} × {get_world_size()} = {eff_bs}")
    logger.info(f"{'='*60}")

    # DataLoader
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=phase_cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=getattr(phase_cfg, "num_workers", 8),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_cfg["lr"],
        weight_decay=phase_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    scheduler = build_scheduler(optimizer, phase_cfg, len(dataloader))

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training loop
    for epoch in range(resume_epoch, epochs):
        epoch_start = time.time()

        if sampler is not None:
            sampler.set_epoch(epoch)

        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()

        # Train
        metrics = train_one_epoch(
            model, dataloader, optimizer, scaler, scheduler,
            phase_cfg, device, epoch, logger, use_amp
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        if is_main_process():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            logger.info(
                f"Phase {phase_num} Epoch {epoch}/{epochs-1} | "
                f"Loss: {metrics['loss']:.4f} | LR: {current_lr:.2e} | "
                f"Time: {format_time(epoch_time)} | Peak GPU: {peak_mem:.1f}GB"
            )
            metrics_tracker.log_epoch(phase_num, epoch, metrics, current_lr, epoch_time)
            metrics_tracker.log_gpu_stats(phase_num, epoch)

        # Checkpoint every 5 epochs + last epoch
        if epoch % 5 == 0 or epoch == epochs - 1:
            is_best = metrics["loss"] < ckpt_mgr.best_metric
            if is_best:
                ckpt_mgr.best_metric = metrics["loss"]
            ckpt_mgr.save(model, optimizer, scaler, phase_num, epoch, metrics, is_best)
            if is_main_process():
                logger.info(f"  Checkpoint saved (best={is_best})")

    logger.info(f"Phase {phase_num} complete!")


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TransDiffuser Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config .py file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--phase", type=int, default=None, help="Run only this phase (1, 2, or 3)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # --- Setup DDP ---
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # --- A100 TF32 optimization ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # --- Load config ---
    cfg = load_config(args.config)

    # --- Output manager ---
    project_root = Path(__file__).parent
    output_mgr = OutputManager(project_root, args.config, resume_run=args.resume)

    # --- Logger ---
    logger = setup_logger(output_mgr)
    logger.info(f"Output directory: {output_mgr.run_dir}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device} | World size: {get_world_size()}")
    logger.info(f"AMP: {'disabled' if args.no_amp else 'enabled (float16)'}")
    logger.info(f"TF32: enabled")

    # --- Metrics + Checkpoints ---
    metrics_tracker = MetricsTracker(output_mgr)
    ckpt_mgr = CheckpointManager(output_mgr, keep_last_n=5)

    # ============================================
    # TODO: Build your model and dataset here
    # ============================================
    #
    # from your_model import GaussianTransDiffuser
    # from your_dataset import NavSimDataset
    #
    # model = GaussianTransDiffuser(cfg).to(device)
    # train_dataset = NavSimDataset(cfg)
    #
    # --- PLACEHOLDER ---
    model = nn.Linear(10, 10).to(device)
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    # --- END PLACEHOLDER ---

    # Log model info
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Wrap with DDP
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=getattr(cfg, "find_unused_parameters", False),
        )
        logger.info("Model wrapped with DistributedDataParallel")

    # Optional: torch.compile for extra speed (PyTorch 2.0+)
    # model = torch.compile(model)

    # --- Resume ---
    resume_phase, resume_epoch = 1, 0
    if args.resume:
        resume_phase, resume_epoch = ckpt_mgr.load(args.resume, model)
        resume_epoch += 1  # start from next epoch
        logger.info(f"Resumed from phase {resume_phase}, epoch {resume_epoch}")

    # --- Run training phases ---
    use_amp = not args.no_amp
    total_start = time.time()

    phases_to_run = [args.phase] if args.phase else [1, 2, 3]

    for phase_num in phases_to_run:
        phase_cfg = getattr(cfg, f"phase{phase_num}", None)
        if phase_cfg is None:
            logger.warning(f"Phase {phase_num} not found in config, skipping")
            continue

        start_epoch = resume_epoch if phase_num == resume_phase else 0
        run_phase(
            phase_num, phase_cfg, model, train_dataset, device,
            output_mgr, ckpt_mgr, metrics_tracker, logger,
            use_amp=use_amp, resume_epoch=start_epoch,
        )

    # --- Final summary ---
    total_time = time.time() - total_start
    metrics_tracker.save_summary(total_time)
    logger.info(f"{'='*60}")
    logger.info(f"TRAINING COMPLETE — Total time: {format_time(total_time)}")
    logger.info(f"All outputs saved to: {output_mgr.run_dir}")
    logger.info(f"{'='*60}")

    metrics_tracker.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()