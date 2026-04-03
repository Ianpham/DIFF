"""
Training script for adapted TransDiffuser.
============================================
Supports: single-GPU, multi-GPU (DDP via torchrun), AMP, TensorBoard, checkpointing.

All outputs → <project>/outputs/run_YYYYMMDD_HHMMSS_jobXXXX/
    ├── checkpoints/     # latest.pt, best.pt, epoch_XXXX.pt
    ├── logs/            # train.log
    ├── tensorboard/     # TensorBoard events
    ├── metrics/         # per-epoch JSON
    ├── config_args.json # args used for this run
    └── summary.json     # final training summary

Usage:
    Single GPU:  python trainslurm.py --dataset basic --num_epochs 120
    Multi GPU:   torchrun --nproc_per_node=4 trainslurm.py --dataset basic --num_epochs 120
    Resume:      python trainslurm.py --resume outputs/run_XXXX/checkpoints/latest.pt

    Key new flags:
        --encoder_level 1       # 0=scene only, 1=+history, 2=+interaction, 3=+full decorr
        --diffusion_steps 10    # paper uses 10
        --decorr_weight 0.02    # paper β=0.02
        --noise_schedule cosine # cosine > linear for trajectory data
        --use_action_space      # convert GT to action-space deltas (paper Eq. 1)
        --scheduler onecycle    # paper uses OneCycle; also supports cosine, cosine_warmup

SLURM:
    sbatch slurm_1gpu.sh
    sbatch slurm_4gpu.sh
"""

import os
import sys
import json
import math
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets.navsim.navsim_utilize.data import (
    NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
)
from adapters import EncoderAdapter
from engine2 import create_transdiffuser_adapted, EvalMetrics

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
# ============================================================
# 0. Pre-training debug checks
# ============================================================

def audit_batch_shapes(adapted_batch, model, logger):
    """Run once on first batch. Catches 90% of shape bugs."""
    logger.info("=" * 60)
    logger.info("BATCH SHAPE AUDIT")
    logger.info("=" * 60)

    # unwrap DDP → TransDiffuserWithDiffusion → TransDiffuserIntegrated
    raw   = model.module if hasattr(model, 'module') else model
    inner = raw.model    if hasattr(raw,   'model')  else raw

    # ── Log all keys in adapted_batch ──
    logger.info("  Adapted batch keys:")
    for key, val in adapted_batch.items():
        if isinstance(val, torch.Tensor):
            logger.info(f"    {key:<24} : Tensor {tuple(val.shape)}  dtype={val.dtype}")
        elif isinstance(val, dict):
            logger.info(f"    {key:<24} : dict   keys={list(val.keys())}")
        elif isinstance(val, (list, tuple)):
            logger.info(f"    {key:<24} : {type(val).__name__:<6} len={len(val)}")
        else:
            logger.info(f"    {key:<24} : {type(val).__name__}  = {val}")

    # ── Required tensor checks ──
    assert 'gt_trajectory' in adapted_batch, \
        f"FATAL: gt_trajectory missing. Keys: {list(adapted_batch.keys())}"
    gt = adapted_batch['gt_trajectory']
    assert isinstance(gt, torch.Tensor), \
        f"FATAL: gt_trajectory is {type(gt).__name__}, expected Tensor"
    logger.info(f"  gt_trajectory  : {tuple(gt.shape)}  (expect [B, T, 2] or [B, T, C])")

    assert 'agent' in adapted_batch, \
        f"FATAL: agent missing. Keys: {list(adapted_batch.keys())}"
    agent = adapted_batch['agent']
    assert isinstance(agent, torch.Tensor), \
        f"FATAL: agent is {type(agent).__name__}, expected Tensor"
    logger.info(f"  agent states   : {tuple(agent.shape)}  (expect [B, N, state_dim])")

    # ── Optional sensor tensors ──
    for key in ['lidar', 'camera', 'bev', 'img']:
        if key in adapted_batch:
            val = adapted_batch[key]
            if isinstance(val, torch.Tensor):
                logger.info(f"  {key:<16} : {tuple(val.shape)}")
            else:
                logger.info(f"  {key:<16} : {type(val).__name__} (not a Tensor, skipping shape)")

    # ── Model config ──
    logger.info(f"  traj_channels  : {inner.traj_channels}")
    logger.info(f"  future_horizon : {inner.future_horizon}")
    logger.info(f"  max_agents     : {inner.max_agents}")
    logger.info(f"  learn_sigma    : {inner.learn_sigma}")
    logger.info(f"  use_action_space: {inner.use_action_space}")
    logger.info(f"  max_timesteps  : {inner.max_timesteps}")
    logger.info(f"  decorr_weight  : {inner.decorr_weight}")
    logger.info(f"  encoder_level  : {raw.encoder_level}")

    # ── Horizon assertion ──
    T_gt = gt.shape[1] if gt.dim() == 3 else gt.shape[-2]
    assert T_gt == inner.future_horizon, (
        f"FATAL: GT horizon {T_gt} != model.future_horizon {inner.future_horizon}"
    )

    logger.info("  ✓ shape checks passed")
    logger.info("=" * 60)


@torch.no_grad()
def test_forward_pass(model, adapted_batch, logger):
    """Verify full forward pass runs and output shapes are correct."""
    logger.info("=" * 60)
    logger.info("FORWARD PASS TEST")
    logger.info("=" * 60)

    model.eval()

    # TransDiffuserWithDiffusion.forward() handles noise internally —
    # just call it the same way the training loop does.
    loss_dict = model(adapted_batch)

    logger.info(f"  total_loss     : {loss_dict['total_loss'].item():.6f}")
    logger.info(f"  diffusion_loss : {loss_dict['diffusion_loss'].item():.6f}")
    logger.info(f"  decorr_loss    : {loss_dict['decorr_loss'].item():.6f}")
    logger.info(f"  ✓ forward pass OK")
    logger.info("=" * 60)

    model.train()

 
def overfit_one_batch(model, adapted_batch, logger, steps=200):
    """
    Overfit a single batch for 200 steps.
    If loss does not reach < 0.1 the model is broken — training is halted.
 
    DDP-safe: saves & restores model state so rank 0's weights don't diverge
    from other ranks.  The actual training loop + broadcast below handles
    re-syncing after this function returns.
    """
    import torch
 
    logger.info("=" * 60)
    logger.info("OVERFIT TEST  (1 batch, 200 steps)")
    logger.info("=" * 60)
 
    # ── Unwrap DDP ──
    raw_model = model.module if hasattr(model, 'module') else model
 
    # ── Save original weights + optimizer state so we can restore later ──
    # This prevents rank 0's weights from diverging after the overfit test.
    saved_state = {
        k: v.clone() for k, v in raw_model.state_dict().items()
    }
 
    raw_model.train()
    device = next(raw_model.parameters()).device
 
    adapted_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in adapted_batch.items()
    }
 
    # Use a throwaway optimizer — never touches the real one
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=1e-3)
    losses = []
    first_loss = None
 
    for step in range(steps):
        optimizer.zero_grad()
        loss_dict = raw_model(adapted_batch)
        loss = loss_dict['total_loss']
 
        if torch.isnan(loss) or torch.isinf(loss):
            # Restore weights before raising so model isn't left corrupted
            raw_model.load_state_dict(saved_state)
            logger.error(f"  step {step}: NaN/Inf loss detected — aborting overfit test")
            raise RuntimeError(f"NaN/Inf loss at overfit step {step}")
 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
 
        val = loss.item()
        losses.append(val)
        if first_loss is None:
            first_loss = val
 
        if step % 20 == 0:
            recent_avg = sum(losses[-10:]) / min(len(losses), 10)
            early_avg  = sum(losses[:10])  / min(len(losses), 10)
            trend = "↓ decreasing" if recent_avg < early_avg else "→ not decreasing"
            logger.info(
                f"  step {step:3d}/{steps} | "
                f"loss={val:.4f} | "
                f"diff={loss_dict['diffusion_loss'].item():.4f} | "
                f"decorr={loss_dict['decorr_loss'].item():.6f} | "
                f"{trend}"
            )
 
    final_loss  = losses[-1]
    avg_last_10 = sum(losses[-10:]) / 10
    reduction   = (first_loss - final_loss) / (first_loss + 1e-8) * 100
 
    logger.info("-" * 60)
    logger.info(f"  first loss     : {first_loss:.4f}")
    logger.info(f"  final loss     : {final_loss:.4f}")
    logger.info(f"  avg last 10    : {avg_last_10:.4f}")
    logger.info(f"  reduction      : {reduction:.1f}%")
 
    # ── CRITICAL: Restore original weights ──
    # The overfit test is diagnostic only — we don't want its 200 steps
    # of Adam updates to persist into the real training run.
    raw_model.load_state_dict(saved_state)
    del saved_state
    logger.info("  (model weights restored to pre-overfit state)")
 
    if avg_last_10 < 0.05:
        logger.info("  ✓ EXCELLENT — model overfits well, safe to train")
    elif avg_last_10 < 0.1:
        logger.info("  ✓ PASSED — model is learning, safe to train")
    elif reduction > 30:
        logger.warning("  ~ PARTIAL — loss reducing but slow, check lr / architecture")
    else:
        logger.error("  ✗ FAILED — model not learning")
        raise RuntimeError(
            f"Overfit test FAILED: {first_loss:.4f} → {final_loss:.4f} "
            f"({reduction:.1f}% reduction). Fix model before full training."
        )
 
    logger.info("=" * 60)
    model.train()


# ============================================================
# 1. Distributed utilities
# ============================================================

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend    = "nccl",
            init_method= "env://",
            world_size = world_size,
            rank       = rank,
            timeout    = timedelta(minutes=30),
        )
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"[DDP] Initialized: {world_size} GPUs, backend=nccl")
        dist.barrier()
        return local_rank
    print("[Single GPU] No DDP environment detected")
    return 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# ============================================================
# 2. Output manager
# ============================================================

class OutputManager:
    def __init__(self, project_root, args, resume_path=None):
        outputs_base = Path(project_root) / "outputs"

        if resume_path:
            self.run_dir = Path(resume_path).parent.parent
            assert self.run_dir.exists(), f"Resume dir not found: {self.run_dir}"
        else:
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            slurm_id   = os.environ.get("SLURM_JOB_ID", "local")
            self.run_dir = outputs_base / f"run_{timestamp}_job{slurm_id}"

        if is_main_process():
            self.run_dir.mkdir(parents=True, exist_ok=True)
            for sub in ["checkpoints", "logs", "tensorboard", "metrics"]:
                (self.run_dir / sub).mkdir(exist_ok=True)
            with open(self.run_dir / "config_args.json", "w") as f:
                json.dump(vars(args), f, indent=2)

        if dist.is_initialized():
            dist.barrier()

    @property
    def checkpoint_dir(self): return self.run_dir / "checkpoints"

    @property
    def log_dir(self): return self.run_dir / "logs"

    @property
    def tb_dir(self): return self.run_dir / "tensorboard"

    @property
    def metrics_dir(self): return self.run_dir / "metrics"


# ============================================================
# 3. Logger
# ============================================================

def setup_logger(output_manager):
    logger = logging.getLogger("transdiffuser")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        "[%(asctime)s] [Rank %(rank)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = get_rank()
            return True

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if is_main_process() else logging.WARNING)
    ch.setFormatter(fmt)
    ch.addFilter(RankFilter())
    logger.addHandler(ch)

    if is_main_process():
        fh = logging.FileHandler(output_manager.log_dir / "train.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        fh.addFilter(RankFilter())
        logger.addHandler(fh)

    return logger


# ============================================================
# 4. Metrics tracker
# ============================================================

class MetricsTracker:
    def __init__(self, output_manager):
        self.output_manager = output_manager
        self.enabled        = is_main_process()
        self.history        = {"epochs": []}
        self.writer         = None

        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(output_manager.tb_dir))

    def log_epoch(self, epoch, metrics, lr, epoch_time):
        if not self.enabled:
            return
        record = {
            "epoch":           epoch,
            "lr":              lr,
            "epoch_time_sec":  round(epoch_time, 1),
            "timestamp":       datetime.now().isoformat(),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        }
        self.history["epochs"].append(record)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)
        self.writer.add_scalar("train/lr",         lr,          epoch)
        self.writer.add_scalar("train/epoch_time", epoch_time,  epoch)
        self.writer.add_scalar(
            "system/gpu_mem_allocated_gb",
            torch.cuda.memory_allocated() / 1e9, epoch
        )
        self.writer.add_scalar(
            "system/gpu_mem_peak_gb",
            torch.cuda.max_memory_allocated() / 1e9, epoch
        )
        self.writer.flush()

        with open(self.output_manager.metrics_dir / f"epoch_{epoch:04d}.json", "w") as f:
            json.dump(record, f, indent=2)

    def save_summary(self, total_time, args):
        if not self.enabled:
            return
        self.history.update({
            "total_time_sec":   round(total_time, 1),
            "total_time_human": format_time(total_time),
            "finished_at":      datetime.now().isoformat(),
            "num_gpus":         get_world_size(),
            "gpu_name":         torch.cuda.get_device_name(0),
            "args":             vars(args),
        })
        with open(self.output_manager.run_dir / "summary.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()


# ============================================================
# 5. Checkpoint manager
# ============================================================

class CheckpointManager:
    def __init__(self, output_manager, keep_last_n=5):
        self.ckpt_dir    = output_manager.checkpoint_dir
        self.keep_last_n = keep_last_n
        self.best_loss   = float("inf")
        self.saved_ckpts = []

    def save(self, model, optimizer, scheduler, scaler, epoch, metrics, is_best=False):
        if not is_main_process():
            return
        model_state = (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        )
        state = {
            "epoch":                epoch,
            "model_state_dict":     model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "metrics":              metrics,
            "best_loss":            self.best_loss,
            "timestamp":            datetime.now().isoformat(),
        }
        torch.save(state, self.ckpt_dir / "latest.pt")

        ckpt_path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, ckpt_path)
        self.saved_ckpts.append(ckpt_path)

        while len(self.saved_ckpts) > self.keep_last_n:
            old = self.saved_ckpts.pop(0)
            if old.exists() and "best" not in old.name:
                old.unlink()

        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")

    def load(self, path, model, optimizer=None, scheduler=None, scaler=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_loss = ckpt.get("best_loss", float("inf"))
        return ckpt["epoch"]


# ============================================================
# 6. Scheduler builder
# ============================================================

def build_scheduler(args, optimizer, steps_per_epoch):
    """
    Build LR scheduler based on args.scheduler.
    Options:
      - 'onecycle'       : OneCycleLR (paper default)
      - 'cosine'         : CosineAnnealingLR (original)
      - 'cosine_warmup'  : Linear warmup + cosine decay
    """
    total_steps = args.num_epochs * steps_per_epoch

    if args.scheduler == 'onecycle':
        # Paper: OneCycle with lr=1e-4, 120 epochs
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr          = args.learning_rate,
            total_steps     = total_steps,
            pct_start       = 0.1,       # 10% warmup
            anneal_strategy = 'cos',
            div_factor      = 10,        # initial_lr = max_lr / 10
            final_div_factor= 100,       # final_lr  = initial_lr / 100
        )

    elif args.scheduler == 'cosine':
        # Original: CosineAnnealingLR (no warmup)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )

    elif args.scheduler == 'cosine_warmup':
        # Linear warmup for warmup_ratio of steps, then cosine decay
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TransDiffuser Training (DDP + AMP)")

    # Dataset
    parser.add_argument("--dataset",    type=str, default="basic",
                        choices=["basic", "enhanced", "phase"])
    parser.add_argument("--data_split", type=str, default="mini")

    # Adapter
    parser.add_argument("--mode", type=str, default="efficient",
                        choices=["auto", "minimal", "efficient", "full"])

    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth",       type=int, default=4)
    parser.add_argument("--num_heads",   type=int, default=4)
    parser.add_argument("--max_agents",  type=int, default=1)

    # Training
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--num_epochs",    type=int,   default=120)      # paper: 120
    parser.add_argument("--learning_rate", type=float, default=1e-4)     # paper: 1e-4
    parser.add_argument("--weight_decay",  type=float, default=1e-5)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--log_interval",  type=int,   default=10)

    # ── Scheduler (NEW) ──
    parser.add_argument("--scheduler", type=str, default="onecycle",
                        choices=["onecycle", "cosine", "cosine_warmup"],
                        help="LR scheduler: onecycle (paper), cosine, cosine_warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for cosine_warmup scheduler")

    # ── Diffusion config (NEW) ──
    parser.add_argument("--diffusion_steps", type=int, default=10,
                        help="Number of diffusion timesteps (paper: 10)")
    parser.add_argument("--noise_schedule", type=str, default="cosine",
                        choices=["linear", "cosine"],
                        help="Noise schedule: cosine (recommended) or linear")

    # ── Encoder level (NEW) ──
    parser.add_argument("--encoder_level", type=int, default=1,
                        choices=[0, 1, 2, 3],
                        help="Encoder feature level: "
                             "0=scene only, 1=+history, "
                             "2=+interaction+block_decorr, 3=+full_decorr")

    # ── Decorrelation (NEW) ──
    parser.add_argument("--decorr_weight", type=float, default=0.02,
                        help="Decorrelation loss weight β (paper: 0.02)")

    # ── Action-space (NEW) ──
    parser.add_argument("--use_action_space", action="store_true", default=True,
                        help="Convert GT to action-space deltas (paper Eq. 1)")
    parser.add_argument("--no_action_space", dest="use_action_space", action="store_false",
                        help="Keep GT as absolute waypoints")

    # ── Temporal downsample (NEW) ──
    parser.add_argument("--use_temporal_downsample", action="store_true", default=False,
                        help="Enable coarse/fine temporal scaling (off by default)")

    # Eval
    parser.add_argument("--eval_every",   type=int, default=5,
                        help="Run full eval every N epochs")
    parser.add_argument("--eval_batches", type=int, default=20,
                        help="Number of batches to use for eval")
    parser.add_argument("--eval_steps",   type=int, default=10,
                        help="DDIM reverse diffusion steps during eval")

    # Infra
    parser.add_argument("--num_workers", type=int,  default=8)
    parser.add_argument("--no_amp",      action="store_true")
    parser.add_argument("--resume",      type=str,  default=None)
    parser.add_argument("--save_every",  type=int,  default=5)

    args = parser.parse_args()

    # ── Distributed setup ────────────────────────────────────────────
    local_rank = setup_distributed()
    device     = torch.device(f"cuda:{local_rank}")

    torch.backends.cuda.matmul.allow_tf32    = True
    torch.backends.cudnn.allow_tf32          = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark           = True

    use_amp = not args.no_amp

    # ── Outputs + logger ─────────────────────────────────────────────
    project_root    = Path(__file__).resolve().parent
    output_mgr      = OutputManager(project_root, args, resume_path=args.resume)
    logger          = setup_logger(output_mgr)
    metrics_tracker = MetricsTracker(output_mgr)
    ckpt_mgr        = CheckpointManager(output_mgr, keep_last_n=5)

    logger.info("=" * 70)
    logger.info("  TransDiffuser Training")
    logger.info("=" * 70)
    logger.info(f"  Output dir      : {output_mgr.run_dir}")
    logger.info(f"  Device          : {device}  |  World size: {get_world_size()}")
    logger.info(f"  AMP             : {'enabled (float16)' if use_amp else 'disabled'}")
    if os.environ.get("SLURM_JOB_ID"):
        logger.info(f"  SLURM Job       : {os.environ['SLURM_JOB_ID']}")
        logger.info(f"  SLURM Node      : {os.environ.get('SLURM_NODELIST', 'N/A')}")
    logger.info(f"  GPU             : {torch.cuda.get_device_name(device)}")
    logger.info(f"  GPU Memory      : {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    logger.info(f"  ── Key config ──")
    logger.info(f"  encoder_level   : {args.encoder_level}")
    logger.info(f"  diffusion_steps : {args.diffusion_steps}")
    logger.info(f"  noise_schedule  : {args.noise_schedule}")
    logger.info(f"  decorr_weight   : {args.decorr_weight}")
    logger.info(f"  use_action_space: {args.use_action_space}")
    logger.info(f"  scheduler       : {args.scheduler}")
    logger.info(f"  num_epochs      : {args.num_epochs}")
    logger.info(f"  learning_rate   : {args.learning_rate}")
    logger.info(f"  Args            : {vars(args)}")

    # ── 1. Dataset ───────────────────────────────────────────────────
    logger.info("Creating dataset...")
    if args.dataset == "basic":
        dataset = NavsimDataset(
            data_split        = args.data_split,
            extract_labels    = True,
            compute_acceleration = True,
        )
    elif args.dataset == "enhanced":
        dataset = EnhancedNavsimDataset(
            data_split        = args.data_split,
            extract_labels    = True,
            extract_route_info= True,
        )
    elif args.dataset == "phase":
        dataset = PhaseNavsimDataset(
            data_split      = args.data_split,
            enable_phase_0  = True,
            extract_labels  = True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"  Dataset: {len(dataset)} samples ({args.dataset}, split={args.data_split})")

    # ── 2. Adapter ───────────────────────────────────────────────────
    logger.info("Creating adapter...")
    adapter = EncoderAdapter(dataset, mode=args.mode)
    if is_main_process():
        adapter.print_summary()

    # ── 3. Model ─────────────────────────────────────────────────────
    logger.info("Creating TransDiffuser...")
    model = create_transdiffuser_adapted(
        adapter              = adapter,
        hidden_size          = args.hidden_size,
        depth                = args.depth,
        num_heads            = args.num_heads,
        max_agents           = args.max_agents,
        # ── NEW configurable params ──
        decorr_weights       = args.decorr_weight,
        diffusion_steps      = args.diffusion_steps,
        noise_schedule       = args.noise_schedule,
        encoder_level        = args.encoder_level,
        use_action_space     = args.use_action_space,
        use_temporal_downsample = args.use_temporal_downsample,
    ).to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Params: {total_params:,} total  |  {trainable_params:,} trainable")

    if dist.is_initialized():
        model = DDP(
            model,
            device_ids          = [local_rank],
            output_device       = local_rank,
            find_unused_parameters = True,
        )
        logger.info("  Wrapped with DistributedDataParallel")

    # ── 4. DataLoader ────────────────────────────────────────────────
    batch_size    = min(adapter.get_optimal_batch_size(), args.batch_size)
    is_distributed= dist.is_initialized()
    sampler       = DistributedSampler(dataset, shuffle=True) if is_distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = (sampler is None),
        sampler         = sampler,
        num_workers     = args.num_workers,
        pin_memory      = True,
        persistent_workers = (args.num_workers > 0),
        prefetch_factor = 2 if args.num_workers > 0 else None,
        drop_last       = True,
        collate_fn      = dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    eff_bs = batch_size * get_world_size()
    steps_per_epoch = len(dataloader)
    logger.info(f"  Batch: {batch_size}/GPU × {get_world_size()} GPU = {eff_bs} effective")
    logger.info(f"  Steps/epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {steps_per_epoch * args.num_epochs}")

    # # ── 5. Pre-training debug checks (rank 0 only) ───────────────────
    # if is_main_process():
    #     logger.info("\n")
    #     logger.info("=" * 70)
    #     logger.info("  PRE-TRAINING DEBUG CHECKS")
    #     logger.info("=" * 70)

    #     _debug_batch = next(iter(dataloader))
    #     _debug_batch = adapter.adapt_batch(_debug_batch)
    #     _debug_batch = {
    #         k: v.to(device) if isinstance(v, torch.Tensor) else v
    #         for k, v in _debug_batch.items()
    #     }

    #     # Print all tensor shapes and stats
    #     logger.info("--- Batch contents ---")
    #     for key, val in _debug_batch.items():
    #         if isinstance(val, torch.Tensor):
    #             logger.info(
    #                 f"  {key:<20} shape={tuple(val.shape)}"
    #                 f"  dtype={val.dtype}"
    #                 f"  nan={torch.isnan(val).any().item()}"
    #                 f"  inf={torch.isinf(val).any().item()}"
    #                 f"  min={val.float().min():.4f}"
    #                 f"  max={val.float().max():.4f}"
    #             )

    #     logger.info("\n--- Check 1: Batch shape audit ---")
    #     audit_batch_shapes(_debug_batch, model, logger)

    #     logger.info("\n--- Check 2: Forward pass test ---")
    #     test_forward_pass(model, _debug_batch, logger)

    #     logger.info("\n--- Check 3: Overfit test ---")
    #     overfit_one_batch(model, _debug_batch, logger, steps=200)

    #     del _debug_batch
    #     logger.info("\n✓ ALL PRE-TRAINING CHECKS PASSED — starting training\n")

    # # All ranks wait for rank 0 debug checks before training starts
    # if dist.is_initialized():
    #     if is_main_process():
    #         logger.info("  Syncing model weights across ranks...")
    #     dist.barrier()  # ← ADD THIS first, so both ranks meet here
    #     raw = model.module if hasattr(model, 'module') else model
    #     for param in raw.parameters():
    #         dist.broadcast(param.data, src=0)
    #     for buf in raw.buffers():
    #         dist.broadcast(buf, src=0)
    #     dist.barrier()
    #     if is_main_process():
    #         logger.info("  ✓ All ranks synchronized")

    # ── 6. Optimizer + scheduler + scaler ────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.learning_rate,
        weight_decay = args.weight_decay,
    )

    scheduler = build_scheduler(args, optimizer, steps_per_epoch)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(f"  Scheduler: {args.scheduler}")

    # ── 7. Resume ────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = ckpt_mgr.load(
            args.resume, model, optimizer, scheduler, scaler
        ) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}, continuing from {start_epoch}")

    # ── 8. Training loop ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  STARTING TRAINING")
    logger.info("=" * 70)

    # unwrap DDP once for eval — DDP model doesn't have .evaluate()
    eval_model = model.module if hasattr(model, 'module') else model

    total_start = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()
        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        torch.cuda.reset_peak_memory_stats()

        running_loss          = 0.0
        running_diffusion     = 0.0
        running_decorr        = 0.0
        running_grad_norm     = 0.0
        running_decorr_levels = {}   # per-level decorr accumulators, reset each epoch
        num_batches           = 0

        for batch_idx, batch in enumerate(dataloader):

            # ── adapt + move to device ────────────────────────────
            adapted_batch = adapter.adapt_batch(batch)
            adapted_batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in adapted_batch.items()
            }

            # ── shape audit on very first batch of training ───────
            if batch_idx == 0 and epoch == start_epoch and is_main_process():
                audit_batch_shapes(adapted_batch, model, logger)

            optimizer.zero_grad(set_to_none=True)

            # ── forward pass ──────────────────────────────────────
            # TransDiffuserWithDiffusion.forward() handles all noise
            # generation internally via self.diffusion.q_sample().
            # Do NOT add noise here — the wrapper does it.
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss_dict = model(adapted_batch)
                loss      = loss_dict["total_loss"]

            # ── backward ──────────────────────────────────────────
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)
                scaler.step(optimizer)
                scaler.update()
                if batch_idx == 0 and is_main_process():
                    logger.info(f"  AMP scaler scale: {scaler.get_scale()}")
            else:
                loss.backward()
                if args.grad_clip > 0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()

            scheduler.step()

            # ── accumulate ────────────────────────────────────────
            diff_val   = loss_dict["diffusion_loss"]
            decorr_val = loss_dict["decorr_loss"]
            running_loss      += loss.item()
            running_diffusion += diff_val.item()   if torch.is_tensor(diff_val)   else float(diff_val)
            running_decorr    += decorr_val.item() if torch.is_tensor(decorr_val) else float(decorr_val)
            running_grad_norm += grad_norm.item()  if torch.is_tensor(grad_norm)  else float(grad_norm)
            num_batches       += 1

            # Accumulate per-level decorrelation breakdown
            decorr_level_keys = [
                k for k in loss_dict if k.startswith('decorr_') and k != 'decorr_loss'
            ]
            for k in decorr_level_keys:
                v = loss_dict[k]
                running_decorr_levels[k] = running_decorr_levels.get(k, 0.0) + (
                    v.item() if torch.is_tensor(v) else float(v)
                )

            # ── per-batch log ─────────────────────────────────────
            if batch_idx % args.log_interval == 0 and is_main_process():
                mem_gb = torch.cuda.memory_allocated() / 1e9
                gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                gn_flag = "⚠ clipped" if gn_val >= args.grad_clip * 0.95 else ""
                decorr_display = (
                    decorr_val.item() if torch.is_tensor(decorr_val) else float(decorr_val)
                )
                # Build per-level string: "fused=0.0012 temporal=0.0008 ..."
                level_parts = []
                for k in decorr_level_keys:
                    v = loss_dict[k]
                    short_name = k.replace('decorr_', '')
                    level_parts.append(
                        f"{short_name}={v.item() if torch.is_tensor(v) else v:.6f}"
                    )
                level_str = " ".join(level_parts) if level_parts else ""

                logger.info(
                    f"  Epoch {epoch+1} [{batch_idx:>4d}/{len(dataloader)}]"
                    f"  loss={loss.item():.4f}"
                    f"  diff={diff_val.item() if torch.is_tensor(diff_val) else diff_val:.4f}"
                    f"  decorr={decorr_display:.6f}"
                    f"  [{level_str}]"
                    f"  gnorm={gn_val:.3f} {gn_flag}"
                    f"  lr={optimizer.param_groups[0]['lr']:.2e}"
                    f"  gpu={mem_gb:.1f}GB"
                )

        # ── epoch summary ─────────────────────────────────────────
        epoch_time    = time.time() - epoch_start
        avg_loss      = running_loss      / max(num_batches, 1)
        avg_diffusion = running_diffusion / max(num_batches, 1)
        avg_decorr    = running_decorr    / max(num_batches, 1)
        avg_grad_norm = running_grad_norm / max(num_batches, 1)

        avg_loss_t   = reduce_tensor(torch.tensor(avg_loss,      device=device)).item()
        avg_diff_t   = reduce_tensor(torch.tensor(avg_diffusion, device=device)).item()
        avg_decorr_t = reduce_tensor(torch.tensor(avg_decorr,    device=device)).item()

        peak_mem   = torch.cuda.max_memory_allocated() / 1e9
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_metrics = {
            "loss":           avg_loss_t,
            "diffusion_loss": avg_diff_t,
            "decorr_loss":    avg_decorr_t,
            "avg_grad_norm":  round(avg_grad_norm, 4),
            "peak_gpu_gb":    round(peak_mem, 2),
        }

        # Add per-level decorr averages to epoch metrics + tensorboard
        for k in running_decorr_levels:
            avg_val = running_decorr_levels[k] / max(num_batches, 1)
            epoch_metrics[k] = round(avg_val, 8)
            if metrics_tracker.writer:
                metrics_tracker.writer.add_scalar(f"train/{k}", avg_val, epoch + 1)

        if is_main_process():
            decorr_status = (
                f" active (β={args.decorr_weight})"
                if avg_decorr_t > 1e-8
                else f"⚠ inactive (encoder_level={args.encoder_level}, β={args.decorr_weight})"
            )
            # Per-level epoch summary line
            level_summary_parts = []
            for k in running_decorr_levels:
                avg_val = running_decorr_levels[k] / max(num_batches, 1)
                short_name = k.replace('decorr_', '')
                level_summary_parts.append(f"{short_name}={avg_val:.6f}")
            level_summary = " | ".join(level_summary_parts) if level_summary_parts else "none"

            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}"
                f"  loss={avg_loss_t:.4f}"
                f"  diff={avg_diff_t:.4f}"
                f"  decorr={avg_decorr_t:.6f} {decorr_status}"
                f"  gnorm={avg_grad_norm:.3f}"
                f"  lr={current_lr:.2e}"
                f"  time={format_time(epoch_time)}"
                f"  peak_gpu={peak_mem:.1f}GB"
            )
            logger.info(
                f"  decorr breakdown: [{level_summary}]"
            )

        # ── eval every N epochs ───────────────────────────────────
        run_eval = (
            is_main_process()
            and (
                (epoch + 1) % args.eval_every == 0
                or epoch == args.num_epochs - 1
            )
        )

        if run_eval:
            logger.info(f"\nRunning eval at epoch {epoch+1}...")
            eval_metrics = eval_model.evaluate(
                dataloader  = dataloader,
                adapter     = adapter,
                device      = device,
                n_batches   = args.eval_batches,
                eval_steps  = args.eval_steps,
            )
            eval_metrics.log(logger, epoch + 1)

            # push eval metrics to tensorboard + epoch JSON
            for k, v in eval_metrics.to_dict().items():
                if metrics_tracker.writer:
                    metrics_tracker.writer.add_scalar(k, v, epoch + 1)
            epoch_metrics.update({
                k.replace("eval/", ""): v
                for k, v in eval_metrics.to_dict().items()
            })

            # ADE trend warning across eval cycles
            if not hasattr(eval_model, '_ade_history'):
                eval_model._ade_history = []
            eval_model._ade_history.append(eval_metrics.ade)
            if len(eval_model._ade_history) >= 3:
                recent = eval_model._ade_history[-3:]
                if recent[-1] >= recent[0]:
                    logger.warning(
                        f"  ⚠ ADE not improving over last 3 evals: "
                        f"{recent[0]:.4f} → {recent[1]:.4f} → {recent[2]:.4f}  "
                        f"Check GT coord units and traj_channels padding."
                    )

        # All ranks sync after eval
        if dist.is_initialized():
            dist.barrier()

        # ── metrics + checkpoint ──────────────────────────────────
        metrics_tracker.log_epoch(epoch + 1, epoch_metrics, current_lr, epoch_time)

        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            is_best = avg_loss_t < ckpt_mgr.best_loss
            if is_best:
                ckpt_mgr.best_loss = avg_loss_t
            ckpt_mgr.save(
                model, optimizer, scheduler, scaler,
                epoch + 1, epoch_metrics, is_best,
            )
            if is_main_process():
                logger.info(f"  Checkpoint saved  best={is_best}")

    # ── 9. Finish ─────────────────────────────────────────────────────
    total_time = time.time() - total_start
    metrics_tracker.save_summary(total_time, args)

    logger.info("=" * 70)
    logger.info(f"  TRAINING COMPLETE — {format_time(total_time)}")
    logger.info(f"  Best loss : {ckpt_mgr.best_loss:.4f}")
    logger.info(f"  Outputs   : {output_mgr.run_dir}")
    logger.info("=" * 70)

    metrics_tracker.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()