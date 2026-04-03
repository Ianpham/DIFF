# ============================================================
# GaussianFormer3D × TransDiffuser — Configuration
# Scaled for 4× NVIDIA A100-SXM4-40GB (DDP / multi-GPU)
# SuperPod: vinfast_tamp@sdc2-hpc-login-mgmt001
# ============================================================
#
# LAUNCH COMMAND (torchrun):
#   torchrun --nproc_per_node=4 train.py --config config_4xA100.py
#
# LAUNCH COMMAND (SLURM):
#   srun --partition=vinai --nodes=1 --ntasks-per-node=4 \
#        --gpus-per-task=1 --cpus-per-task=8 --mem=256G \
#        --time=10-00:00:00 \
#        python -m torch.distributed.launch --nproc_per_node=4 \
#        train.py --config config_4xA100.py
#
# EFFECTIVE BATCH SIZE = batch_size_per_gpu × num_gpus
# e.g., phase1: 8 × 4 = 32 effective batch size
# ============================================================

# --- Multi-GPU settings ---
num_gpus = 4
dist_backend = "nccl"           # best for A100 NVLink
find_unused_parameters = False  # set True only if needed

# --- Scene geometry (FULL resolution) ---
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]  # full 640K voxels
num_classes = 17
empty_label = 0

# --- Sensor (full resolution) ---
num_cameras = 8
input_img_size = (448, 800)

# --- Voxelization (fine-grained) ---
lidar_voxel_size = [0.075, 0.075, 0.2]

# --- Gaussian representation (full capacity) ---
num_gaussians = 25600
embed_dims = 128
semantic_dim = 17
gaussian_prop_dims = 28

# --- Encoder (full) ---
num_encoder_blocks = 6

# --- Pooling ---
num_gaussian_tokens = 256
pooling_method = "fps"

# --- Decoder (full) ---
decoder_embed_dim = 256
decoder_num_heads = 8
decoder_num_layers = 6
decoder_ff_dim = 1024

# --- Planning ---
trajectory_length = 8
action_dim = 2
num_diffusion_steps = 10
num_trajectory_candidates = 30
decorrelation_beta = 0.02

# --- Training phases (PER-GPU batch sizes) ---
# Effective batch = batch_size × 4 GPUs
# LR scaled with linear scaling rule: lr_new = lr_base × (eff_batch / base_batch)
# Warmup recommended when using large effective batch

phase1 = dict(
    epochs=24,
    batch_size=8,          # per GPU → effective 32
    lr=4e-4,               # scaled: 2e-4 × (32/8) = 8e-4, but cap at 4e-4 for stability
    warmup_epochs=2,       # warmup for large batch
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0),
)

phase2 = dict(
    epochs=120,
    batch_size=64,         # per GPU → effective 256
    lr=2e-4,               # scaled: 1e-4 × 2 (conservative for diffusion)
    warmup_epochs=3,
    weight_decay=0.01,
    scheduler="onecycle",
    grad_clip=10.0,
    loss_weights=dict(diffusion=1.0, decorrelation=0.02),
)

phase3 = dict(
    epochs=60,
    batch_size=8,          # per GPU → effective 32
    lr=4e-5,               # scaled: 2e-5 × 2 (conservative for fine-tune)
    warmup_epochs=1,
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02),
)

# --- DataLoader settings for multi-GPU ---
num_workers = 8            # per GPU — 8 CPUs/GPU is good for A100 DGX
pin_memory = True
persistent_workers = True
prefetch_factor = 2

# --- Data paths (SuperPod) ---
_base_path = "/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser/DDPM"

info_file = f"{_base_path}/datasets/navsim/download/openscene_infos_train.pkl"
sensor_root = f"{_base_path}/datasets/navsim/download/mini_sensor_blobs/mini"
data_root = f"{_base_path}/datasets/navsim/download/"
occ_label_root = f"{_base_path}/datasets/occ_mini/openscene-v1.0/occupancy/mini"
navsim_root = f"{_base_path}/datasets/navsim/"
maps_root = f"{_base_path}/datasets/navsim/download/maps/"
work_dir = "./outputs/"

# --- Performance settings for A100 ---
# Add to your train.py:
#
#   import torch
#   torch.backends.cuda.matmul.allow_tf32 = True
#   torch.backends.cudnn.allow_tf32 = True
#   torch.set_float32_matmul_precision('high')
#
# Mixed precision (highly recommended):
#   scaler = torch.amp.GradScaler('cuda')
#   with torch.amp.autocast('cuda', dtype=torch.float16):
#       loss = model(batch)
#
# DDP init in train.py:
#   torch.distributed.init_process_group(backend="nccl")
#   local_rank = int(os.environ["LOCAL_RANK"])
#   torch.cuda.set_device(local_rank)
#   model = model.to(local_rank)
#   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
#
# DistributedSampler:
#   from torch.utils.data.distributed import DistributedSampler
#   train_sampler = DistributedSampler(train_dataset, shuffle=True)
#   train_loader = DataLoader(dataset, batch_size=cfg.batch_size,
#                             sampler=train_sampler, num_workers=cfg.num_workers,
#                             pin_memory=True, persistent_workers=True)
#   # Remember: train_sampler.set_epoch(epoch) in training loop!