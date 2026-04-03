# ============================================================
# GaussianFormer3D × TransDiffuser — Configuration
# Scaled for 1× NVIDIA A100-SXM4-40GB
# SuperPod: vinfast_tamp@sdc2-hpc-login-mgmt001
# ============================================================

# --- Scene geometry (FULL resolution) ---
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]  # full 640K voxels — A100 can handle it
num_classes = 17
empty_label = 0

# --- Sensor (full resolution) ---
num_cameras = 8
input_img_size = (448, 800)  # full resolution

# --- Voxelization (fine-grained) ---
lidar_voxel_size = [0.075, 0.075, 0.2]  # original fine voxels

# --- Gaussian representation (full capacity) ---
num_gaussians = 25600  # full
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

# --- Training phases (scaled for 1× A100 40GB) ---
# Rule of thumb: ~5× more batch than 7.5GB GPU
# A100 has TF32 by default → great throughput
phase1 = dict(
    epochs=24,
    batch_size=8,          # OCC training — heavy on VRAM due to 640K voxel grid
    lr=2e-4,
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0),
)

phase2 = dict(
    epochs=120,
    batch_size=64,         # Diffusion only — lightweight, can push batch high
    lr=1e-4,
    weight_decay=0.01,
    scheduler="onecycle",
    grad_clip=10.0,
    loss_weights=dict(diffusion=1.0, decorrelation=0.02),
)

phase3 = dict(
    epochs=60,
    batch_size=8,          # Joint fine-tune — similar VRAM to phase1
    lr=2e-5,
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02),
)

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
# Enable in your train.py or launcher:
#   torch.backends.cuda.matmul.allow_tf32 = True
#   torch.backends.cudnn.allow_tf32 = True
#   torch.set_float32_matmul_precision('high')
#
# Optional: use torch.compile() for extra speed on A100
#   model = torch.compile(model)
#
# Mixed precision (AMP) is highly recommended:
#   scaler = torch.amp.GradScaler('cuda')
#   with torch.amp.autocast('cuda', dtype=torch.float16):
#       loss = model(batch)