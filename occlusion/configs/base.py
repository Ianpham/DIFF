
# ###########phase init

# point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
# occ_size = [200, 200, 16]
# num_classes = 17
# empty_label = 0
# num_cameras = 8
# input_img_size = (448, 800)
# lidar_voxel_size = [0.075, 0.075, 0.2]
# num_gaussians = 25600
# embed_dims = 128
# semantic_dim = 17
# gaussian_prop_dims = 28
# num_encoder_blocks = 6
# num_gaussian_tokens = 256
# pooling_method = "fps"
# decoder_embed_dim = 256
# decoder_num_heads = 8
# decoder_num_layers = 6
# decoder_ff_dim = 1024
# trajectory_length = 8
# action_dim = 2
# num_diffusion_steps = 10
# num_trajectory_candidates = 30
# decorrelation_beta = 0.02

# phase1 = dict(epochs=24, batch_size=8, lr=2e-4, weight_decay=0.01, scheduler="cosine", grad_clip=35.0,
#     loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0))
# phase2 = dict(epochs=120, batch_size=64, lr=1e-4, weight_decay=0.01, scheduler="onecycle", grad_clip=10.0,
#     loss_weights=dict(diffusion=1.0, decorrelation=0.02))
# phase3 = dict(epochs=60, batch_size=8, lr=2e-5, weight_decay=0.01, scheduler="cosine", grad_clip=35.0,
#     loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02))

# # data_root = "/data/openscene/"
# # occ_label_root = "/data/openscene/occupancy/"
# # navsim_root = "/data/navsim/"
# # work_dir = "./work_dirs/"

# data_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/"
# occ_label_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/"
# navsim_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/"
# work_dir = "./work_dirs/"

# # ============================================================
# # GaussianFormer3D × TransDiffuser — Configuration phase 1
# # Scaled for single GPU (~7.5 GB VRAM)
# # ============================================================

# # --- Scene geometry ---
# point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
# occ_size = [200, 200, 16]
# num_classes = 17
# empty_label = 0

# # --- Sensor ---
# num_cameras = 8
# input_img_size = (448, 800)

# # --- Voxelization (larger voxels to reduce count for low VRAM) ---
# lidar_voxel_size = [0.2, 0.2, 0.4]

# # --- Gaussian representation (reduced for 7.5 GB GPU) ---
# num_gaussians = 4096
# embed_dims = 128
# semantic_dim = 17
# gaussian_prop_dims = 28

# # --- Encoder (reduced) ---
# num_encoder_blocks = 2

# # --- Pooling ---
# num_gaussian_tokens = 64
# pooling_method = "fps"

# # --- Decoder (reduced) ---
# decoder_embed_dim = 128
# decoder_num_heads = 4
# decoder_num_layers = 3
# decoder_ff_dim = 512

# # --- Planning ---
# trajectory_length = 8
# action_dim = 2
# num_diffusion_steps = 10
# num_trajectory_candidates = 30
# decorrelation_beta = 0.02

# # --- Training phases (batch_size=1 for single small GPU) ---
# phase1 = dict(
#     epochs=24,
#     batch_size=1,
#     lr=2e-4,
#     weight_decay=0.01,
#     scheduler="cosine",
#     grad_clip=35.0,
#     loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0),
# )

# phase2 = dict(
#     epochs=120,
#     batch_size=4,
#     lr=1e-4,
#     weight_decay=0.01,
#     scheduler="onecycle",
#     grad_clip=10.0,
#     loss_weights=dict(diffusion=1.0, decorrelation=0.02),
# )

# phase3 = dict(
#     epochs=60,
#     batch_size=1,
#     lr=2e-5,
#     weight_decay=0.01,
#     scheduler="cosine",
#     grad_clip=35.0,
#     loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02),
# )

# # --- Data paths (your machine) ---
# data_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/"
# occ_label_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/"
# navsim_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/"
# maps_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps/"
# work_dir = "./work_dirs/"

# ============================================================
# GaussianFormer3D × TransDiffuser — Configuration
# Aggressively scaled for single GPU (~7.5 GB VRAM) phase 2
# ============================================================

# --- Scene geometry ---
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

# Reduced occ grid: 50x50x8 = 20,000 voxels (was 200x200x16 = 640,000)
occ_size = [50, 50, 8]

num_classes = 17
empty_label = 0

# --- Sensor ---
num_cameras = 8
input_img_size = (224, 400)  # halved from (448, 800)

# --- Voxelization (large voxels) ---
lidar_voxel_size = [0.4, 0.4, 0.8]

# --- Gaussian representation (minimal) ---
num_gaussians = 2048
embed_dims = 64
semantic_dim = 17
gaussian_prop_dims = 28

# --- Encoder (minimal) ---
num_encoder_blocks = 1

# --- Pooling ---
num_gaussian_tokens = 32
pooling_method = "fps"

# --- Decoder (minimal) ---
decoder_embed_dim = 64
decoder_num_heads = 4
decoder_num_layers = 2
decoder_ff_dim = 256

# --- Planning ---
trajectory_length = 8
action_dim = 2
num_diffusion_steps = 10
num_trajectory_candidates = 16
decorrelation_beta = 0.02

# --- Training phases (batch_size=1 everywhere) ---
phase1 = dict(
    epochs=24,
    batch_size=1,
    lr=2e-4,
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0),
)

phase2 = dict(
    epochs=120,
    batch_size=1,
    lr=1e-4,
    weight_decay=0.01,
    scheduler="onecycle",
    grad_clip=10.0,
    loss_weights=dict(diffusion=1.0, decorrelation=0.02),
)

phase3 = dict(
    epochs=60,
    batch_size=1,
    lr=2e-5,
    weight_decay=0.01,
    scheduler="cosine",
    grad_clip=35.0,
    loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02),
)

# Combined pkl generated by build_dataset.py
info_file = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/openscene_infos_train.pkl"

# Sensor blobs (cameras + LiDAR)
sensor_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"

# data_root: used by train.py to construct default info_file path
# Set to same dir as info_file so fallback works
data_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/"

# Occupancy labels (sparse .npy from OpenScene v1.0)
occ_label_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"

# NavSim root (for maps, logs)
navsim_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/"
maps_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps/"

# Work directory
work_dir = "./work_dirs/"