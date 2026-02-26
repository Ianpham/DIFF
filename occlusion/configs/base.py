point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]
num_classes = 17
empty_label = 0
num_cameras = 8
input_img_size = (448, 800)
lidar_voxel_size = [0.075, 0.075, 0.2]
num_gaussians = 25600
embed_dims = 128
semantic_dim = 17
gaussian_prop_dims = 28
num_encoder_blocks = 6
num_gaussian_tokens = 256
pooling_method = "fps"
decoder_embed_dim = 256
decoder_num_heads = 8
decoder_num_layers = 6
decoder_ff_dim = 1024
trajectory_length = 8
action_dim = 2
num_diffusion_steps = 10
num_trajectory_candidates = 30
decorrelation_beta = 0.02

phase1 = dict(epochs=24, batch_size=8, lr=2e-4, weight_decay=0.01, scheduler="cosine", grad_clip=35.0,
    loss_weights=dict(occ_ce=1.0, occ_lovasz=1.0))
phase2 = dict(epochs=120, batch_size=64, lr=1e-4, weight_decay=0.01, scheduler="onecycle", grad_clip=10.0,
    loss_weights=dict(diffusion=1.0, decorrelation=0.02))
phase3 = dict(epochs=60, batch_size=8, lr=2e-5, weight_decay=0.01, scheduler="cosine", grad_clip=35.0,
    loss_weights=dict(occ_ce=0.5, occ_lovasz=0.5, diffusion=1.0, decorrelation=0.02))

data_root = "/data/openscene/"
occ_label_root = "/data/openscene/occupancy/"
navsim_root = "/data/navsim/"
work_dir = "./work_dirs/"
