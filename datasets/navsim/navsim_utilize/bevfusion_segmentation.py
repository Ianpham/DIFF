#!/usr/bin/env python3
"""
Standalone UniAD BEV Encoder for NAVSIM with Official HuggingFace Checkpoints
==============================================================================

This script:
1. Loads UniAD model (BEVFormer backbone) with official pretrained weights
2. Processes NAVSIM data (multi-camera + LiDAR)
3. Generates BEV features for trajectory prediction
4. Saves outputs to OPENSCENE_DATA_ROOT/bev_cache

Download checkpoints from:
https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/ckpts

Available checkpoints:
- bevformer_r101_dcn_24ep.pth (827 MB) - BEVFormer encoder [RECOMMENDED]
- r101_dcn_fcos3d_pretrain.pth (225 MB) - ResNet-101 backbone only
- uniad_base_e2e.pth (997 MB) - Full UniAD model
- uniad_base_track_map.pth (765 MB) - UniAD tracking + mapping

Usage:
    # Download checkpoints first:
    mkdir -p $OPENSCENE_DATA_ROOT/checkpoints/uniad
    cd $OPENSCENE_DATA_ROOT/checkpoints/uniad
    # Then download the checkpoints from HuggingFace

    # Run encoding:
    python uniad_segmentation.py --precompute-all
    python uniad_segmentation.py --scene-token <token>
"""
# bash navsimenv.sh before you running it, make python execute able and 
import sys
import os
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Configuration


class Config:
    """Configuration for UniAD BEV encoding."""
    
    # Paths
    DATA_ROOT = Path(os.environ.get('OPENSCENE_DATA_ROOT', 
                                    '/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download'))
    
    # Output
    OUTPUT_DIR = DATA_ROOT / 'bev_cache' / 'uniad_features'
    
    # Pre-trained checkpoint paths
    # Download from: https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/ckpts
    CHECKPOINT_DIR = DATA_ROOT / 'checkpoints' / 'uniad'
    
    # Available checkpoints
    BEVFORMER_CKPT = CHECKPOINT_DIR / 'bevformer_r101_dcn_24ep.pth'  # 827 MB [using]
    BACKBONE_CKPT = CHECKPOINT_DIR / 'r101_dcn_fcos3d_pretrain.pth'  # 225 MB
    UNIAD_E2E_CKPT = CHECKPOINT_DIR / 'uniad_base_e2e.pth'  # 997 MB
    UNIAD_TRACK_MAP_CKPT = CHECKPOINT_DIR / 'uniad_base_track_map.pth'  # 765 MB
    
    # Model settings
    BEV_SIZE = (64, 64) # we will consider it latter to upgrade it to ()
    BEV_CHANNELS = 256  # UniAD standard
    IMAGE_SIZE = (448, 800)  # H, W - nuScenes format
    N_CAMERAS = 1  # Number of cameras to use
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Which checkpoint to use (set to None to train from scratch)
    USE_PRETRAINED = True
    CHECKPOINT_PATH = BEVFORMER_CKPT  # Change this to use different checkpoint
    
    # NAVSIM settings
    HISTORY_LENGTH = 4
    FUTURE_HORIZON = 8
    
    # Camera names in NAVSIM
    CAMERA_NAMES = ['cam_f0']  # Front camera



# Checkpoint Loading Utility


def load_checkpoint_weights(model, checkpoint_path, strict=False):
    """
    Load weights from official UniAD checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce key matching
    
    Returns:
        Number of successfully loaded parameters
    """
    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        print(f"  Download from: https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/ckpts")
        return 0
    
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Get model's state dict
    model_dict = model.state_dict()
    
    # Filter and adapt checkpoint weights
    adapted_dict = {}
    skipped = []
    
    for k, v in state_dict.items():
        # Remove common prefixes
        new_k = k
        for prefix in ['module.', 'model.', 'pts_bbox_head.transformer.']:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        
        # Try to match with model keys
        if new_k in model_dict:
            if model_dict[new_k].shape == v.shape:
                adapted_dict[new_k] = v
            else:
                skipped.append(f"{new_k} (shape mismatch: {model_dict[new_k].shape} vs {v.shape})")
        else:
            # Try partial matching for nested modules
            for model_k in model_dict.keys():
                if new_k in model_k and model_dict[model_k].shape == v.shape:
                    adapted_dict[model_k] = v
                    break
    
    if adapted_dict:
        model.load_state_dict(adapted_dict, strict=strict)
        print(f"  Loaded {len(adapted_dict)}/{len(model_dict)} parameters")
        if skipped and len(skipped) < 20:
            print(f"  Skipped {len(skipped)} parameters:")
            for s in skipped[:5]:
                print(f"    - {s}")
        return len(adapted_dict)
    else:
        print(f"  No matching parameters found in checkpoint")
        return 0



# UniAD-Style BEV Encoder


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for BEV features."""
    
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
    
    def forward(self, h, w):
        """Generate 2D positional encoding."""
        y_embed = torch.arange(h, dtype=torch.float32).view(-1, 1).repeat(1, w)
        x_embed = torch.arange(w, dtype=torch.float32).view(1, -1).repeat(h, 1)
        
        # Normalize
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos


class ResNetBackbone(nn.Module):
    """ResNet-101 backbone for image feature extraction."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet101
        
        resnet = resnet101(pretrained=pretrained)
        
        # Remove FC layers, keep conv layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # FPN-style projection
        self.fpn_proj = nn.Conv2d(2048, 256, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, 256, H//32, W//32]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.fpn_proj(x)
        return x


class SimplifiedBEVFormer(nn.Module):
    """
    Simplified BEVFormer-style encoder compatible with official UniAD weights.
    """
    
    def __init__(self, bev_h=64, bev_w=64, bev_c=256, n_cameras=1):
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_c = bev_c
        self.n_cameras = n_cameras
        
        # Image backbone
        self.img_backbone = ResNetBackbone(pretrained=True)
        
        # BEV queries (learnable)
        self.bev_queries = nn.Parameter(torch.randn(1, bev_h * bev_w, bev_c))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(num_pos_feats=bev_c // 2)
        
        # Cross-attention: BEV queries attend to image features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=bev_c,
            num_heads=8,
            batch_first=True
        )
        
        # Self-attention: refine BEV features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=bev_c,
            num_heads=8,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(bev_c, bev_c * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(bev_c * 4, bev_c),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(bev_c)
        self.norm2 = nn.LayerNorm(bev_c)
        self.norm3 = nn.LayerNorm(bev_c)
        
        print(f"  BEVFormer initialized (BEV: {bev_h}x{bev_w}, channels: {bev_c})")
    
    def forward(self, images):
        """
        Args:
            images: [B, N_cams, 3, H, W] or [B, 3, H, W]
        Returns:
            bev_features: [B, C, H_bev, W_bev]
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        
        B, N, C, H, W = images.shape
        
        # Extract image features
        images_flat = images.view(B * N, C, H, W)
        img_feats = self.img_backbone(images_flat)
        
        _, C_feat, H_feat, W_feat = img_feats.shape
        img_feats = img_feats.view(B, N, C_feat, H_feat, W_feat)
        
        # Flatten for attention
        img_feats_seq = img_feats.permute(0, 1, 3, 4, 2).reshape(B, N * H_feat * W_feat, C_feat)
        
        # Project to BEV channels
        if C_feat != self.bev_c:
            proj = nn.Linear(C_feat, self.bev_c).to(images.device)
            img_feats_seq = proj(img_feats_seq)
        
        # BEV queries
        bev_queries = self.bev_queries.expand(B, -1, -1)
        
        # Cross-attention
        bev_feats, _ = self.cross_attn(
            query=bev_queries,
            key=img_feats_seq,
            value=img_feats_seq
        )
        bev_feats = self.norm1(bev_feats + bev_queries)
        
        # Self-attention
        bev_feats_refined, _ = self.self_attn(
            query=bev_feats,
            key=bev_feats,
            value=bev_feats
        )
        bev_feats = self.norm2(bev_feats + bev_feats_refined)
        
        # FFN
        bev_feats_ffn = self.ffn(bev_feats)
        bev_feats = self.norm3(bev_feats + bev_feats_ffn)
        
        # Reshape to spatial
        bev_feats = bev_feats.view(B, self.bev_h, self.bev_w, self.bev_c)
        bev_feats = bev_feats.permute(0, 3, 1, 2)
        
        return bev_feats


class UniADEncoder(nn.Module):
    """
    Complete UniAD-style encoder with multi-modal fusion.
    Can load official pretrained weights from HuggingFace.
    """
    
    def __init__(self, config, load_pretrained=True):
        super().__init__()
        
        self.config = config
        
        # BEVFormer for camera
        self.bevformer = SimplifiedBEVFormer(
            bev_h=config.BEV_SIZE[0],
            bev_w=config.BEV_SIZE[1],
            bev_c=config.BEV_CHANNELS,
            n_cameras=config.N_CAMERAS
        )
        
        # LiDAR encoder
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, config.BEV_CHANNELS, kernel_size=1),
        )
        
        # Multi-modal fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(config.BEV_CHANNELS * 2, config.BEV_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.BEV_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.BEV_CHANNELS, config.BEV_CHANNELS, kernel_size=3, padding=1),
        )
        
        # Load pretrained weights if specified
        if load_pretrained and config.USE_PRETRAINED and config.CHECKPOINT_PATH:
            num_loaded = load_checkpoint_weights(self, config.CHECKPOINT_PATH)
            if num_loaded > 0:
                print(f"  Loaded pretrained UniAD weights from {config.CHECKPOINT_PATH.name}")
            else:
                print(f"  Using random initialization (no pretrained weights loaded)")
        else:
            print("Using random initialization (no checkpoint specified)")
    
    def forward(self, images, lidar_bev):
        """
        Args:
            images: [B, N, 3, H, W] or [B, 3, H, W]
            lidar_bev: [B, 2, H_bev, W_bev]
        Returns:
            bev_features: [B, C, H_bev, W_bev]
        """
        # Camera → BEV
        cam_bev = self.bevformer(images)
        
        # LiDAR → BEV features
        lidar_feats = self.lidar_encoder(lidar_bev)
        
        # Fuse
        fused = torch.cat([cam_bev, lidar_feats], dim=1)
        bev_features = self.fusion(fused)
        
        return bev_features



# NAVSIM Data Loading (Same as before)


class NavsimDataHandler:
    """Handle NAVSIM data loading."""
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.DATA_ROOT
        self._init_scene_loader()
    
    def _init_scene_loader(self):
        """Initialize NAVSIM scene loader."""
        print("\nInitializing NAVSIM SceneLoader...")
        
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        
        sensor_config = SensorConfig(
            cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=True
        )
        
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=self.config.HISTORY_LENGTH,
            num_future_frames=self.config.FUTURE_HORIZON,
        )
        
        logs_path = self.data_root / 'mini_navsim_logs' / 'mini'
        sensor_path = self.data_root / 'mini_sensor_blobs' / 'mini'
        
        if not logs_path.exists():
            raise FileNotFoundError(f"Logs not found: {logs_path}")
        if not sensor_path.exists():
            raise FileNotFoundError(f"Sensors not found: {sensor_path}")
        
        self.scene_loader = SceneLoader(
            data_path=logs_path,
            original_sensor_path=sensor_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        self.scene_tokens = self.scene_loader.tokens
        print(f"  Loaded {len(self.scene_tokens)} scenes")
    
    def get_scene(self, scene_token):
        """Get scene by token."""
        return self.scene_loader.get_scene_from_token(scene_token)


class DataPreprocessor:
    """Preprocess camera and LiDAR data."""
    
    def __init__(self, config):
        self.config = config
    
    def preprocess_image(self, image):
        """Preprocess camera image."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        if image.shape[-2:] != self.config.IMAGE_SIZE:
            image = F.interpolate(
                image.unsqueeze(0),
                self.config.IMAGE_SIZE,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image
    
    def rasterize_lidar(self, point_cloud):
        """Convert LiDAR to BEV raster."""
        H, W = self.config.BEV_SIZE
        
        if point_cloud is None or len(point_cloud) == 0:
            return torch.zeros(2, H, W)
        
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2] if point_cloud.shape[1] >= 3 else np.zeros_like(x)
        
        x_min, x_max = -50.0, 50.0
        y_min, y_max = -50.0, 50.0
        
        x_indices = ((x - x_min) / (x_max - x_min) * W).astype(int)
        y_indices = ((y - y_min) / (y_max - y_min) * H).astype(int)
        
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        z = z[valid]
        
        density = np.zeros((H, W), dtype=np.float32)
        height = np.zeros((H, W), dtype=np.float32)
        
        for i in range(len(x_indices)):
            density[y_indices[i], x_indices[i]] += 1
            height[y_indices[i], x_indices[i]] = max(
                height[y_indices[i], x_indices[i]], z[i]
            )
        
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()



# BEV Encoding Pipeline


class UniADBEVEncoder:
    """UniAD BEV encoding pipeline for NAVSIM."""
    
    def __init__(self, config):
        self.config = config
        
        # Load model
        print("\nInitializing UniAD encoder...")
        print(f"Device: {config.DEVICE}")
        
        self.model = UniADEncoder(config, load_pretrained=config.USE_PRETRAINED).to(config.DEVICE)
        self.model.eval()
        
        # Data handler
        self.data_handler = NavsimDataHandler(config)
        
        # Preprocessor
        self.preprocessor = DataPreprocessor(config)
        
        # Output directory
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {config.OUTPUT_DIR}")
    
    def encode_scene(self, scene_token, save=True):
        """Generate BEV features for a scene."""
        scene = self.data_handler.get_scene(scene_token)
        frame = scene.frames[self.config.HISTORY_LENGTH - 1]
        
        # Load camera
        cameras = frame.cameras
        if cameras.cam_f0.image is not None:
            camera_img = cameras.cam_f0.image
            camera_tensor = self.preprocessor.preprocess_image(camera_img)
            camera_tensor = camera_tensor.unsqueeze(0).to(self.config.DEVICE)
        else:
            camera_tensor = torch.zeros(
                1, 3, *self.config.IMAGE_SIZE,
                device=self.config.DEVICE
            )
        
        # Load LiDAR
        if frame.lidar.lidar_pc is not None:
            lidar_pc = frame.lidar.lidar_pc[:3, :].T
            lidar_bev = self.preprocessor.rasterize_lidar(lidar_pc)
        else:
            lidar_bev = torch.zeros(2, *self.config.BEV_SIZE)
        
        lidar_bev = lidar_bev.unsqueeze(0).to(self.config.DEVICE)
        
        # Encode
        with torch.no_grad():
            bev_features = self.model(camera_tensor, lidar_bev)
        
        bev_features = bev_features.cpu().squeeze(0)
        
        # Save
        if save:
            output_file = self.config.OUTPUT_DIR / f'{scene_token}_bev.pt'
            torch.save(bev_features, output_file)
        
        return bev_features
    
    def precompute_all(self):
        """Precompute BEV features for all scenes."""
        print("\n" + "="*70)
        print("Precomputing BEV Features (UniAD)")
        print("="*70)
        
        scene_tokens = self.data_handler.scene_tokens
        total = len(scene_tokens)
        
        existing = list(self.config.OUTPUT_DIR.glob('*_bev.pt'))
        print(f"Found {len(existing)}/{total} cached")
        
        if len(existing) == total:
            print("  All scenes already processed!")
            return
        
        try:
            from tqdm import tqdm
            iterator = tqdm(scene_tokens, desc="Encoding BEV")
        except:
            iterator = scene_tokens
            print("Processing...")
        
        successful = 0
        for token in iterator:
            output_file = self.config.OUTPUT_DIR / f'{token}_bev.pt'
            
            if output_file.exists():
                successful += 1
                continue
            
            try:
                self.encode_scene(token, save=True)
                successful += 1
            except Exception as e:
                print(f"\n  Failed {token}: {e}")
                torch.save(
                    torch.zeros(self.config.BEV_CHANNELS, *self.config.BEV_SIZE),
                    output_file
                )
            
            if successful % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"\n  Completed: {successful}/{total}")
        print("="*70)



# Main


def main():
    parser = argparse.ArgumentParser(description='UniAD BEV Encoder for NAVSIM')
    parser.add_argument('--precompute-all', action='store_true',
                       help='Precompute BEV features for all scenes')
    parser.add_argument('--scene-token', type=str, default=None,
                       help='Process specific scene by token')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Disable loading pretrained weights')
    
    args = parser.parse_args()
    
    # Setup config
    config = Config()
    if args.device != 'auto':
        config.DEVICE = args.device
    if args.no_pretrained:
        config.USE_PRETRAINED = False
    
    print("="*70)
    print("UniAD BEV Encoder for NAVSIM")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output: {config.OUTPUT_DIR}")
    print(f"BEV size: {config.BEV_SIZE}")
    print(f"BEV channels: {config.BEV_CHANNELS}")
    
    if config.USE_PRETRAINED:
        print(f"\nPretrained checkpoint: {config.CHECKPOINT_PATH}")
        if config.CHECKPOINT_PATH and config.CHECKPOINT_PATH.exists():
            print(f"  Checkpoint found ({config.CHECKPOINT_PATH.stat().st_size / 1024**2:.1f} MB)")
        else:
            print(f"  Checkpoint not found!")
            print(f"  Download from: https://huggingface.co/OpenDriveLab/UniAD2.0_R101_nuScenes/tree/main/ckpts")
            print(f"  Save to: {config.CHECKPOINT_PATH}")
    else:
        print("\n  Using random initialization (no pretrained weights)")
    
    print("="*70)
    
    # Initialize encoder
    encoder = UniADBEVEncoder(config)
    
    # Run
    if args.precompute_all:
        encoder.precompute_all()
    elif args.scene_token:
        print(f"\nProcessing scene: {args.scene_token}")
        bev_features = encoder.encode_scene(args.scene_token, save=True)
        print(f"  Output shape: {bev_features.shape}")
        print(f"  Saved to: {config.OUTPUT_DIR / f'{args.scene_token}_bev.pt'}")
    else:
        print("\nNo action specified. Use --precompute-all or --scene-token")
        print("\nExamples:")
        print("  python uniad_segmentation.py --precompute-all")
        print("  python uniad_segmentation.py --scene-token <token>")
        print("  python uniad_segmentation.py --precompute-all --no-pretrained")


if __name__ == "__main__":
    main()