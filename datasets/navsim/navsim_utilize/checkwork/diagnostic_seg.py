"""
Diagnostic script to identify segmentation failures in Phase 1
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os

print("\n" + "="*70)
print("NAVSIM Phase 1 - Segmentation Diagnostic")
print("="*70)

# Check 1: GPU/Device availability
print("\n1. DEVICE CONFIGURATION")
print("-" * 70)
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ⚠ Running on CPU (very slow)")

# Check 2: Dataset paths
print("\n2. DATASET PATHS")
print("-" * 70)
data_root = Path(os.environ.get('OPENSCENE_DATA_ROOT', './data'))
print(f"  Data root: {data_root}")
print(f"  Exists: {data_root.exists()}")

sensor_blob_path = data_root / 'mini_sensor_blobs' / 'mini'
print(f"  Sensor blobs: {sensor_blob_path}")
print(f"  Exists: {sensor_blob_path.exists()}")

if sensor_blob_path.exists():
    date_folders = list(sensor_blob_path.glob('*/'))
    print(f"  Date folders: {len(date_folders)}")
    
    if date_folders:
        sample_date = date_folders[0]
        print(f"\n  Sample date folder: {sample_date.name}")
        
        # Check cameras
        cam_folder = sample_date / 'CAM_F0'
        if cam_folder.exists():
            cam_files = list(cam_folder.glob('*.jpg')) + list(cam_folder.glob('*.png'))
            print(f"    CAM_F0 images: {len(cam_files)}")
        else:
            print(f"    CAM_F0: NOT FOUND")
        
        # Check LiDAR
        lidar_folder = sample_date / 'MergedPointCloud'
        if lidar_folder.exists():
            lidar_files = list(lidar_folder.glob('*.pcd'))
            print(f"    MergedPointCloud: {len(lidar_files)} PCD files")
        else:
            print(f"    MergedPointCloud: NOT FOUND")

# Check 3: Checkpoint
print("\n3. BEVFUSION CHECKPOINT")
print("-" * 70)
checkpoint_path = "./checkpoints/bevfusion-seg.pth"
print(f"  Path: {checkpoint_path}")
print(f"  Exists: {Path(checkpoint_path).exists()}")

# Check 4: Test model forward pass
print("\n4. MODEL FORWARD PASS TEST")
print("-" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple test model
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        
        # Simplified BEVFusion-like architecture
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))  # Downsample to 8x8
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(64, num_classes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, camera, lidar):
        cam_feat = self.camera_encoder(camera)
        
        # Resize LiDAR to match camera features
        lidar_resized = torch.nn.functional.interpolate(
            lidar, size=cam_feat.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        lidar_feat = self.lidar_encoder(lidar_resized)
        
        # Fuse
        fused = torch.cat([cam_feat, lidar_feat], dim=1)
        fused_feat = self.fusion(fused)
        
        # Segment + Upsample to BEV size
        seg = self.head(fused_feat)
        seg = torch.nn.functional.interpolate(
            seg, size=(64, 64),
            mode='bilinear', align_corners=False
        )
        return seg

model = SimpleSegmentationModel(num_classes=6)
model = model.to(device)
model.eval()

# Test inference
try:
    with torch.no_grad():
        # Create dummy inputs
        camera = torch.randn(1, 3, 224, 224).to(device)
        lidar = torch.randn(1, 2, 64, 64).to(device)
        
        output = model(camera, lidar)
        
        print(f"  ✓ Forward pass successful")
        print(f"    Camera input: {camera.shape}")
        print(f"    LiDAR input: {lidar.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output min: {output.min():.4f}")
        print(f"    Output max: {output.max():.4f}")
        print(f"    Output mean: {output.mean():.4f}")
        
        # Check if all zeros
        if output.abs().sum() < 1e-6:
            print(f"    ⚠ WARNING: Output is all zeros! Model not learning.")
        else:
            print(f"    ✓ Output has non-zero values")
            
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")

# Check 5: Data loading issues
print("\n5. DATA LOADING DIAGNOSTICS")
print("-" * 70)

try:
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    
    print("  ✓ NAVSIM libraries imported successfully")
    
    logs_path = data_root / 'mini_navsim_logs' / 'mini'
    sensor_path = data_root / 'mini_sensor_blobs' / 'mini'
    
    if logs_path.exists() and sensor_path.exists():
        sensor_config = SensorConfig(
            cam_f0=True,
            lidar_pc=True
        )
        
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=4,
            num_future_frames=8,
        )
        
        loader = SceneLoader(
            data_path=logs_path,
            original_sensor_path=sensor_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        print(f"  ✓ SceneLoader initialized")
        print(f"  ✓ Total scenes: {len(loader.tokens)}")
        
        if len(loader.tokens) > 0:
            # Try loading first scene
            token = loader.tokens[0]
            scene = loader.get_scene_from_token(token)
            frame = scene.frames[0]
            
            print(f"\n  Sample frame analysis:")
            print(f"    Frame token: {frame.token}")
            
            if hasattr(frame, 'camera'):
                print(f"    Camera available: {frame.camera is not None}")
            
            if hasattr(frame, 'lidar'):
                print(f"    LiDAR available: {frame.lidar is not None}")
                if frame.lidar is not None:
                    pc_shape = frame.lidar.lidar_pc.shape
                    print(f"    LiDAR shape: {pc_shape}")
            
            if hasattr(frame, 'ego_status'):
                print(f"    Ego status available: {frame.ego_status is not None}")
    else:
        print(f"  ⚠ Required paths not found")
        print(f"    Logs: {logs_path.exists()}")
        print(f"    Sensor: {sensor_path.exists()}")

except ImportError as e:
    print(f"  ✗ Cannot import NAVSIM: {e}")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")

# Check 6: Common issues
print("\n6. COMMON ISSUES & FIXES")
print("-" * 70)
print("""
Issue: Semantic coverage all 0.0
Possible causes:

1. ✗ BEVFusion checkpoint not loaded properly
   → Fallback model being used
   → Fallback model hasn't learned anything
   FIX: Train fallback model or provide real checkpoint

2. ✗ Camera image preprocessing issues
   → Wrong normalization
   → Wrong tensor shape
   → Image too dark/bright
   FIX: Verify normalization matches training

3. ✗ LiDAR rasterization broken
   → No points in grid
   → Wrong coordinate system
   → Points filtered out
   FIX: Check rasterization bounds

4. ✗ Model output layer mismatch
   → Sigmoid squashing all outputs to near 0
   → Batch normalization issues
   FIX: Check model architecture

5. ✗ No training signal
   → Using untrained model
   → Using model trained on different dataset
   FIX: Use pre-trained BEVFusion weights

RECOMMENDED FIXES:
- Option A: Use pre-trained BEVFusion checkpoint
- Option B: Switch to simpler detection (drivable area only)
- Option C: Use pretrained semantic segmentation backbone
- Option D: Add training loop to learn from your data
""")

print("\n" + "="*70)
print("Diagnostic Complete")
print("="*70 + "\n")