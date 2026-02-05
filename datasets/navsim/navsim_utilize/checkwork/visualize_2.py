#!/usr/bin/env python3
"""
Quick Phase 1 Visualization - Simple & Fast
Examine results from navsim_dataset_phase1_corrected.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("="*70)
print("Quick Phase 1 Visualization - Corrected Dataset")
print("="*70)

# Setup paths
data_root = Path("/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download")
cache_dir = data_root / 'cache' / 'transdiffuser_mini_phase1_corrected'
bev_cache_dir = cache_dir / 'bevfusion_semantic'

print(f"\nCache directory: {cache_dir}")
print(f"BEV cache directory: {bev_cache_dir}")
print(f"Cache exists: {cache_dir.exists()}")
print(f"BEV cache exists: {bev_cache_dir.exists()}")

# Find cached semantic BEV files
if not bev_cache_dir.exists():
    print("\n✗ BEV cache directory not found!")
    print("Run the corrected dataset initialization first:")
    print("  from navsim_dataset_phase1_corrected import NavsimDatasetPhase1Corrected")
    print("  dataset = NavsimDatasetPhase1Corrected()")
    exit(1)

semantic_files = list(bev_cache_dir.glob('*_semantic.pt'))
print(f"\nFound {len(semantic_files)} cached semantic BEV files")

if len(semantic_files) == 0:
    print("\n✗ No cached semantic BEV files found!")
    print("Run dataset initialization first.")
    exit(1)

# Load first semantic BEV
semantic_file = semantic_files[0]
print(f"\nLoading semantic: {semantic_file.name}")

try:
    semantic_bev = torch.load(semantic_file)
    print(f"✓ Semantic BEV loaded: {semantic_bev.shape}")
except Exception as e:
    print(f"✗ Error loading semantic: {e}")
    exit(1)

# Now try to load from dataset to get full sample
print("\n" + "="*70)
print("Loading Full Sample from Dataset")
print("="*70)

try:
    from bev_liar_img import NavsimDatasetPhase1Corrected
    
    print("Initializing dataset...")
    dataset = NavsimDatasetPhase1Corrected(
        data_root=data_root,
        precompute_bev=False,  # Don't recompute
        use_cache=True,
        device='cpu'
    )
    print(f"✓ Dataset initialized with {len(dataset)} scenes")
    
    # Load first sample
    print(f"\nLoading sample 0...")
    sample = dataset[0]
    print("✓ Sample loaded")
    
except Exception as e:
    print(f"⚠ Could not load full dataset: {e}")
    print("Will use just semantic BEV for visualization")
    sample = None

# Create visualization
print("\n" + "="*70)
print("Analyzing Data")
print("="*70)

if sample is not None:
    # Full sample available
    semantic = sample['context']['semantic'].numpy()  # [6, H, W]
    lidar = sample['context']['lidar'].numpy()        # [2, H, W]
    velocity = sample['context']['velocity'].numpy()  # [2, H, W]
    
    print("\nSample structure:")
    for key in sample.keys():
        if key == 'context':
            print(f"\n{key}:")
            for subkey, value in sample[key].items():
                if isinstance(value, torch.Tensor):
                    print(f"  {subkey:12s}: {list(value.shape)}")
        else:
            if isinstance(sample[key], torch.Tensor):
                print(f"{key:15s}: {list(sample[key].shape)}")
    
else:
    # Only semantic BEV available
    semantic = semantic_bev.numpy()
    print(f"\nSemantic BEV shape: {semantic.shape}")
    
    # Create dummy LiDAR and velocity for visualization
    lidar = np.zeros((2, semantic.shape[1], semantic.shape[2]), dtype=np.float32)
    velocity = np.zeros((2, semantic.shape[1], semantic.shape[2]), dtype=np.float32)
    print("⚠ Using placeholder LiDAR and velocity")

# Print statistics
print("\n" + "="*70)
print("Channel Statistics")
print("="*70)

channel_names = [
    'Drivable Area',
    'Ped Crossing', 
    'Walkway',
    'Stop Line',
    'Carpark',
    'Divider',
    'LiDAR Density',
    'LiDAR Height',
    'Velocity X',
    'Velocity Y'
]

print(f"\n{'Channel':<20s} {'Non-Zero':<12s} {'Min':<8s} {'Max':<8s} {'Mean':<8s} {'Cov %':<8s}")
print("-"*80)

# Semantic channels
for i in range(6):
    nz = (semantic[i] > 0).sum()
    total = semantic[i].size
    coverage = (nz / total * 100) if total > 0 else 0
    min_val = semantic[i].min()
    max_val = semantic[i].max()
    mean_val = semantic[i].mean()
    print(f"{channel_names[i]:<20s} {nz:7d}/{total:7d} {min_val:7.3f} {max_val:7.3f} {mean_val:7.3f} {coverage:6.1f}%")

# LiDAR channels
for i in range(2):
    nz = (lidar[i] > 0).sum()
    total = lidar[i].size
    coverage = (nz / total * 100) if total > 0 else 0
    min_val = lidar[i].min()
    max_val = lidar[i].max()
    mean_val = lidar[i].mean()
    print(f"{channel_names[6+i]:<20s} {nz:7d}/{total:7d} {min_val:7.3f} {max_val:7.3f} {mean_val:7.3f} {coverage:6.1f}%")

# Velocity channels
for i in range(2):
    nz = (np.abs(velocity[i]) > 0.1).sum()
    total = velocity[i].size
    coverage = (nz / total * 100) if total > 0 else 0
    min_val = velocity[i].min()
    max_val = velocity[i].max()
    mean_val = velocity[i].mean()
    print(f"{channel_names[8+i]:<20s} {nz:7d}/{total:7d} {min_val:7.3f} {max_val:7.3f} {mean_val:7.3f} {coverage:6.1f}%")

# Calculate total semantic coverage
print("\n" + "="*70)
print("Summary Statistics")
print("="*70)

total_semantic = np.zeros_like(semantic[0])
for i in range(6):
    total_semantic = np.maximum(total_semantic, semantic[i])

semantic_coverage = (total_semantic > 0).sum() / total_semantic.size * 100
print(f"\nTotal Semantic Coverage: {semantic_coverage:.1f}%")
print(f"Individual Channel Coverage:")
for i in range(6):
    cov = (semantic[i] > 0).sum() / semantic[i].size * 100
    print(f"  {channel_names[i]:<20s}: {cov:6.1f}%")

# Create visualization
print("\n" + "="*70)
print("Creating Visualization")
print("="*70)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Row 1: Semantic channels (0-3)
for i in range(4):
    ax = axes[0, i]
    im = ax.imshow(semantic[i], cmap='viridis', vmin=0, vmax=1)
    cov = (semantic[i] > 0).sum() / semantic[i].size * 100
    ax.set_title(f"{channel_names[i]}\n({cov:.1f}%)", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Row 2: Semantic channels (4-5) + LiDAR
for i in range(2):
    ax = axes[1, i]
    im = ax.imshow(semantic[4+i], cmap='viridis', vmin=0, vmax=1)
    cov = (semantic[4+i] > 0).sum() / semantic[4+i].size * 100
    ax.set_title(f"{channel_names[4+i]}\n({cov:.1f}%)", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

for i in range(2):
    ax = axes[1, 2+i]
    im = ax.imshow(lidar[i], cmap='hot', vmin=0, vmax=1)
    cov = (lidar[i] > 0).sum() / lidar[i].size * 100
    ax.set_title(f"{channel_names[6+i]}\n({cov:.1f}%)", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Row 3: Velocity + Combined views
for i in range(2):
    ax = axes[2, i]
    vmax = max(abs(velocity[i].min()), abs(velocity[i].max()), 0.1)
    im = ax.imshow(velocity[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f"{channel_names[8+i]}\n(range: [{velocity[i].min():.2f}, {velocity[i].max():.2f}])", 
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Combined RGB view
ax = axes[2, 2]
rgb = np.zeros((semantic.shape[1], semantic.shape[2], 3))
rgb[:, :, 0] = np.clip(semantic[0], 0, 1)  # Drivable -> Red
rgb[:, :, 1] = np.clip(lidar[0], 0, 1)     # LiDAR Density -> Green
rgb[:, :, 2] = np.clip(semantic[1], 0, 1)  # Ped Crossing -> Blue
ax.imshow(rgb)
ax.set_title('RGB: Drivable/LiDAR/Crossing', fontsize=10, fontweight='bold')
ax.axis('off')

# Velocity magnitude
ax = axes[2, 3]
vel_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)
im = ax.imshow(vel_mag, cmap='plasma', vmin=0, vmax=15)
ax.set_title(f'Velocity Mag\nmax: {vel_mag.max():.2f} m/s', fontsize=10, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)

title_str = f'Phase 1 BEV - All 10 Channels\nTotal Semantic Coverage: {semantic_coverage:.1f}%'
if sample is not None:
    title_str += f'\nFull Dataset Sample'
else:
    title_str += f'\nSemantic BEV Only'

plt.suptitle(title_str, fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = Path('./phase1_quick_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved to: {output_path}")

# Also show
try:
    plt.show()
except:
    print("(Display not available, figure saved)")

print("\n" + "="*70)
print("✓ Visualization complete!")
print("="*70)
print("\nKey observations:")
print(f"  ✓ Semantic coverage: {semantic_coverage:.1f}% (should be >30%)")
print(f"  ✓ Drivable area: {(semantic[0]>0).sum()/semantic[0].size*100:.1f}% (main channel)")
print(f"  ✓ LiDAR density: {(lidar[0]>0).sum()/lidar[0].size*100:.1f}%")
print(f"  ✓ Velocity range: [{velocity[0].min():.2f}, {velocity[0].max():.2f}] m/s")
print("\nIf semantic coverage < 10%, there might be a detection issue.")
print("If all channels are zero, check cache directory.")
print("="*70)