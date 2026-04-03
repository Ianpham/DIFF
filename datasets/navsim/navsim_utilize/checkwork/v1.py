#!/usr/bin/env python3
"""
Single Image Visualization - All Semantic BEV Channels with Statistics
Creates one comprehensive image for easy examination
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

print("="*80)
print("Creating Single Comprehensive Visualization")
print("="*80)

data_root = Path("/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download")
bev_cache_dir = data_root / 'cache' / 'transdiffuser_mini_phase1_corrected' / 'bevfusion_semantic'

# Load a sample
semantic_files = list(bev_cache_dir.glob('*_semantic.pt'))
sample_file = semantic_files[0]

print(f"\nLoading: {sample_file.stem}")
data = torch.load(sample_file).numpy()

print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Channel names
channel_names = [
    'Ch0: Drivable Area',
    'Ch1: Ped Crossing',
    'Ch2: Walkway',
    'Ch3: Stop Line',
    'Ch4: Carpark',
    'Ch5: Divider'
]

# Create single comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

print("\nCreating 6 channel visualizations...")

# Top row and middle: Individual channels in 2x3 grid
for i in range(6):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    
    channel_data = data[i]
    
    # Calculate statistics
    nz = (channel_data > 0).sum()
    total = channel_data.size
    coverage = nz / total * 100
    min_val = channel_data.min()
    max_val = channel_data.max()
    mean_val = channel_data.mean()
    std_val = channel_data.std()
    
    # Create heatmap
    im = ax.imshow(channel_data, cmap='viridis', vmin=0, vmax=max(channel_data.max(), 0.1))
    
    # Title with statistics
    title = f"{channel_names[i]}\n"
    title += f"Coverage: {coverage:.1f}% | Mean: {mean_val:.3f} ± {std_val:.3f}\n"
    title += f"Range: [{min_val:.3f}, {max_val:.3f}]"
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

# Bottom row: Summary statistics
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.axis('off')

# Create statistics table
stats_text = "SEMANTIC BEV STATISTICS\n\n"
stats_text += f"{'Channel':<20} {'Coverage %':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12}\n"
stats_text += "-" * 95 + "\n"

total_coverage = np.zeros_like(data[0])
for i in range(6):
    nz = (data[i] > 0).sum()
    total = data[i].size
    coverage = nz / total * 100
    min_val = data[i].min()
    max_val = data[i].max()
    mean_val = data[i].mean()
    std_val = data[i].std()
    
    total_coverage = np.maximum(total_coverage, data[i])
    
    stats_text += f"{channel_names[i]:<20} {coverage:13.1f}% {min_val:11.4f} {max_val:11.4f} {mean_val:11.4f} {std_val:11.4f}\n"

overall_coverage = (total_coverage > 0).sum() / total_coverage.size * 100
stats_text += "-" * 95 + "\n"
stats_text += f"{'OVERALL COVERAGE':<20} {overall_coverage:13.1f}%\n"

# Add interpretation
stats_text += "\n" + "="*95 + "\n"
stats_text += "INTERPRETATION:\n\n"

if data[0].max() == 0:
    stats_text += "⚠ WARNING: Drivable Area (Ch0) has NO detection! This is abnormal.\n"
    stats_text += "  → Expected: 30-45% coverage from LiDAR density\n"
    stats_text += "  → Actual: 0.0%\n\n"

if overall_coverage < 10:
    stats_text += "⚠ CRITICAL: Total coverage is very low (<10%)!\n"
    stats_text += "  → Dataset might have detection failure\n\n"
elif overall_coverage > 80:
    stats_text += "  Good: Overall semantic coverage is high (>80%)\n"
    stats_text += "  → Heuristic detection is working well\n\n"

if coverage < 1 and data[0].max() > 0:
    stats_text += "  Note: Multiple channels have valid non-zero values\n\n"

# Add recommendations
stats_text += "RECOMMENDATIONS:\n"
stats_text += "1. Check if drivable area detection is being overwritten\n"
stats_text += "2. Verify LiDAR loading is working (check LiDAR BEV separately)\n"
stats_text += "3. Consider reducing heuristic blend ratio (currently 0.5 neural + 0.5 heuristic for Ch0)\n"
stats_text += "4. Run 'diagnostic_semantic_bev_detailed.py' for deeper analysis\n"

ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Main title
fig.suptitle(f'Phase 1 Semantic BEV Analysis - {sample_file.stem}\nAll 6 Channels with Coverage Statistics',
            fontsize=16, fontweight='bold', y=0.98)

# Save
output_path = Path('./semantic_bev_single_image.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n  Saved to: {output_path}")
print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

try:
    plt.show()
except:
    print("(Display not available, image saved)")

# Print same statistics to console
print("\n" + "="*80)
print("CONSOLE OUTPUT")
print("="*80)
print(stats_text)