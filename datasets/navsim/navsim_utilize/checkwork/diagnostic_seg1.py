#!/usr/bin/env python3
"""
Diagnostic: Examine cached semantic BEV files in detail
Check what's actually in the cache and identify the problem
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print("DIAGNOSTIC: Examining Cached Semantic BEV Files")
print("="*80)

data_root = Path("/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download")
bev_cache_dir = data_root / 'cache' / 'transdiffuser_mini_phase1_corrected' / 'bevfusion_semantic'

print(f"\nBEV cache directory: {bev_cache_dir}")
print(f"Exists: {bev_cache_dir.exists()}")

# Find all semantic files
semantic_files = list(bev_cache_dir.glob('*_semantic.pt'))
print(f"\nTotal cached files: {len(semantic_files)}")

if len(semantic_files) == 0:
    print("No files found!")
    exit(1)

# Examine multiple samples to see pattern
print("\n" + "="*80)
print("Examining 10 Random Samples")
print("="*80)

import random
sample_files = random.sample(semantic_files, min(10, len(semantic_files)))

channel_names = [
    'Ch0-Drivable',
    'Ch1-PedCross',
    'Ch2-Walkway',
    'Ch3-StopLine',
    'Ch4-Carpark',
    'Ch5-Divider'
]

all_stats = {f'Ch{i}': {'coverage': [], 'min': [], 'max': [], 'mean': []} for i in range(6)}

for idx, file in enumerate(sample_files):
    print(f"\n{idx+1}. {file.stem}")
    print("   " + "-"*70)
    
    try:
        data = torch.load(file).numpy()
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        
        print(f"   {'Channel':<15} {'Non-Zero':<12} {'Min':<10} {'Max':<10} {'Mean':<10}")
        print("   " + "-"*70)
        
        for i in range(6):
            nz = (data[i] > 0).sum()
            total = data[i].size
            cov = nz / total * 100
            min_val = data[i].min()
            max_val = data[i].max()
            mean_val = data[i].mean()
            
            all_stats[f'Ch{i}']['coverage'].append(cov)
            all_stats[f'Ch{i}']['min'].append(min_val)
            all_stats[f'Ch{i}']['max'].append(max_val)
            all_stats[f'Ch{i}']['mean'].append(mean_val)
            
            print(f"   {channel_names[i]:<15} {nz:7d}/{total:7d} {min_val:9.3f} {max_val:9.3f} {mean_val:9.3f}")
            
    except Exception as e:
        print(f"     Error: {e}")

# Print summary
print("\n" + "="*80)
print("Summary Across All 10 Samples")
print("="*80)

print(f"\n{'Channel':<15} {'Avg Coverage':<15} {'Min Coverage':<15} {'Max Coverage':<15}")
print("-"*80)

for i in range(6):
    key = f'Ch{i}'
    avg_cov = np.mean(all_stats[key]['coverage'])
    min_cov = np.min(all_stats[key]['coverage'])
    max_cov = np.max(all_stats[key]['coverage'])
    
    print(f"{channel_names[i]:<15} {avg_cov:13.1f}% {min_cov:13.1f}% {max_cov:13.1f}%")

# Detailed look at one file
print("\n" + "="*80)
print("Detailed Analysis of First File")
print("="*80)

first_file = sample_files[0]
print(f"\nFile: {first_file.stem}")

data = torch.load(first_file).numpy()

print("\nValue distribution (percentiles):")
print(f"{'Channel':<15} {'0%':<10} {'25%':<10} {'50%':<10} {'75%':<100} {'100%':<10}")
print("-"*80)

for i in range(6):
    p0 = np.percentile(data[i], 0)
    p25 = np.percentile(data[i], 25)
    p50 = np.percentile(data[i], 50)
    p75 = np.percentile(data[i], 75)
    p100 = np.percentile(data[i], 100)
    
    print(f"{channel_names[i]:<15} {p0:<9.3f} {p25:<9.3f} {p50:<9.3f} {p75:<9.3f} {p100:<9.3f}")

# Check if values are mostly in specific ranges
print("\n" + "="*80)
print("Value Range Analysis")
print("="*80)

print("\nPercentage of pixels in different value ranges:")
print(f"{'Channel':<15} {'=0.0':<10} {'(0-0.1)':<10} {'(0.1-0.2)':<10} {'(0.2-1.0)':<10} {'>1.0':<10}")
print("-"*80)

for i in range(6):
    zero = (data[i] == 0).sum() / data[i].size * 100
    small = ((data[i] > 0) & (data[i] <= 0.1)).sum() / data[i].size * 100
    medium = ((data[i] > 0.1) & (data[i] <= 0.2)).sum() / data[i].size * 100
    large = ((data[i] > 0.2) & (data[i] <= 1.0)).sum() / data[i].size * 100
    huge = (data[i] > 1.0).sum() / data[i].size * 100
    
    print(f"{channel_names[i]:<15} {zero:9.1f}% {small:9.1f}% {medium:9.1f}% {large:9.1f}% {huge:9.1f}%")

# Visualize all 6 channels side by side
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    im = ax.imshow(data[i], cmap='viridis', vmin=0, vmax=data[i].max())
    
    nz = (data[i] > 0).sum()
    total = data[i].size
    cov = nz / total * 100
    
    ax.set_title(f"{channel_names[i]}\nCoverage: {cov:.1f}% | Max: {data[i].max():.3f}", 
                 fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle(f"Semantic BEV Channels\n{first_file.stem}\nAll 6 Channels with Coverage %", 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('./semantic_bev_diagnostic.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n  Saved to: {output_path}")

try:
    plt.show()
except:
    print("(Display not available)")

# Check what's wrong
print("\n" + "="*80)
print("Analysis & Diagnosis")
print("="*80)

print("""
OBSERVATIONS:
  1. Drivable Area (Ch0) = 0.0% - This is the MAIN problem!
     → The neural model is not detecting drivable area
     → OR the heuristic is completely replacing it
     → OR the blending is wrong

  2. Walkway (Ch2) and Stop Line (Ch3) have high coverage
     → These are from HEURISTIC detection (not neural)
     → Suggests heuristics are too aggressive

  3. Expected behavior:
     → Ch0 (Drivable) should be 30-45% (from LiDAR)
     → Ch1 (Ped Crossing) should be 5-15% (rare)
     → Others should be 5-20% each

POSSIBLE CAUSES:
  A) Heuristic detection is TOO STRONG (overwriting neural)
     → Heuristic output on wrong channels
     → Blend ratio wrong (0.5/0.4 might be incorrect)

  B) Neural model output is wrong
     → Not detecting drivable area
     → Channel order mismatch
     → Model weights not loading

  C) Channel assignment is wrong
     → Semantic classes order doesn't match code
     → BEVFusion checkpoint has different class order

NEXT STEPS:
  1. Check the heuristic_detector output directly
  2. Check neural model output before blending
  3. Verify channel ordering matches BEVFusion
  4. Consider disabling heuristics to test pure neural
""")

print("="*80)