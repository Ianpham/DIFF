#!/usr/bin/env python3
"""Debug: what's actually in the occ .npy files?"""
import os
import numpy as np
from collections import Counter

OCC_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"

# Load a few different samples
scenes = sorted(os.listdir(OCC_ROOT))[:3]

for scene in scenes:
    occ_dir = os.path.join(OCC_ROOT, scene, "occ_gt")
    if not os.path.isdir(occ_dir):
        continue
    
    # Load occ and flow for same frame
    occ_files = sorted([f for f in os.listdir(occ_dir) if "occ_final" in f])
    flow_files = sorted([f for f in os.listdir(occ_dir) if "flow_final" in f])
    
    if not occ_files:
        continue
    
    occ_path = os.path.join(occ_dir, occ_files[0])
    occ = np.load(occ_path)
    
    print(f"\n{'='*60}")
    print(f"Scene: {scene}")
    print(f"File: {occ_files[0]}")
    print(f"Shape: {occ.shape}, dtype: {occ.dtype}")
    print(f"Col 0 — min: {occ[:,0].min()}, max: {occ[:,0].max()}, unique: {len(np.unique(occ[:,0]))}")
    print(f"Col 1 — min: {occ[:,1].min()}, max: {occ[:,1].max()}, unique: {len(np.unique(occ[:,1]))}")
    
    # First 10 rows
    print(f"\nFirst 10 rows:")
    print(occ[:10])
    
    # Last 10 rows
    print(f"\nLast 10 rows:")
    print(occ[-10:])
    
    # If col 1 has small values (0-16), it's (voxel_idx, class)
    # If col 1 has large values, it's something else
    col1_vals = np.unique(occ[:, 1])
    if col1_vals.max() <= 20:
        print(f"\nCol 1 looks like CLASS LABELS: {col1_vals}")
        # Count per class
        classes, counts = np.unique(occ[:, 1], return_counts=True)
        for c, n in zip(classes, counts):
            print(f"  class {c}: {n} voxels")
    else:
        print(f"\nCol 1 has LARGE values — NOT class labels")
        print(f"  Col 1 unique count: {len(col1_vals)}")
        
        # Maybe it's (x*Y*Z + y*Z + z, class) packed differently
        # Or maybe col0 and col1 are coordinates and class is separate
        # Or maybe the array is actually (N, 2) where both are coords
        # and class info is in a separate file
        
        # Check if values fit into grid coordinates
        grid_200x200x16 = 200 * 200 * 16  # 640000
        grid_512x512x40 = 512 * 512 * 40  # 10485760
        print(f"  200x200x16 grid size: {grid_200x200x16}")
        print(f"  Max col0: {occ[:,0].max()}, fits 200x200x16: {occ[:,0].max() < grid_200x200x16}")
        print(f"  Max col1: {occ[:,1].max()}, fits 200x200x16: {occ[:,1].max() < grid_200x200x16}")
        
        # Maybe (col0, col1) = (flat_index_occupied, flat_index_empty)?
        # Or (voxel_coord_packed, semantic_and_visibility_packed)?
        
        # Try interpreting col0 as flat index, unravel
        sample_idx = occ[0, 0]
        for grid in [(200, 200, 16), (512, 512, 40), (256, 256, 32)]:
            total = grid[0] * grid[1] * grid[2]
            if sample_idx < total:
                z = sample_idx % grid[2]
                y = (sample_idx // grid[2]) % grid[1]
                x = sample_idx // (grid[1] * grid[2])
                print(f"  Index {sample_idx} in {grid} → x={x}, y={y}, z={z}")
    
    # Also check flow file
    if flow_files:
        flow_path = os.path.join(occ_dir, flow_files[0])
        flow = np.load(flow_path)
        print(f"\nFlow file: {flow_files[0]}")
        print(f"  Shape: {flow.shape}, dtype: {flow.dtype}")
        print(f"  Col 0 — min: {flow[:,0].min():.3f}, max: {flow[:,0].max():.3f}")
        print(f"  Col 1 — min: {flow[:,1].min():.3f}, max: {flow[:,1].max():.3f}")
        print(f"  First 5 rows: {flow[:5]}")
        
        # Flow and occ should have same number of rows
        print(f"  Occ rows: {occ.shape[0]}, Flow rows: {flow.shape[0]}")
        print(f"  Same length: {occ.shape[0] == flow.shape[0]}")

    # Check if there are other files in occ_gt
    all_files = sorted(os.listdir(occ_dir))
    print(f"\nAll files in {scene}/occ_gt/: {all_files[:10]}")
    
    # Check file sizes
    for f in all_files[:4]:
        fp = os.path.join(occ_dir, f)
        sz = os.path.getsize(fp)
        print(f"  {f}: {sz} bytes")