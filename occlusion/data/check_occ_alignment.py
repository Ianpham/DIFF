#!/usr/bin/env python3
"""
Verify occ voxels align spatially with 3D bounding boxes.
If axis order is wrong, they won't match.
"""
import os
import pickle
import numpy as np

# Paths
PKL = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/openscene_infos_train.pkl"
OCC_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"
PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
OCC_SIZE = [200, 200, 16]

with open(PKL, "rb") as f:
    infos = pickle.load(f)

# Take first frame with occ
info = infos[0]
occ_path = info.get("occ_gt_path", "")
print(f"Token: {info['token']}")
print(f"Scene: {info['scene_name']}")
print(f"Occ path: {occ_path}")

# Load sparse occ
sparse = np.load(occ_path)
print(f"Sparse shape: {sparse.shape}")
print(f"First 5: {sparse[:5]}")

# Get 3D box centers
anns = info.get("anns", {})
gt_boxes = np.array(anns.get("gt_boxes", []))
gt_names = anns.get("gt_names", [])
if len(gt_boxes) > 0:
    print(f"\n3D Boxes: {len(gt_boxes)}")
    print(f"Box center ranges: x=[{gt_boxes[:,0].min():.1f}, {gt_boxes[:,0].max():.1f}], "
          f"y=[{gt_boxes[:,1].min():.1f}, {gt_boxes[:,1].max():.1f}], "
          f"z=[{gt_boxes[:,2].min():.1f}, {gt_boxes[:,2].max():.1f}]")

# Unravel occ indices and convert to world coords
# Test BOTH axis orderings
indices = sparse[:, 0].astype(np.int64)
labels = sparse[:, 1].astype(np.int64)

# Filter to non-empty (labels > 0, exclude class 0 which might be unknown)
mask = labels > 0
indices = indices[mask]
labels = labels[mask]
print(f"\nOcc voxels (non-empty): {len(indices)}")

X, Y, Z = OCC_SIZE
voxel_size = (PC_RANGE[3:] - PC_RANGE[:3]) / np.array(OCC_SIZE)
print(f"Voxel size: {voxel_size}")

# Option A: unravel as (x, y, z) — C-order with shape (X, Y, Z)
# flat_index = x * (Y * Z) + y * Z + z
xA = indices // (Y * Z)
yA = (indices % (Y * Z)) // Z
zA = indices % Z

world_xA = PC_RANGE[0] + (xA + 0.5) * voxel_size[0]
world_yA = PC_RANGE[1] + (yA + 0.5) * voxel_size[1]
world_zA = PC_RANGE[2] + (zA + 0.5) * voxel_size[2]

print(f"\nOption A (X-major, C-order):")
print(f"  x range: [{world_xA.min():.1f}, {world_xA.max():.1f}]")
print(f"  y range: [{world_yA.min():.1f}, {world_yA.max():.1f}]")
print(f"  z range: [{world_zA.min():.1f}, {world_zA.max():.1f}]")

# Option B: unravel as (z, y, x) — some datasets use Z-major
# flat_index = z * (Y * X) + y * X + x
zB = indices // (Y * X)
yB = (indices % (Y * X)) // X
xB = indices % X

world_xB = PC_RANGE[0] + (xB + 0.5) * voxel_size[0]
world_yB = PC_RANGE[1] + (yB + 0.5) * voxel_size[1]
world_zB = PC_RANGE[2] + (zB + 0.5) * voxel_size[2]

print(f"\nOption B (Z-major):")
print(f"  x range: [{world_xB.min():.1f}, {world_xB.max():.1f}]")
print(f"  y range: [{world_yB.min():.1f}, {world_yB.max():.1f}]")
print(f"  z range: [{world_zB.min():.1f}, {world_zB.max():.1f}]")

# Option C: Fortran order (column-major)
# flat_index = x + y * X + z * X * Y
xC = indices % X
yC = (indices // X) % Y
zC = indices // (X * Y)

world_xC = PC_RANGE[0] + (xC + 0.5) * voxel_size[0]
world_yC = PC_RANGE[1] + (yC + 0.5) * voxel_size[1]
world_zC = PC_RANGE[2] + (zC + 0.5) * voxel_size[2]

print(f"\nOption C (Fortran/column-major, x-minor):")
print(f"  x range: [{world_xC.min():.1f}, {world_xC.max():.1f}]")
print(f"  y range: [{world_yC.min():.1f}, {world_yC.max():.1f}]")
print(f"  z range: [{world_zC.min():.1f}, {world_zC.max():.1f}]")

# Now compare: which option's x,y,z ranges best match the 3D box centers?
if len(gt_boxes) > 0:
    # Only compare trucks (class 10) since they dominate
    truck_mask = labels == 10
    
    # Get truck box centers
    truck_box_mask = np.array([n == "truck" for n in gt_names])
    if truck_box_mask.sum() == 0:
        truck_box_mask = np.array([n in ("truck", "car", "bus") for n in gt_names])
    
    if truck_box_mask.sum() > 0:
        truck_boxes = gt_boxes[truck_box_mask]
        print(f"\nTruck box centers ({truck_box_mask.sum()} boxes):")
        print(f"  x: [{truck_boxes[:,0].min():.1f}, {truck_boxes[:,0].max():.1f}]")
        print(f"  y: [{truck_boxes[:,1].min():.1f}, {truck_boxes[:,1].max():.1f}]")
        print(f"  z: [{truck_boxes[:,2].min():.1f}, {truck_boxes[:,2].max():.1f}]")
        
        print(f"\nTruck occ voxels ({truck_mask.sum()} voxels):")
        for name, wx, wy, wz in [
            ("Option A", world_xA[truck_mask], world_yA[truck_mask], world_zA[truck_mask]),
            ("Option B", world_xB[truck_mask], world_yB[truck_mask], world_zB[truck_mask]),
            ("Option C", world_xC[truck_mask], world_yC[truck_mask], world_zC[truck_mask]),
        ]:
            print(f"  {name}: x=[{wx.min():.1f},{wx.max():.1f}] "
                  f"y=[{wy.min():.1f},{wy.max():.1f}] "
                  f"z=[{wz.min():.1f},{wz.max():.1f}]")
        
        print(f"\n>>> The option whose x,y,z ranges best match the truck box centers is CORRECT")