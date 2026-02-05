"""
Quick test to check if scene annotations are loaded
"""

import os
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

# Set paths
data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
map_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
os.environ['NUPLAN_MAPS_ROOT'] = map_root

print("Testing Scene Annotations Loading")
print("=" * 70)

# Your current config
sensor_config = SensorConfig(
    cam_f0=False,
    cam_l0=False, cam_l1=False, cam_l2=False,
    cam_r0=False, cam_r1=False, cam_r2=False,
    cam_b0=False,
    lidar_pc=True
)

scene_filter = SceneFilter(
    log_names=None,
    num_history_frames=4,
    num_future_frames=8,
)

scene_loader = SceneLoader(
    data_path=data_root / 'mini_navsim_logs' / 'mini',
    original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',
    scene_filter=scene_filter,
    sensor_config=sensor_config
)

print(f"✓ SceneLoader created with {len(scene_loader.tokens)} scenes\n")

# Load first scene
token = scene_loader.tokens[0]
scene = scene_loader.get_scene_from_token(token)

print(f"Scene Token: {token}")
print(f"Scene Log: {scene.scene_metadata.log_name}")
print(f"Scene Map: {scene.scene_metadata.map_name}")
print(f"Number of frames: {len(scene.frames)}")
print()

# Check frame annotations
frame_idx = 3  # history_length - 1
frame = scene.frames[frame_idx]

print(f"Frame {frame_idx} analysis:")
print(f"  Ego pose: {frame.ego_status.ego_pose}")
print(f"  Annotations object: {frame.annotations}")

if frame.annotations is None:
    print("  ❌ ANNOTATIONS ARE NONE!")
    print("\nThis is the problem! Annotations are not being loaded.")
    print("\nSolutions:")
    print("  1. Check if the pickle files contain annotations")
    print("  2. Verify data paths are correct")
    print("  3. Ensure you're using the right data split (mini vs trainval)")
else:
    print(f"  ✓ Annotations exist")
    print(f"  Number of boxes: {len(frame.annotations.boxes)}")
    
    if len(frame.annotations.boxes) > 0:
        print(f"  Object names: {frame.annotations.names[:5]}...")
        print(f"  First box: {frame.annotations.boxes[0]}")
        print(f"  Box shape: {frame.annotations.boxes[0].shape}")
        
        # Check if velocity_3d exists
        if hasattr(frame.annotations, 'velocity_3d') and frame.annotations.velocity_3d is not None:
            print(f"  ✓ velocity_3d exists: {frame.annotations.velocity_3d.shape}")
        else:
            print(f"  ❌ velocity_3d is missing!")
    else:
        print(f"  ⚠️  Annotations exist but boxes array is empty!")

# Test the extractor directly
print("\n" + "=" * 70)
print("Testing BEV Label Extractor Directly")
print("=" * 70)

from bev_label_extractor_fixed import BEVLabelExtractor

extractor = BEVLabelExtractor(
    bev_size=(200, 200),
    bev_range=50.0,
    map_root=map_root
)

try:
    labels = extractor.extract_all_labels(scene, frame_idx)
    print("✓ Label extraction succeeded!")
    print("\nLabel statistics:")
    for key, val in labels.items():
        import numpy as np
        nonzero = np.count_nonzero(val)
        print(f"  {key:25s}: non-zero={nonzero:6d}, dtype={val.dtype}")
except Exception as e:
    print(f"❌ Label extraction failed with error:")
    print(f"   {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)