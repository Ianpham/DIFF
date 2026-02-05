"""
Quick NAVSIM Object Attribute Inspector
"""
import os
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
import numpy as np


def inspect_object(obj, name="Object", max_depth=2, current_depth=0):
    """Recursively inspect an object's attributes."""
    indent = "  " * current_depth
    print(f"{indent}{name}:")
    print(f"{indent}  Type: {type(obj).__name__}")
    
    if current_depth >= max_depth:
        return
    
    for attr in sorted(dir(obj)):
        if attr.startswith('_'):
            continue
        
        try:
            value = getattr(obj, attr)
            
            if callable(value):
                continue  # Skip methods
            
            print(f"{indent}  .{attr}:")
            print(f"{indent}    Type: {type(value).__name__}")
            
            if isinstance(value, np.ndarray):
                print(f"{indent}    Shape: {value.shape}")
                print(f"{indent}    Dtype: {value.dtype}")
            elif hasattr(value, 'shape'):
                print(f"{indent}    Shape: {value.shape}")
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                print(f"{indent}    Length: {len(value)}")
                print(f"{indent}    First item type: {type(value[0]).__name__}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"{indent}    Value: {value}")
            elif hasattr(value, '__dict__') and current_depth < max_depth - 1:
                # Nested object
                inspect_object(value, f".{attr}", max_depth, current_depth + 1)
                
        except Exception as e:
            print(f"{indent}  .{attr}: Error - {e}")


print("="*70)
print("NAVSIM DATASET OBJECT INSPECTOR")
print("="*70)

# Setup
data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])

sensor_config = SensorConfig(
    cam_f0=True,
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

print("\nLoading scene loader...")
scene_loader = SceneLoader(
    data_path=data_root / 'mini_navsim_logs' / 'mini',
    original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',
    scene_filter=scene_filter,
    sensor_config=sensor_config
)

# Get first scene
print(f"\nLoaded {len(scene_loader.tokens)} scenes")
print("Inspecting first scene...\n")

token = scene_loader.tokens[0]
scene = scene_loader.get_scene_from_token(token)

print("="*70)
print("1. SCENE OBJECT")
print("="*70)
inspect_object(scene, "Scene", max_depth=1)

print("\n" + "="*70)
print("2. FRAME OBJECT (first history frame)")
print("="*70)
frame = scene.frames[0]
inspect_object(frame, "Frame", max_depth=1)

print("\n" + "="*70)
print("3. EGO STATUS")
print("="*70)
inspect_object(frame.ego_status, "EgoStatus", max_depth=1)

print("\n" + "="*70)
print("4. LIDAR OBJECT")
print("="*70)
if frame.lidar is not None:
    inspect_object(frame.lidar, "Lidar", max_depth=1)
else:
    print("No lidar data in this frame")

print("\n" + "="*70)
print("5. CAMERA OBJECT")
print("="*70)
if frame.cameras is not None:
    inspect_object(frame.cameras, "Camera", max_depth=1)
else:
    print("No camera data in this frame")

print("\n" + "="*70)
print("6. AGENTS (if any)")
print("="*70)
if hasattr(frame, 'agents') and frame.agents is not None:
    print(f"Number of agents: {len(frame.agents) if hasattr(frame.agents, '__len__') else 'N/A'}")
    if hasattr(frame.agents, '__len__') and len(frame.agents) > 0:
        print("\nFirst agent:")
        inspect_object(frame.agents[0], "Agent", max_depth=1)
else:
    print("No agents data")

print("\n" + "="*70)
print("7. FUTURE TRAJECTORY")
print("="*70)
future_traj = scene.get_future_trajectory(num_trajectory_frames=8)
inspect_object(future_traj, "FutureTrajectory", max_depth=1)

print("\n" + "="*70)
print("INSPECTION COMPLETE")
print("="*70)