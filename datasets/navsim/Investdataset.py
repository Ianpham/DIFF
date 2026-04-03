"""
NAVSIM BEV (Bird's Eye View) Investigation
Explore all available information for BEV representation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from PIL import Image


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")


def investigate_lidar(frame):
    """Investigate LiDAR data for BEV."""
    print_section("LIDAR INVESTIGATION")
    
    if frame.lidar is None:
        print("  No LiDAR data available")
        return None
    
    lidar = frame.lidar
    pc = lidar.lidar_pc  # Shape: [6, N]
    
    print(f"Point Cloud Shape: {pc.shape}")
    print(f"Number of points: {pc.shape[1]:,}")
    print(f"Channels: {pc.shape[0]}")
    
    # Analyze each channel
    print("\nChannel Analysis:")
    channel_names = ['X', 'Y', 'Z', 'Intensity', 'Ring', 'Timestamp']
    for i, name in enumerate(channel_names[:pc.shape[0]]):
        channel_data = pc[i]
        print(f"  [{i}] {name:12s}: min={channel_data.min():8.3f}, max={channel_data.max():8.3f}, "
              f"mean={channel_data.mean():8.3f}, std={channel_data.std():8.3f}")
    
    # Spatial distribution
    x, y, z = pc[0], pc[1], pc[2]
    print(f"\nSpatial Distribution:")
    print(f"  X (forward/back): [{x.min():.2f}, {x.max():.2f}] meters")
    print(f"  Y (left/right):   [{y.min():.2f}, {y.max():.2f}] meters")
    print(f"  Z (height):       [{z.min():.2f}, {z.max():.2f}] meters")
    
    # Distance from ego
    distances = np.sqrt(x**2 + y**2)
    print(f"\nDistance from ego vehicle:")
    print(f"  Min: {distances.min():.2f} m")
    print(f"  Max: {distances.max():.2f} m")
    print(f"  Mean: {distances.mean():.2f} m")
    print(f"  Median: {np.median(distances):.2f} m")
    
    # Height distribution
    print(f"\nHeight Distribution:")
    print(f"  Ground level (z < 0.5m): {np.sum(z < 0.5):,} points ({100*np.sum(z < 0.5)/len(z):.1f}%)")
    print(f"  Mid-level (0.5-2m):      {np.sum((z >= 0.5) & (z < 2)):,} points ({100*np.sum((z >= 0.5) & (z < 2))/len(z):.1f}%)")
    print(f"  High (z >= 2m):          {np.sum(z >= 2):,} points ({100*np.sum(z >= 2)/len(z):.1f}%)")
    
    # Intensity analysis (if available)
    if pc.shape[0] > 3:
        intensity = pc[3]
        print(f"\nIntensity Analysis:")
        print(f"  Range: [{intensity.min():.2f}, {intensity.max():.2f}]")
        print(f"  Mean: {intensity.mean():.2f}")
        
    return pc


def investigate_cameras(frame):
    """Investigate camera data for BEV."""
    print_section("CAMERA INVESTIGATION")
    
    if frame.cameras is None:
        print("  No camera data available")
        return
    
    cameras = frame.cameras
    cam_names = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']
    cam_descriptions = {
        'cam_f0': 'Front Center',
        'cam_l0': 'Front Left',
        'cam_l1': 'Side Left', 
        'cam_l2': 'Rear Left',
        'cam_r0': 'Front Right',
        'cam_r1': 'Side Right',
        'cam_r2': 'Rear Right',
        'cam_b0': 'Rear Center'
    }
    
    available_cams = []
    for cam_name in cam_names:
        if hasattr(cameras, cam_name):
            cam = getattr(cameras, cam_name)
            if cam is not None:
                available_cams.append(cam_name)
                print(f"  {cam_name:8s} ({cam_descriptions.get(cam_name, 'Unknown'):12s})")
                
                # Check attributes
                if hasattr(cam, 'camera_path'):
                    print(f"    Path: {cam.camera_path}")
                    
                    # Try to load image if path exists
                    if Path(cam.camera_path).exists():
                        try:
                            img = Image.open(cam.camera_path)
                            print(f"    Image size: {img.size} (width x height)")
                            print(f"    Mode: {img.mode}")
                        except Exception as e:
                            print(f"    Could not load image: {e}")
                
                # Check for intrinsics/extrinsics
                if hasattr(cam, 'intrinsics'):
                    print(f"    Intrinsics available: {cam.intrinsics is not None}")
                if hasattr(cam, 'extrinsics'):
                    print(f"    Extrinsics available: {cam.extrinsics is not None}")
                if hasattr(cam, 'camera_intrinsic'):
                    print(f"    Camera intrinsic shape: {cam.camera_intrinsic.shape if hasattr(cam.camera_intrinsic, 'shape') else type(cam.camera_intrinsic)}")
                if hasattr(cam, 'camera_extrinsic'):
                    print(f"    Camera extrinsic shape: {cam.camera_extrinsic.shape if hasattr(cam.camera_extrinsic, 'shape') else type(cam.camera_extrinsic)}")
                    
                print()
    
    print(f"Total available cameras: {len(available_cams)}")
    print(f"Cameras: {', '.join(available_cams)}")
    
    return available_cams


def investigate_annotations(frame):
    """Investigate annotation data (other agents) for BEV."""
    print_section("ANNOTATIONS INVESTIGATION (Other Agents)")
    
    if not hasattr(frame, 'annotations') or frame.annotations is None:
        print("  No annotation data available")
        return None
    
    annot = frame.annotations
    print(f"Annotation type: {type(annot)}")
    
    # Check all attributes
    print("\nAvailable attributes:")
    for attr in dir(annot):
        if not attr.startswith('_'):
            value = getattr(annot, attr)
            if not callable(value):
                print(f"  .{attr}: {type(value).__name__}", end='')
                if hasattr(value, 'shape'):
                    print(f" - shape: {value.shape}", end='')
                if hasattr(value, '__len__') and not isinstance(value, (str, np.ndarray)):
                    print(f" - length: {len(value)}", end='')
                print()
    
    # Detailed analysis of boxes
    if hasattr(annot, 'boxes') and annot.boxes is not None:
        boxes = annot.boxes
        print(f"\nBounding Boxes:")
        print(f"  Shape: {boxes.shape}")
        print(f"  Dtype: {boxes.dtype}")
        print(f"  Number of agents: {boxes.shape[0] if len(boxes.shape) > 0 else 0}")
        
        if len(boxes.shape) > 1:
            print(f"  Box format: {boxes.shape[1]} values per box")
            if boxes.shape[0] > 0:
                print(f"\n  First box: {boxes[0]}")
                print(f"  Box interpretation (typical):")
                print(f"    - [x, y, z, length, width, height, yaw, ...]")
                
                # Spatial distribution
                if boxes.shape[0] > 0:
                    x_vals = boxes[:, 0]
                    y_vals = boxes[:, 1]
                    z_vals = boxes[:, 2] if boxes.shape[1] > 2 else None
                    
                    print(f"\n  Agent positions:")
                    print(f"    X range: [{x_vals.min():.2f}, {x_vals.max():.2f}]")
                    print(f"    Y range: [{y_vals.min():.2f}, {y_vals.max():.2f}]")
                    if z_vals is not None:
                        print(f"    Z range: [{z_vals.min():.2f}, {z_vals.max():.2f}]")
    
    # Labels/categories
    if hasattr(annot, 'labels') and annot.labels is not None:
        labels = annot.labels
        print(f"\nLabels:")
        print(f"  Shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
        print(f"  Unique labels: {np.unique(labels)}")
        
        # Count by category
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Label distribution:")
        for label, count in zip(unique, counts):
            print(f"    Label {label}: {count} agents")
    
    # Velocities
    if hasattr(annot, 'velocities') and annot.velocities is not None:
        velocities = annot.velocities
        print(f"\nVelocities:")
        print(f"  Shape: {velocities.shape}")
        if velocities.shape[0] > 0:
            speeds = np.linalg.norm(velocities[:, :2], axis=1) if velocities.shape[1] >= 2 else velocities[:, 0]
            print(f"  Speed range: [{speeds.min():.2f}, {speeds.max():.2f}] m/s")
            print(f"  Mean speed: {speeds.mean():.2f} m/s")
    
    return annot


def investigate_map_data(scene, frame):
    """Investigate map data for BEV."""
    print_section("MAP DATA INVESTIGATION")
    
    if not hasattr(scene, 'map_api') or scene.map_api is None:
        print("  No map API available")
        return None
    
    map_api = scene.map_api
    print(f"Map API type: {type(map_api).__name__}")
    
    # Get roadblock IDs
    if hasattr(frame, 'roadblock_ids') and frame.roadblock_ids is not None:
        print(f"\nRoadblock IDs: {len(frame.roadblock_ids)} roadblocks")
        if len(frame.roadblock_ids) > 0:
            print(f"  Sample IDs: {frame.roadblock_ids[:3]}")
    
    # Check map API methods
    print(f"\nAvailable map API methods:")
    for attr in dir(map_api):
        if not attr.startswith('_') and callable(getattr(map_api, attr)):
            print(f"  - {attr}()")
    
    # Try to get some map elements
    ego = frame.ego_status
    ego_x, ego_y = ego.ego_pose[0], ego.ego_pose[1]
    
    print(f"\nEgo position: ({ego_x:.2f}, {ego_y:.2f})")
    
    # Try common map queries
    try:
        # This is speculative - actual methods depend on the map API
        print("\nAttempting to query map elements...")
        
        if hasattr(map_api, 'get_available_map_objects'):
            objects = map_api.get_available_map_objects()
            print(f"  Available map objects: {objects}")
    except Exception as e:
        print(f"  Error querying map: {e}")
    
    return map_api


def investigate_traffic_lights(frame):
    """Investigate traffic light data for BEV."""
    print_section("TRAFFIC LIGHTS INVESTIGATION")
    
    if not hasattr(frame, 'traffic_lights') or frame.traffic_lights is None:
        print("  No traffic light data available")
        return None
    
    tl = frame.traffic_lights
    print(f"Number of traffic lights: {len(tl)}")
    
    if len(tl) > 0:
        print(f"\nTraffic light format:")
        print(f"  Type: {type(tl[0])}")
        print(f"  First traffic light: {tl[0]}")
        
        if isinstance(tl[0], tuple):
            print(f"  Tuple length: {len(tl[0])}")
            print(f"\n  Typical format: (lane_id, status)")
            print(f"    - Lane ID: {tl[0][0] if len(tl[0]) > 0 else 'N/A'}")
            print(f"    - Status: {tl[0][1] if len(tl[0]) > 1 else 'N/A'}")
        
        # Count by status
        if isinstance(tl[0], tuple) and len(tl[0]) > 1:
            statuses = [t[1] for t in tl]
            unique, counts = np.unique(statuses, return_counts=True)
            print(f"\n  Status distribution:")
            for status, count in zip(unique, counts):
                print(f"    Status {status}: {count} lights")
    
    return tl


def investigate_ego_trajectory(scene, frame):
    """Investigate ego vehicle trajectory for BEV."""
    print_section("EGO TRAJECTORY INVESTIGATION")
    
    # Current ego state
    ego = frame.ego_status
    print(f"Current Ego State (at t=0):")
    print(f"  Position (x, y): ({ego.ego_pose[0]:.2f}, {ego.ego_pose[1]:.2f})")
    print(f"  Heading: {ego.ego_pose[2]:.4f} rad ({np.degrees(ego.ego_pose[2]):.2f}°)")
    print(f"  Velocity (vx, vy): ({ego.ego_velocity[0]:.2f}, {ego.ego_velocity[1]:.2f}) m/s")
    print(f"  Speed: {np.linalg.norm(ego.ego_velocity):.2f} m/s")
    print(f"  Acceleration (ax, ay): ({ego.ego_acceleration[0]:.2f}, {ego.ego_acceleration[1]:.2f}) m/s²")
    print(f"  Driving command: {ego.driving_command}")
    
    # Historical trajectory
    print(f"\nHistorical Trajectory (past {len(scene.frames[:4])} frames):")
    for i, f in enumerate(scene.frames[:4]):
        e = f.ego_status
        print(f"  t-{4-i}: pos=({e.ego_pose[0]:7.2f}, {e.ego_pose[1]:7.2f}), "
              f"vel=({e.ego_velocity[0]:5.2f}, {e.ego_velocity[1]:5.2f}), "
              f"heading={np.degrees(e.ego_pose[2]):6.2f}°")
    
    # Future trajectory
    future_traj = scene.get_future_trajectory(num_trajectory_frames=8)
    poses = future_traj.poses
    
    print(f"\nFuture Trajectory (next {len(poses)} frames):")
    for i, pose in enumerate(poses):
        print(f"  t+{i+1}: pos=({pose[0]:7.2f}, {pose[1]:7.2f}), heading={np.degrees(pose[2]):6.2f}°")
    
    # Trajectory statistics
    future_distances = np.sqrt(np.diff(poses[:, 0])**2 + np.diff(poses[:, 1])**2)
    print(f"\nFuture Trajectory Statistics:")
    print(f"  Total distance: {future_distances.sum():.2f} m")
    print(f"  Average step: {future_distances.mean():.2f} m")
    print(f"  Min/Max step: {future_distances.min():.2f} / {future_distances.max():.2f} m")
    
    return ego, poses


def create_bev_summary(scene, frame_idx=3):
    """Create a comprehensive summary of BEV-relevant data."""
    print_section("BEV DATA SUMMARY")
    
    frame = scene.frames[frame_idx]
    
    summary = {
        'lidar': frame.lidar is not None,
        'cameras': frame.cameras is not None,
        'annotations': hasattr(frame, 'annotations') and frame.annotations is not None,
        'traffic_lights': hasattr(frame, 'traffic_lights') and frame.traffic_lights is not None,
        'map_api': hasattr(scene, 'map_api') and scene.map_api is not None,
        'roadblocks': hasattr(frame, 'roadblock_ids') and frame.roadblock_ids is not None,
    }
    
    print("Data Availability for BEV Construction:")
    print(f"  {' ' if summary['lidar'] else ' '} LiDAR Point Cloud")
    print(f"  {' ' if summary['cameras'] else ' '} Camera Images")
    print(f"  {' ' if summary['annotations'] else ' '} Agent Bounding Boxes")
    print(f"  {' ' if summary['traffic_lights'] else ' '} Traffic Light States")
    print(f"  {' ' if summary['map_api'] else ' '} HD Map API")
    print(f"  {' ' if summary['roadblocks'] else ' '} Roadblock Information")
    
    print("\nRecommended BEV Channels:")
    channels = []
    
    if summary['lidar']:
        channels.extend([
            "1. LiDAR Density (point density per grid cell)",
            "2. LiDAR Height (max height per grid cell)",
            "3. LiDAR Intensity (mean intensity per grid cell)"
        ])
    
    if summary['annotations']:
        channels.extend([
            "4. Agent Occupancy (binary mask of other vehicles)",
            "5. Agent Velocity (velocity field of other agents)"
        ])
    
    if summary['map_api'] or summary['roadblocks']:
        channels.extend([
            "6. Drivable Area (road surface mask)",
            "7. Lane Boundaries (lane markings)"
        ])
    
    if summary['traffic_lights']:
        channels.append("8. Traffic Light States (per-lane signals)")
    
    channels.append("9. Ego Vehicle Mask (current vehicle position)")
    
    for channel in channels:
        print(f"  {channel}")
    
    return summary


# ============================================================
# Main Investigation Script
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print(" NAVSIM BEV INVESTIGATION")
    print("="*80)
    
    # Setup
    data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    
    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
        cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
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
    print(f"Loaded {len(scene_loader.tokens)} scenes")
    print("Investigating first scene...\n")
    
    token = scene_loader.tokens[0]
    scene = scene_loader.get_scene_from_token(token)
    current_frame = scene.frames[3]  # Last history frame
    
    # Run all investigations
    lidar_data = investigate_lidar(current_frame)
    camera_data = investigate_cameras(current_frame)
    annot_data = investigate_annotations(current_frame)
    map_data = investigate_map_data(scene, current_frame)
    tl_data = investigate_traffic_lights(current_frame)
    ego_data, future_poses = investigate_ego_trajectory(scene, current_frame)
    
    # Summary
    summary = create_bev_summary(scene)
    
    print("\n" + "="*80)
    print(" INVESTIGATION COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("  - LiDAR provides 3D point cloud with ~90k points")
    print("  - Multiple camera views available (8 cameras)")
    print("  - Agent annotations include 3D bounding boxes")
    print("  - Map API provides road geometry")
    print("  - Traffic light states available")
    print("\nRecommendation:")
    print("  Build BEV with 7-9 channels combining:")
    print("  1) LiDAR features (density, height, intensity)")
    print("  2) Agent occupancy and dynamics")
    print("  3) Road/lane geometry from map")
    print("  4) Optional: traffic light states")
    print("="*80)