"""
Configurable NAVSIM Dataset Wrapper for TransDiffuser
Supports different dataset sizes and splits
"""

import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List

# Import NAVSIM official classes
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig


class NavsimDataset(Dataset):
    """
    Configurable NAVSIM wrapper for TransDiffuser.
    Supports different dataset sizes and configurations.
    """
    
    # Available dataset configurations
    DATASET_CONFIGS = {
        'mini': {
            'log_path': 'mini_navsim_logs/mini',
            'sensor_path': 'mini_sensor_blobs/mini',
            'description': 'Mini split (~4k scenes, fast for testing)'
        },
        'trainval': {
            'log_path': 'navsim_logs/trainval',
            'sensor_path': 'sensor_blobs/trainval', 
            'description': 'Full training + validation split'
        },
        'test': {
            'log_path': 'navsim_logs/test',
            'sensor_path': 'sensor_blobs/test',
            'description': 'Test split'
        }
    }
    
    def __init__(
        self,
        split: str = 'mini',
        bev_size: Tuple[int, int] = (64, 64),
        history_length: int = 4,
        future_horizon: int = 8,
        max_scenes: Optional[int] = None,
        log_names: Optional[List[str]] = None,
        use_cache: bool = True,
        sensors: Optional[Dict[str, bool]] = None,
    ):
        """
        Initialize NAVSIM dataset.
        
        Args:
            split: Dataset split to use ('mini', 'trainval', 'test')
            bev_size: (H, W) size of BEV representation
            history_length: Number of historical timesteps (default 4)
            future_horizon: Number of future waypoints (default 8)
            max_scenes: Maximum number of scenes to load (None = all)
            log_names: Specific log names to load (None = all)
            use_cache: Whether to cache preprocessed data
            sensors: Dictionary of sensor configurations (None = default)
        """
        # Validate split
        if split not in self.DATASET_CONFIGS:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.DATASET_CONFIGS.keys())}")
        
        # Get paths from environment
        data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        self.split = split
        self.bev_size = bev_size
        self.history_length = history_length
        self.future_horizon = future_horizon
        self.use_cache = use_cache
        self.max_scenes = max_scenes
        self.cache_dir = data_root / 'cache' / f'transdiffuser_{split}'
        
        print(f"Initializing NAVSIM Dataset ({split} split)...")
        
        # Sensor configuration
        if sensors is None:
            sensors = {
                'cam_f0': True,   # Front camera
                'cam_l0': False,  # Left cameras
                'cam_l1': False,
                'cam_l2': False,
                'cam_r0': False,  # Right cameras
                'cam_r1': False,
                'cam_r2': False,
                'cam_b0': False,  # Back camera
                'lidar_pc': True  # LiDAR point cloud
            }
        
        sensor_config = SensorConfig(**sensors)
        
        # Scene filter
        scene_filter = SceneFilter(
            log_names=log_names,
            num_history_frames=history_length,
            num_future_frames=future_horizon,
        )
        
        # Get paths for this split
        config = self.DATASET_CONFIGS[split]
        log_path = data_root / config['log_path']
        sensor_path = data_root / config['sensor_path']
        
        print(f"Loading from: {log_path}")
        
        # Create scene loader
        self.scene_loader = SceneLoader(
            data_path=log_path,
            original_sensor_path=sensor_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        self.scene_tokens = self.scene_loader.tokens
        
        # Limit scenes if requested
        if max_scenes is not None:
            self.scene_tokens = self.scene_tokens[:max_scenes]
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Loaded {len(self)} scenes from {split} split")
        print(f"  - History frames: {history_length}")
        print(f"  - Future frames: {future_horizon}")
    
    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample."""
        token = self.scene_tokens[idx]
        
        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load scene
        scene: Scene = self.scene_loader.get_scene_from_token(token)
        
        # Process
        sample = self._process_scene(scene)
        
        # Cache
        if self.use_cache:
            cache_file = self.cache_dir / f'{token}.pt'
            torch.save(sample, cache_file)
        
        return sample
    
    def get_raw_scene(self, idx: int) -> Scene:
        """Get raw NAVSIM Scene object for inspection."""
        token = self.scene_tokens[idx]
        return self.scene_loader.get_scene_from_token(token)
    
    def _process_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Convert NAVSIM Scene to TransDiffuser format."""
        
        # Get current frame (last history frame)
        current_frame = scene.frames[self.history_length - 1]
        
        # 1. LiDAR context
        if current_frame.lidar is not None:
            # NAVSIM lidar format: [6, N] where first 3 channels are x, y, z
            point_cloud = current_frame.lidar.lidar_pc[:3, :].T  # Extract x,y,z and transpose to [N, 3]
            lidar_bev = self._rasterize_lidar(point_cloud)
        else:
            lidar_bev = torch.zeros(2, *self.bev_size)
        
        # 2. BEV placeholder (TODO: implement BEV encoder)
        camera_bev = torch.zeros(7, *self.bev_size)
        
        context = {
            'BEV': camera_bev,
            'lidar': lidar_bev,
        }
        
        # 3. Current agent state
        # EgoStatus attributes: ego_pose [3], ego_velocity [2], ego_acceleration [2], driving_command [4]
        ego = current_frame.ego_status
        # ego_pose is [x, y, heading], ego_velocity is [vx, vy]
        agent_states = torch.tensor(
            [[ego.ego_pose[0], ego.ego_pose[1], ego.ego_velocity[0], ego.ego_velocity[1], ego.ego_pose[2]]],
            dtype=torch.float32
        )
        
        # 4. Agent history
        history_states = []
        for i in range(self.history_length):
            ego = scene.frames[i].ego_status
            state = torch.tensor(
                [[ego.ego_pose[0], ego.ego_pose[1], ego.ego_velocity[0], ego.ego_velocity[1], ego.ego_pose[2]]],
                dtype=torch.float32
            )
            history_states.append(state)
        
        agent_history = torch.cat(history_states, dim=0).unsqueeze(0)  # [1, 4, 5]
        
        # 5. Ground truth trajectory
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=self.future_horizon
        )
        poses = future_traj.poses  # [8, 3] with (x, y, heading)
        
        waypoints = []
        current_ego = current_frame.ego_status
        
        for i in range(len(poses)):
            x, y, heading = poses[i]
            
            # Calculate velocities from position differences
            if i > 0:
                prev_x, prev_y, _ = poses[i-1]
                dt = 0.5  # Assuming 0.5s timestep
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            else:
                # Use current ego velocity for first waypoint
                vx = current_ego.ego_velocity[0]
                vy = current_ego.ego_velocity[1]
            
            waypoints.append([x, y, vx, vy, heading])
        
        gt_trajectory = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)  # [1, 8, 5]
        
        return {
            'context': context,
            'agent_states': agent_states,
            'agent_history': agent_history,
            'gt_trajectory': gt_trajectory,
        }
    
    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR to BEV grid."""
        H, W = self.bev_size
        
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2] if point_cloud.shape[1] >= 3 else np.zeros_like(x)
        
        # Grid bounds
        x_min, x_max = -50.0, 50.0
        y_min, y_max = -50.0, 50.0
        
        # Convert to indices
        x_indices = ((x - x_min) / (x_max - x_min) * W).astype(int)
        y_indices = ((y - y_min) / (y_max - y_min) * H).astype(int)
        
        # Filter valid
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        z = z[valid]
        
        # Create grids
        density = np.zeros((H, W), dtype=np.float32)
        height = np.zeros((H, W), dtype=np.float32)
        
        for i in range(len(x_indices)):
            density[y_indices[i], x_indices[i]] += 1
            height[y_indices[i], x_indices[i]] = max(height[y_indices[i], x_indices[i]], z[i])
        
        # Normalize
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()


def collate_fn(batch):
    """Collate function for DataLoader."""
    bev = torch.stack([item['context']['BEV'] for item in batch])
    lidar = torch.stack([item['context']['lidar'] for item in batch])
    
    return {
        'context': {'BEV': bev, 'lidar': lidar},
        'agent_states': torch.cat([item['agent_states'] for item in batch], dim=0),
        'agent_history': torch.cat([item['agent_history'] for item in batch], dim=0),
        'gt_trajectory': torch.cat([item['gt_trajectory'] for item in batch], dim=0),
    }


# ============================================================
# Inspection and Test Functions
# ============================================================

def print_scene_details(scene: Scene, idx: int = 0):
    """Print detailed information about a NAVSIM scene."""
    print(f"\n{'='*70}")
    print(f"SCENE #{idx} - DETAILED INSPECTION")
    print(f"{'='*70}")
    
    # Scene metadata
    print("\n[SCENE METADATA]")
    if hasattr(scene, 'scene_metadata'):
        meta = scene.scene_metadata
        print(f"  Log name: {meta.log_name if hasattr(meta, 'log_name') else 'N/A'}")
        print(f"  Scenario type: {meta.scenario_type if hasattr(meta, 'scenario_type') else 'N/A'}")
        print(f"  Initial timestamp: {meta.initial_timestamp if hasattr(meta, 'initial_timestamp') else 'N/A'}")
    
    # Frames
    print(f"\n[FRAMES]")
    print(f"  Total frames: {len(scene.frames)}")
    print(f"  History frames: {len([f for i, f in enumerate(scene.frames) if i < 4])}")
    print(f"  Current frame: frame[3]")
    print(f"  Future frames: {len([f for i, f in enumerate(scene.frames) if i >= 4])}")
    
    # Current frame details
    current_frame = scene.frames[3]  # Last history frame
    print(f"\n[CURRENT FRAME - frame[3]]")
    print(f"  Token: {current_frame.token}")
    print(f"  Timestamp: {current_frame.timestamp}")
    
    # Ego vehicle
    print(f"\n[EGO VEHICLE STATE]")
    ego = current_frame.ego_status
    print(f"  Position (x, y, heading):")
    print(f"    ego_pose: {ego.ego_pose}")
    print(f"  Velocity (vx, vy):")
    print(f"    ego_velocity: {ego.ego_velocity}")
    print(f"  Acceleration (ax, ay):")
    print(f"    ego_acceleration: {ego.ego_acceleration}")
    print(f"  Driving command:")
    print(f"    {ego.driving_command} (shape: {ego.driving_command.shape})")
    print(f"  Global frame: {ego.in_global_frame}")
    
    # LiDAR
    print(f"\n[LIDAR DATA]")
    if current_frame.lidar is not None:
        lidar = current_frame.lidar
        print(f"  Available: Yes")
        print(f"  Path: {lidar.lidar_path}")
        print(f"  Point cloud shape: {lidar.lidar_pc.shape}")
        print(f"  Point cloud dtype: {lidar.lidar_pc.dtype}")
        print(f"  Number of points: {lidar.lidar_pc.shape[1]}")
        print(f"  Channels: {lidar.lidar_pc.shape[0]} (x, y, z, intensity, ring, timestamp)")
        print(f"  X range: [{lidar.lidar_pc[0].min():.2f}, {lidar.lidar_pc[0].max():.2f}]")
        print(f"  Y range: [{lidar.lidar_pc[1].min():.2f}, {lidar.lidar_pc[1].max():.2f}]")
        print(f"  Z range: [{lidar.lidar_pc[2].min():.2f}, {lidar.lidar_pc[2].max():.2f}]")
    else:
        print(f"  Available: No")
    
    # Cameras
    print(f"\n[CAMERA DATA]")
    if current_frame.cameras is not None:
        cameras = current_frame.cameras
        cam_list = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']
        for cam_name in cam_list:
            if hasattr(cameras, cam_name):
                cam = getattr(cameras, cam_name)
                if cam is not None and hasattr(cam, 'camera_path'):
                    print(f"  {cam_name}: Available")
                    if hasattr(cam, 'camera_path'):
                        print(f"    Path: {cam.camera_path}")
    else:
        print(f"  Available: No")
    
    # Annotations (other agents)
    print(f"\n[ANNOTATIONS - Other Agents]")
    if hasattr(current_frame, 'annotations') and current_frame.annotations is not None:
        annot = current_frame.annotations
        if hasattr(annot, 'boxes'):
            print(f"  Number of agents: {len(annot.boxes) if hasattr(annot.boxes, '__len__') else 0}")
            if hasattr(annot.boxes, '__len__') and len(annot.boxes) > 0:
                print(f"  Agent boxes shape: {annot.boxes.shape if hasattr(annot.boxes, 'shape') else 'N/A'}")
        if hasattr(annot, 'labels'):
            print(f"  Labels available: Yes")
    else:
        print(f"  Available: No")
    
    # Traffic lights
    print(f"\n[TRAFFIC LIGHTS]")
    if hasattr(current_frame, 'traffic_lights') and current_frame.traffic_lights is not None:
        print(f"  Number of traffic lights: {len(current_frame.traffic_lights)}")
        if len(current_frame.traffic_lights) > 0:
            print(f"  First traffic light: {current_frame.traffic_lights[0]}")
    else:
        print(f"  Available: No")
    
    # Roadblocks
    print(f"\n[ROADBLOCKS]")
    if hasattr(current_frame, 'roadblock_ids') and current_frame.roadblock_ids is not None:
        print(f"  Number of roadblock IDs: {len(current_frame.roadblock_ids)}")
        if len(current_frame.roadblock_ids) > 0:
            print(f"  Sample IDs: {current_frame.roadblock_ids[:3]}")
    else:
        print(f"  Available: No")
    
    # Future trajectory
    print(f"\n[FUTURE TRAJECTORY]")
    future_traj = scene.get_future_trajectory(num_trajectory_frames=8)
    print(f"  Poses shape: {future_traj.poses.shape}")
    print(f"  Poses dtype: {future_traj.poses.dtype}")
    print(f"  First pose (x, y, heading): {future_traj.poses[0]}")
    print(f"  Last pose (x, y, heading): {future_traj.poses[-1]}")
    
    # Map API
    print(f"\n[MAP API]")
    if hasattr(scene, 'map_api') and scene.map_api is not None:
        print(f"  Available: Yes")
        print(f"  Type: {type(scene.map_api).__name__}")
    else:
        print(f"  Available: No")
    
    print(f"\n{'='*70}\n")


# ============================================================
# Main Test Script
# ============================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("NAVSIM DATASET - ORIGINAL DATA INSPECTION & TEST")
    print("=" * 70)
    
    # Create dataset
    print("\n[1] Creating Dataset...")
    dataset = NavsimDataset(
        split='mini',
        bev_size=(64, 64),
        use_cache=False,  # Disable cache for testing
        max_scenes=100  # Limit for faster testing
    )
    
    print(f"\n  Dataset created: {len(dataset)} scenes")
    
    # Inspect original NAVSIM data structure
    print("\n[2] Inspecting Original NAVSIM Scene...")
    raw_scene = dataset.get_raw_scene(0)
    print_scene_details(raw_scene, idx=0)
    
    # Test processed sample
    print("\n[3] Testing Processed Sample...")
    sample = dataset[0]
    
    print("\nProcessed Sample Shapes:")
    print(f"  BEV:           {sample['context']['BEV'].shape}")
    print(f"  LiDAR:         {sample['context']['lidar'].shape}")
    print(f"  Agent state:   {sample['agent_states'].shape}")
    print(f"  Agent history: {sample['agent_history'].shape}")
    print(f"  GT trajectory: {sample['gt_trajectory'].shape}")
    
    print("\nProcessed Sample Values:")
    print(f"  Agent state (x, y, vx, vy, heading):")
    print(f"    {sample['agent_states'][0]}")
    print(f"  First history state:")
    print(f"    {sample['agent_history'][0, 0]}")
    print(f"  First GT waypoint:")
    print(f"    {sample['gt_trajectory'][0, 0]}")
    print(f"  Last GT waypoint:")
    print(f"    {sample['gt_trajectory'][0, -1]}")
    
    # Test dataloader
    print("\n[4] Testing DataLoader with batch_size=4...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    
    print("\nBatch Shapes:")
    print(f"  BEV:           {batch['context']['BEV'].shape}")
    print(f"  LiDAR:         {batch['context']['lidar'].shape}")
    print(f"  Agent state:   {batch['agent_states'].shape}")
    print(f"  Agent history: {batch['agent_history'].shape}")
    print(f"  GT trajectory: {batch['gt_trajectory'].shape}")
    
    # Compare original vs processed
    print("\n[5] Original vs Processed Comparison...")
    raw_scene = dataset.get_raw_scene(0)
    processed = dataset[0]
    
    current_frame = raw_scene.frames[3]
    ego = current_frame.ego_status
    
    print("\nOriginal NAVSIM:")
    print(f"  ego_pose: {ego.ego_pose}")
    print(f"  ego_velocity: {ego.ego_velocity}")
    
    print("\nProcessed TransDiffuser:")
    print(f"  agent_states: {processed['agent_states'][0]}")
    
    print("\nFuture Trajectory (original):")
    future_traj = raw_scene.get_future_trajectory(num_trajectory_frames=8)
    print(f"  First pose: {future_traj.poses[0]}")
    print(f"  Last pose: {future_traj.poses[-1]}")
    
    print("\nFuture Trajectory (processed):")
    print(f"  First waypoint: {processed['gt_trajectory'][0, 0]}")
    print(f"  Last waypoint: {processed['gt_trajectory'][0, -1]}")
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED!")
    print("=" * 70)