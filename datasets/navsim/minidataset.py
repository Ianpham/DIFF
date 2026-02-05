"""
Minimal NAVSIM Dataset Wrapper for TransDiffuser - Mini Split Only
Simple version for testing with the mini dataset
"""

import torch
import numpy as np
import os


from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple

# Import NAVSIM official classes
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig


class NavsimDataset(Dataset):
    """
    Minimal NAVSIM wrapper for TransDiffuser.
    Designed for testing with mini split.
    """
    
    def __init__(
        self,
        bev_size: Tuple[int, int] = (64, 64),
        history_length: int = 4,  # Mini split has 4 history frames
        future_horizon: int = 8,   # Mini split has 8 future frames
        use_cache: bool = True,
        precompute_bev: bool = True,
        cache_batch_size: int = 16
    ):
        """
        Minimal initialization for mini split.
        
        Args:
            bev_size: (H, W) size of BEV representation
            history_length: Number of historical timesteps (default 4 for mini)
            future_horizon: Number of future waypoints (default 8 for mini)
            use_cache: Whether to cache preprocessed data
        """
        # Get paths from environment
        data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        self.bev_size = bev_size
        self.history_length = history_length
        self.future_horizon = future_horizon
        self.use_cache = use_cache
        self.precompute_bev = precompute_bev
        self.cache_batch_size = cache_batch_size

        self.cache_dir = data_root / 'cache' / 'transdiffuser_mini'
        
        print("Initializing NAVSIM Dataset (Mini Split)...")
        
        # python -c "from navsim.common.dataclasses import SensorConfig; help(SensorConfig.__init__)"
        sensor_config = SensorConfig(
            cam_f0=True,   # Front camera
            cam_l0=False,  # Left cameras
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,  # Right cameras
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,  # Back camera
            lidar_pc=True  # LiDAR point cloud
        )
        
        # Simple scene filter - all scenes python -c "from navsim.common.dataclasses import SceneFilter; help(SceneFilter.__init__)"
        scene_filter = SceneFilter(
            log_names=None,  # All logs
            num_history_frames=history_length,  # 4
            num_future_frames=future_horizon,   # 8
        )
        # Create scene loader python -c "from navsim.common.dataloader import SceneLoader; help(SceneLoader.__init__)"
        # Create scene loader
        print(f"Loading from: {data_root}")
        self.scene_loader = SceneLoader(
            data_path=data_root / 'mini_navsim_logs' / 'mini',  # Changed!
            original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',  # Check if this exists too!
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
                        
        self.scene_tokens = self.scene_loader.tokens
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # loading lightweight BEV encoder
        if self.precompute_bev:
            self.bev_encoder = self._load_lightweight_bev()
            self.bev_encoder.eval()

            # precompute bev features in baches
            self._precompute_all_bev_features()

        
        
        print(f"✓ Loaded {len(self)} scenes from mini split")
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
# Quick Test Script
# ============================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("NAVSIM Mini Dataset Test")
    print("=" * 60)
    
    # Create dataset
    dataset = NavsimDataset(
        bev_size=(64, 64),
        use_cache=True
    )
    
    print(f"\n✓ Dataset created: {len(dataset)} scenes")
    
    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    
    print("\nSample shapes:")
    print(f"  BEV:           {sample['context']['BEV'].shape}")
    print(f"  LiDAR:         {sample['context']['lidar'].shape}")
    print(f"  Agent state:   {sample['agent_states'].shape}")
    print(f"  Agent history: {sample['agent_history'].shape}")
    print(f"  GT trajectory: {sample['gt_trajectory'].shape}")
    
    # Test dataloader
    print("\nTesting DataLoader with batch_size=4...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    
    print("\nBatch shapes:")
    print(f"  BEV:           {batch['context']['BEV'].shape}")
    print(f"  LiDAR:         {batch['context']['lidar'].shape}")
    print(f"  Agent state:   {batch['agent_states'].shape}")
    print(f"  Agent history: {batch['agent_history'].shape}")
    print(f"  GT trajectory: {batch['gt_trajectory'].shape}")
    print(f" BEV : {batch['context']['BEV'].any() == 0}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

