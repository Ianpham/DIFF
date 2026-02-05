"""
NavsimDataset - With UniAD BEV Features
========================================
Uses precomputed BEV features from UniAD encoder with optional upsampling.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import torch.nn.functional as F

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

from .base import BaseNavsimDataset
# make sure you do import in right relative path, if not, the contract will return False everytime.

from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType

# Import your existing utilities
from navsim_utilize.enhancenavsim import BEVLabelExtractor

class NavsimDataset(BaseNavsimDataset):
    """
    NAVSIM dataset with UniAD BEV features.
    
    Provides:
    - LiDAR BEV (rasterized)
    - Camera BEV (from UniAD)
    - BEV labels (12 channels)
    - Agent state (5D or 7D with acceleration)
    - Agent history (4 timesteps)
    - GT trajectory
    
    Resolution Modes:
    - interpolate_bev=True: All features at (200, 200) - upsamples UniAD
    - interpolate_bev=False: Mixed resolution - UniAD at (64, 64), labels at target size
    """
    
    def __init__(
        self,
        data_split: str = "mini",
        bev_size: Tuple[int, int] = (200, 200),  # Target resolution for labels
        bev_range: float = 50.0,
        extract_labels: bool = True,
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        compute_acceleration: bool = False,
        use_uniad_bev: bool = True,  # Use UniAD BEV features
        uniad_cache_dir: str = None,  # Path to precomputed BEV cache
        interpolate_bev: bool = False,  # NEW: Whether to upsample UniAD features
    ):
        """
        Initialize NAVSIM dataset.
        
        Args:
            data_split: Dataset split ('mini', 'train', 'val', etc.)
            bev_size: Target BEV size for labels/LiDAR
            bev_range: Range in meters (default: 50m)
            extract_labels: Extract HD map labels
            map_root: Path to map files
            map_version: Map version string
            compute_acceleration: Include acceleration in agent state
            use_uniad_bev: Use UniAD BEV encoder features
            uniad_cache_dir: Directory containing cached UniAD features
            interpolate_bev: Whether to upsample UniAD to match bev_size
                           - True: Upsample UniAD (64,64)→(200,200), slower but aligned
                           - False: Keep native UniAD (64,64), faster, mixed resolution
        """
        super().__init__()
        
        self.data_split = data_split
        self.bev_size = bev_size  # Target for labels/LiDAR
        self.bev_range = bev_range
        self.extract_labels = extract_labels
        self.compute_acceleration = compute_acceleration
        self.use_uniad_bev = use_uniad_bev
        self.interpolate_bev = interpolate_bev
        
        # Paths
        self.data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        # UniAD BEV cache setup
        self.uniad_bev_size = None  # Will be detected from cache
        if use_uniad_bev:
            if uniad_cache_dir is None:
                # Default to the path you mentioned
                uniad_cache_dir = Path(os.environ.get(
                    'UNIAD_CACHE_DIR',
                    '/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/bev_cache/uniad_features'
                ))
            self.uniad_cache_dir = Path(uniad_cache_dir)
            
            if not self.uniad_cache_dir.exists():
                print(f"⚠ Warning: UniAD cache directory not found: {self.uniad_cache_dir}")
                print(f"  Please run: python uniad_segmentation.py --precompute-all")
                print(f"  Falling back to LiDAR-based BEV")
                self.use_uniad_bev = False
                self.bev_channels = 32
            else:
                print(f"✓ Using precomputed UniAD BEV from: {self.uniad_cache_dir}")
                # Detect BEV feature dimensions from first available file
                self._detect_bev_dimensions()
        else:
            self.bev_channels = 32  # Fallback placeholder channels
            self.uniad_bev_size = bev_size
        
        # Map setup
        if map_root is None:
            map_root = os.environ.get('NUPLAN_MAPS_ROOT')
            if map_root is None:
                raise ValueError("Map root not specified!")
        self.map_root = map_root
        self.map_version = map_version
        
        # History length
        self.history_length = 4
        self.future_horizon = 8
        
        # Initialize scene loader
        self._init_scene_loader()
        
        # Initialize label extractor at target resolution
        if extract_labels:
            self.label_extractor = BEVLabelExtractor(
                bev_size,  # Always use target resolution for labels
                bev_range,
                map_root=map_root,
                map_version=map_version
            )
        else:
            self.label_extractor = None
        
        # Print configuration
        print(f"✓ Initialized {self.__class__.__name__}: {len(self)} scenes")
        print(f"  Label/LiDAR BEV size: {self.bev_size}")
        if self.use_uniad_bev:
            print(f"  UniAD BEV size: {self.uniad_bev_size}")
            if self.interpolate_bev:
                if self.uniad_bev_size != self.bev_size:
                    print(f"  ✓ Will upsample UniAD: {self.uniad_bev_size} → {self.bev_size}")
                else:
                    print(f"  ✓ UniAD already at target size, no upsampling needed")
            else:
                print(f"  Mixed resolution mode: UniAD stays at {self.uniad_bev_size}")
    
    def _detect_bev_dimensions(self):
        """Detect BEV feature dimensions from cached files."""
        # Find first .pt file in cache
        cache_files = list(self.uniad_cache_dir.glob("*_bev.pt"))
        
        if not cache_files:
            print(f"⚠ No cached BEV files found in {self.uniad_cache_dir}")
            self.use_uniad_bev = False
            self.bev_channels = 32  # fallback
            self.uniad_bev_size = self.bev_size
            return
        
        # Load first file to get dimensions
        try:
            sample_bev = torch.load(cache_files[0], map_location='cpu')
            self.bev_channels = sample_bev.shape[0]  # C from (C, H, W)
            cached_h, cached_w = sample_bev.shape[1], sample_bev.shape[2]
            self.uniad_bev_size = (cached_h, cached_w)
            
            print(f"  UniAD BEV channels: {self.bev_channels}")
            print(f"  UniAD cached size: {self.uniad_bev_size}")
                
        except Exception as e:
            print(f"⚠ Error loading sample BEV: {e}")
            self.use_uniad_bev = False
            self.bev_channels = 32
            self.uniad_bev_size = self.bev_size

    def _build_contract(self) -> DataContract:
        """Declare what NavsimDataset provides."""
        
        builder = ContractBuilder(dataset_name="NavsimDataset")
        
        # Add features
        builder.add_feature(
            FeatureType.LIDAR_BEV,
            shape=(2, *self.bev_size),
            dtype="float32",
            description=f"Rasterized LiDAR BEV (density + height) at {self.bev_size}",
        )
        
        # Camera BEV shape depends on interpolation setting
        if self.use_uniad_bev:
            if self.interpolate_bev:
                camera_bev_size = self.bev_size  # Upsampled to match labels
                desc = f"UniAD BEV features (upsampled to {self.bev_size})"
            else:
                camera_bev_size = self.uniad_bev_size  # Native resolution
                desc = f"UniAD BEV features (native {self.uniad_bev_size})"
        else:
            camera_bev_size = self.bev_size
            desc = "Placeholder camera BEV"
        
        camera_bev_channels = self.bev_channels if self.use_uniad_bev else 32
        builder.add_feature(
            FeatureType.CAMERA_BEV,
            shape=(camera_bev_channels, *camera_bev_size),
            dtype="float32",
            description=desc,
        )
        
        if self.extract_labels:
            builder.add_feature(
                FeatureType.BEV_LABELS,
                shape=(12, *self.bev_size),
                dtype="float32",
                description=f"HD map semantic labels (12 channels) at {self.bev_size}",
            )
        
        # Agent state dimension depends on compute_acceleration
        agent_dim = 7 if self.compute_acceleration else 5
        builder.add_feature(
            FeatureType.AGENT_STATE,
            shape=(1, agent_dim),
            dtype="float32",
            description=f"Agent state ({'with' if self.compute_acceleration else 'without'} acceleration)",
        )
        
        builder.add_feature(
            FeatureType.AGENT_HISTORY,
            shape=(1, self.history_length, 5),
            dtype="float32",
            description="Agent trajectory history",
        )
        
        builder.add_feature(
            FeatureType.GT_TRAJECTORY,
            shape=(1, self.future_horizon, 5),
            dtype="float32",
            description="Ground truth future trajectory",
        )
        
        # Set physical limits based on resolution
        if self.interpolate_bev and self.bev_size == (200, 200):
            max_batch = 8
            memory_mb = 160.0
        else:
            max_batch = 16
            memory_mb = 50.0
        
        builder.set_physical_limits(
            max_batch_size=max_batch,
            memory_footprint_mb=memory_mb,
        )
        
        # Set semantic info
        builder.set_semantic_info(
            num_cameras=0,
            bev_channels=12 if self.extract_labels else 0,
            agent_state_dim=agent_dim,
            history_length=self.history_length,
            has_acceleration=self.compute_acceleration,
            has_nearby_agents=False,
            has_vector_maps=False,
        )
        
        return builder.build()
    
    def _init_scene_loader(self):
        """Initialize NAVSIM scene loader."""
        sensor_config = SensorConfig(
            cam_f0=False, cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=True
        )
        
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=self.history_length,
            num_future_frames=self.future_horizon,
        )
        
        self.scene_loader = SceneLoader(
            data_path=self.data_root / 'mini_navsim_logs' / self.data_split,
            original_sensor_path=self.data_root / 'mini_sensor_blobs' / self.data_split,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
    
    def __len__(self):
        return len(self.scene_loader.tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        
        return self._process_scene(scene, token)
    
    def _load_uniad_bev(self, token: str) -> Optional[torch.Tensor]:
        """
        Load precomputed UniAD BEV features with optional upsampling.
        
        Args:
            token: Scene token
            
        Returns:
            BEV features at native or target resolution, or None if failed
        """
        bev_path = self.uniad_cache_dir / f"{token}_bev.pt"
        
        if not bev_path.exists():
            # Don't print warning for every sample, just return None
            return None
        
        try:
            # Load cached features [C, H_native, W_native]
            bev_features = torch.load(bev_path, map_location='cpu')
            
            # Optionally upsample to target resolution
            if self.interpolate_bev:
                current_size = (bev_features.shape[1], bev_features.shape[2])
                
                if current_size != self.bev_size:
                    # Upsample: e.g., (64, 64) → (200, 200)
                    bev_features = F.interpolate(
                        bev_features.unsqueeze(0),  # Add batch dim: [1, C, H, W]
                        size=self.bev_size,         # Target: (200, 200)
                        mode='bilinear',            # Smooth interpolation
                        align_corners=False
                    )
                    bev_features = bev_features.squeeze(0)  # Remove batch: [C, H, W]
            
            return bev_features.float()
            
        except Exception as e:
            print(f"⚠ Error loading UniAD BEV for {token}: {e}")
            return None
    
    def _process_scene(self, scene, token: str) -> Dict:
        """Process scene to extract all modalities."""
        current_frame = scene.frames[self.history_length - 1]
        
        # ===================================================================
        # 1. LiDAR BEV - Always at target resolution
        # ===================================================================
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_bev = torch.zeros(2, *self.bev_size)
        
        # ===================================================================
        # 2. Camera BEV - Use UniAD features or fallback
        # ===================================================================
        if self.use_uniad_bev:
            # Try to load UniAD BEV features
            camera_bev = self._load_uniad_bev(token)
            
            if camera_bev is None:
                # Fallback: Use duplicated LiDAR as placeholder
                # Match the expected output size
                target_size = self.bev_size if self.interpolate_bev else self.uniad_bev_size
                
                if target_size == self.bev_size:
                    # Use LiDAR at same resolution
                    camera_bev = lidar_bev.repeat(self.bev_channels // 2, 1, 1)
                else:
                    # Downsample LiDAR to match UniAD native size
                    lidar_downsampled = F.interpolate(
                        lidar_bev.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    camera_bev = lidar_downsampled.repeat(self.bev_channels // 2, 1, 1)
        else:
            # Not using UniAD - create placeholder from LiDAR
            camera_bev = lidar_bev.repeat(16, 1, 1)
        
        # ===================================================================
        # 3. BEV Labels - Always at target resolution
        # ===================================================================
        if self.extract_labels and self.label_extractor is not None:
            try:
                labels = self.label_extractor.extract_all_labels(
                    scene, self.history_length - 1
                )
                labels_tensor = {
                    k: torch.from_numpy(v).float() 
                    for k, v in labels.items()
                }
            except Exception as e:
                print(f"⚠ Label extraction failed for {token}: {e}")
                labels_tensor = self._get_empty_labels()
        else:
            labels_tensor = {}
        
        # ===================================================================
        # 4. Agent States
        # ===================================================================
        ego = current_frame.ego_status
        
        if self.compute_acceleration and self.history_length > 1:
            # Compute acceleration from velocity change
            prev_ego = scene.frames[self.history_length - 2].ego_status
            dt = 0.1  # 10Hz
            ax = (ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
            ay = (ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
            
            agent_states = torch.tensor(
                [[ego.ego_pose[0], ego.ego_pose[1], 
                  ego.ego_velocity[0], ego.ego_velocity[1],
                  ax, ay, ego.ego_pose[2]]],
                dtype=torch.float32
            )
        else:
            agent_states = torch.tensor(
                [[ego.ego_pose[0], ego.ego_pose[1], 
                  ego.ego_velocity[0], ego.ego_velocity[1], 
                  ego.ego_pose[2]]],
                dtype=torch.float32
            )
        
        # ===================================================================
        # 5. Agent History
        # ===================================================================
        history_states = []
        for i in range(self.history_length):
            ego = scene.frames[i].ego_status
            state = torch.tensor(
                [[ego.ego_pose[0], ego.ego_pose[1], 
                  ego.ego_velocity[0], ego.ego_velocity[1], 
                  ego.ego_pose[2]]],
                dtype=torch.float32
            )
            history_states.append(state)
        agent_history = torch.cat(history_states, dim=0).unsqueeze(0)
        
        # ===================================================================
        # 6. Ground Truth Trajectory
        # ===================================================================
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=self.future_horizon
        )
        poses = future_traj.poses
        waypoints = []
        current_ego = current_frame.ego_status
        
        for i in range(len(poses)):
            x, y, heading = poses[i]
            if i > 0:
                prev_x, prev_y, _ = poses[i-1]
                dt = 0.5
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            else:
                vx = current_ego.ego_velocity[0]
                vy = current_ego.ego_velocity[1]
            waypoints.append([x, y, vx, vy, heading])
        
        gt_trajectory = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
        
        # ===================================================================
        # 7. Return Complete Sample
        # ===================================================================
        return {
            'camera_bev': camera_bev,      # [C, H, W] - size depends on interpolate_bev
            'lidar_bev': lidar_bev,        # [2, bev_size[0], bev_size[1]]
            'labels': labels_tensor,       # Dict with [bev_size[0], bev_size[1]] tensors
            'agent_states': agent_states,  # [1, 5] or [1, 7]
            'agent_history': agent_history,# [1, 4, 5]
            'gt_trajectory': gt_trajectory,# [1, 8, 5]
            'token': token,
        }
    
    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR to BEV (density + height) at target resolution."""
        H, W = self.bev_size
        x, y = point_cloud[:, 0], point_cloud[:, 1]
        z = point_cloud[:, 2] if point_cloud.shape[1] >= 3 else np.zeros_like(x)
        
        x_min, x_max = -self.bev_range, self.bev_range
        y_min, y_max = -self.bev_range, self.bev_range
        
        x_indices = ((x - x_min) / (x_max - x_min) * W).astype(int)
        y_indices = ((y - y_min) / (y_max - y_min) * H).astype(int)
        
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices, y_indices, z = x_indices[valid], y_indices[valid], z[valid]
        
        density = np.zeros((H, W), dtype=np.float32)
        height = np.zeros((H, W), dtype=np.float32)
        
        for i in range(len(x_indices)):
            density[y_indices[i], x_indices[i]] += 1
            height[y_indices[i], x_indices[i]] = max(height[y_indices[i], x_indices[i]], z[i])
        
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()
    
    def _get_empty_labels(self) -> Dict[str, torch.Tensor]:
        """Return empty labels at target resolution."""
        return {
            'drivable_area': torch.zeros(self.bev_size, dtype=torch.float32),
            'lane_boundaries': torch.zeros(self.bev_size, dtype=torch.float32),
            'lane_dividers': torch.zeros(self.bev_size, dtype=torch.float32),
            'vehicle_occupancy': torch.zeros(self.bev_size, dtype=torch.float32),
            'pedestrian_occupancy': torch.zeros(self.bev_size, dtype=torch.float32),
            'velocity_x': torch.zeros(self.bev_size, dtype=torch.float32),
            'velocity_y': torch.zeros(self.bev_size, dtype=torch.float32),
            'ego_mask': torch.zeros(self.bev_size, dtype=torch.float32),
            'traffic_lights': torch.zeros(self.bev_size, dtype=torch.uint8),
            'vehicle_classes': torch.zeros(self.bev_size, dtype=torch.uint8),
            'crosswalks': torch.zeros(self.bev_size, dtype=torch.float32),
            'stop_lines': torch.zeros(self.bev_size, dtype=torch.float32),
        }