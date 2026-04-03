"""
NavsimDataset - With UniAD BEV Features
========================================
Uses precomputed BEV features from UniAD encoder with optional upsampling.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import os
import torch.nn.functional as F

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataclasses import Scene, Frame
from data.base import BaseNavsimDataset
# make sure you do import in right relative path, if not, the contract will return False everytime.

from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
from datasets.navsim.navsim.common.enums import BoundingBoxIndex
from datasets.navsim.navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from shapely.geometry import Point as ShapelyPoint
# Import existing utilities
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
        max_agents: int = 32, # New: set max agent that model count and handle nearby.
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
                           - True: Upsample UniAD (64,64)->(200,200), slower but aligned
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
        self.max_agents = max_agents
        # Paths
        self.data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        # UniAD BEV cache setup
        self.uniad_bev_size = None  # Will be detected from cache

        if use_uniad_bev:
            if uniad_cache_dir is None:
                # Try environment variable first, then SLURM/Lustre path, then local fallback
                uniad_cache_dir = Path(os.environ.get(
                    'UNIAD_CACHE_DIR',
                    '/lustre/scratch/client/vinfast/users/vinfast_tamp/data/DPJI/transdiffuser/DDPM/datasets/navsim/download/bev_cache/uniad_features'
                ))

            self.uniad_cache_dir = Path(uniad_cache_dir)

            if not self.uniad_cache_dir.exists():
                # Try the local path as fallback
                local_fallback = Path('/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/bev_cache/uniad_features')
                if local_fallback.exists():
                    print(f"  UniAD cache not found at Lustre path, using local fallback: {local_fallback}")
                    self.uniad_cache_dir = local_fallback
                else:
                    print(f"  Warning: UniAD cache directory not found: {self.uniad_cache_dir}")
                    print(f"  Please run: python uniad_segmentation.py --precompute-all")
                    print(f"  Falling back to LiDAR-based BEV")
                    self.use_uniad_bev = False
                    self.bev_channels = 32
            
            if self.use_uniad_bev:  # Only proceed if not fallen back
                print(f"  Using precomputed UniAD BEV from: {self.uniad_cache_dir}")
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
        print(f"  Initialized {self.__class__.__name__}: {len(self)} scenes")
        print(f"  Label/LiDAR BEV size: {self.bev_size}")
        if self.use_uniad_bev:
            print(f"  UniAD BEV size: {self.uniad_bev_size}")
            if self.interpolate_bev:
                if self.uniad_bev_size != self.bev_size:
                    print(f"    Will upsample UniAD: {self.uniad_bev_size} -> {self.bev_size}")
                else:
                    print(f"    UniAD already at target size, no upsampling needed")
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

        # camera image
        if self.num_cameras > 0:
            builder.add_feature(
                FeatureType.CAMERA_IMAGES,
                shape=(self.num_cameras, 3, 224, 448),
                dtype="float32",
                description=f"RGB camera images from {self.num_cameras} cameras",
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
            num_cameras=self.num_cameras,  # WAS 0, now actual count
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
            cam_f0=True, cam_l0=True, cam_l1=False, cam_l2=False,
            cam_r0=True, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=True
        )
        # Track which cameras are enabled
        self.camera_names = [
            name for name, enabled in [
                ('cam_f0', sensor_config.cam_f0),
                ('cam_l0', sensor_config.cam_l0),
                ('cam_l1', sensor_config.cam_l1),
                ('cam_l2', sensor_config.cam_l2),
                ('cam_r0', sensor_config.cam_r0),
                ('cam_r1', sensor_config.cam_r1),
                ('cam_r2', sensor_config.cam_r2),
                ('cam_b0', sensor_config.cam_b0),
            ] if enabled
        ]

        self.num_cameras = len(self.camera_names)

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

    def _load_camera_images(
        self, frame, target_size: Tuple[int, int] = (224, 448)
    ) -> Optional[torch.Tensor]:
        """
        Load and stack camera images from a frame.
        
        Args:
            frame: Scene frame with camera data
            target_size: (H, W) to resize images to
            
        Returns:
            [num_cameras, 3, H, W] tensor or None if failed
        """
        images = []
        
        for cam_name in self.camera_names:
            try:
                # Access camera image from frame
                cam_data = getattr(frame.cameras, cam_name, None)
                
                if cam_data is None or cam_data.image is None:
                    # Create zero placeholder for this camera
                    images.append(torch.zeros(3, *target_size))
                    continue
                
                img = cam_data.image  # numpy array, likely (H, W, 3) uint8
                
                if isinstance(img, np.ndarray):
                    # Convert HWC -> CHW, uint8 -> float32 [0, 1]
                    if img.ndim == 3 and img.shape[2] == 3:
                        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    elif img.ndim == 3 and img.shape[0] == 3:
                        img = torch.from_numpy(img).float() / 255.0
                    else:
                        images.append(torch.zeros(3, *target_size))
                        continue
                
                # Resize to target
                if img.shape[1:] != target_size:
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                images.append(img)
                
            except Exception as e:
                images.append(torch.zeros(3, *target_size))
        
        if not images:
            return None
        
        return torch.stack(images, dim=0)  # [num_cameras, 3, H, W]
    
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
                    # Upsample: e.g., (64, 64) -> (200, 200)
                    bev_features = F.interpolate(
                        bev_features.unsqueeze(0),  # Add batch dim: [1, C, H, W]
                        size=self.bev_size,         # Target: (200, 200)
                        mode='bilinear',            # Smooth interpolation
                        align_corners=False
                    )
                    bev_features = bev_features.squeeze(0)  # Remove batch: [C, H, W]
            
            return bev_features.float()
            
        except Exception as e:
            print(f"Error loading UniAD BEV for {token}: {e}")
            return None
    def _extract_multi_agent_states(
            self, 
            scene: Scene,
            current_frame_idx: int,
            max_agents: int = 32,
    )-> Tuple[torch.Tensor, List[str]]:
        """
        Extract unified multi-agent state tensor with ego as agent 0
        
        Combines ego vehicle and nearby dynamic agents (vehicles, pedestrians,
        bicycles) into a single tensor sorted by distance to ego.

        All coordinates are in the **ego frame** of the current timestep,
        which is the native frame of NAVSIM annotations — no transform needed.

        Args:
            scene:             NAVSIM Scene object.
            current_frame_idx: Index of the current (planning) frame.
            max_agents:        Maximum number of agents including ego (N).

        Returns:
            agent_states:  (N, 5)  float32  [x, y, vx, vy, heading]
                        Row 0 is always ego.  Rows [1..N-1] are nearby
                        agents sorted by distance, zero-padded if fewer
                        than max_agents exist.
            track_tokens:  List[str] of length N.  track_tokens[0] = 'ego'.
                        Used downstream by _extract_multi_agent_history
                        to match agents across frames.
        """
        DYNAMIC_AGENT_TYPES = {'vehicle', 'pedestrian', 'bicycle'}

        current_frame = scene.frames[current_frame_idx]
        ego = current_frame.ego_status
        annotations = current_frame.annotations
        # ----- Ego as agent 0 -----
        # In ego frame the ego vehicle is at origin with its own heading = 0,
        # but ego_pose may carry the *global* pose depending on the frame.
        # For consistency with annotations (which ARE in ego frame),
        # ego position is (0, 0) and heading is 0.

        ego_state = np.array([
            0.0,
            0.0,
            ego.ego_velocity[0],
            ego.ego_velocity[1],
            0.0,
        ], dtype = np.float32)

        nearby = [] # list of (distance , state_array, track_token)

        if annotations is not None and len(annotations.boxes) > 0:
            for i in range(len(annotations.boxes)):
                name = annotations.names[i]
                if name not in DYNAMIC_AGENT_TYPES:
                    continue

                box = annotations.boxes[i]
                # box layout: [x, y, heading, length, width, height, ...]
                # these are already in ego frame.

                x = float(box[BoundingBoxIndex.X])
                y = float(box[BoundingBoxIndex.Y])
                heading = float(box[BoundingBoxIndex.HEADING])

                vx = float(annotations.velocity_3d[i][0])
                vy = float(annotations.velocity_3d[i][1])

                distance = np.sqrt(x**2 + y**2)
                nearby.append(
                    (
                        distance,
                        np.array([x, y, vx, vy, heading], dtype = np.float32),
                        annotations.track_tokens[i],
                    )
                )

        # sort by distance (closest first)
        nearby.sort(key = lambda t: t[0])

        # assemble (N, 5) tensor
        # slot 0 = ego, slots 1..max_agents-1 = nearby (zero-padded)
        max_nearby = max_agents - 1
        nearby = nearby[:max_nearby]

        states = np.zeros((max_agents, 5), dtype= np.float32)
        states[0] = ego_state

        track_tokens = ['ego']
        for idx, (_, state_arr, token) in enumerate(nearby):
            states[idx + 1] = state_arr
            track_tokens.append(token)

        return torch.from_numpy(states), track_tokens
    
    def _extract_multi_agent_history(
            self,
            scene: Scene,
            current_frame_idx: int,
            track_tokens: List[str],
            max_agents: int = 32, 
            num_history: int = 4,
    ) -> torch.Tensor:
        """
        Extract multi-agent history aligned to the current ego frame.

        For each agent in track_tokens (row ordering from _extract_multi_agent_states),
        look backwards through history frames, find the same track_token, and
        transform its position into the *current* frame's ego coordinate system.

        The coordinate transform is necessary because each frame's annotations
        are in *that frame's* ego-local coordinates.  We use the global ego_pose
        from each frame to chain the transform:
            p_current = R_cur^{-1} @ (R_hist @ p_hist_local + t_hist - t_cur)

        Args:
            scene:             NAVSIM Scene object.
            current_frame_idx: Index of the planning frame (t=0).
            track_tokens:      Agent ordering from _extract_multi_agent_states.
                            track_tokens[0] = 'ego'.
            max_agents:        N dimension (must match _extract_multi_agent_states).
            num_history:       Number of history steps to look back.

        Returns:
            history: (N, T_hist, 7)  float32
                    [x, y, vx, vy, ax, ay, heading]  in current ego frame.
                    Ordered oldest -> newest (history[..., -1, :] ≈ current state).
                    Zero-filled for agents not visible in a given frame.    
        """

        N = max_agents
        T = num_history
        DYNAMIC_AGENT_TYPES = {'vehicle', 'pedestrian', 'bicycle'}

        history = np.zeros((N, T, 7), dtype = np.float32)

        # current  frame's global ego pose - the reference frame
        cur_ego = scene.frames[current_frame_idx].ego_status

        cur_x, cur_y, cur_h = cur_ego.ego_pose
        cos_cur = np.cos(-cur_h)
        sin_cur = np.sin(-cur_h)

        # determine which histroy frame indices to use
        # walk backward
        # T steps ending at current frame idx
        start_idx = max(0, current_frame_idx - T + 1)
        frame_indices = list(range(start_idx, current_frame_idx + 1))

        # if fewer than T frames available, they fill the *end* of the time axis
        #  (oldest first, so index 0 = index)
        time_offset = T - len(frame_indices)

        for t_local, frame_idx in enumerate(frame_indices):
            t_out = t_local + time_offset # position in output tensor
            frame = scene.frames[frame_idx]
            hist_ego = frame.ego_status
            hist_x, hist_y, hist_h = hist_ego.ego_pose

            # precompute transpose: hist-local -> global -> current local
            cos_hist = np.cos(hist_h)
            sin_hist = np.sin(hist_h)

            # ego position in hist frame is (0, 0); transform to current frame
            ego_global_x = hist_x
            ego_global_y = hist_y
            ego_dx = ego_global_x - cur_x
            ego_dy = ego_global_y - cur_y
            ego_in_cur_x = cos_cur * ego_dx - sin_cur * ego_dy
            ego_in_cur_y = sin_cur * ego_dx + cos_cur * ego_dy
            ego_heading_in_cur = hist_h - cur_h

            # ego velocity: rotate from hist-local to current-local
            hvx, hvy = hist_ego.ego_velocity
            ego_vx_cur = cos_cur * (cos_hist * hvx - sin_hist * hvy) \
                    - sin_cur * (sin_hist * hvx + cos_hist * hvy)
            ego_vy_cur = sin_cur * (cos_hist * hvx - sin_hist * hvy) \
                    + cos_cur * (sin_hist * hvx + cos_hist * hvy)

            # Ego acceleration
            hax, hay = hist_ego.ego_acceleration
            ego_ax_cur = cos_cur * (cos_hist * hax - sin_hist * hay) \
                    - sin_cur * (sin_hist * hax + cos_hist * hay)
            ego_ay_cur = sin_cur * (cos_hist * hax - sin_hist * hay) \
                    + cos_cur * (sin_hist * hax + cos_hist * hay)

            history[0, t_out] = [
                ego_in_cur_x, ego_in_cur_y,
                ego_vx_cur, ego_vy_cur,
                ego_ax_cur, ego_ay_cur,
                ego_heading_in_cur,
            ]

            # other agent
            annotations = frame.annotations
            if annotations is None or len(annotations.boxes) == 0:
                continue

            # Build a lookup: track_token -> index in this frame's annotations
            frame_token_to_idx = {}
            for i, name in enumerate(annotations.names):
                if name in DYNAMIC_AGENT_TYPES:
                    frame_token_to_idx[annotations.track_tokens[i]] = i

            for agent_row in range(1, N):
                # guard for indext out of list range
                if agent_row >= len(track_tokens):
                    continue
                tok = track_tokens[agent_row]
                if tok == '' or tok not in frame_token_to_idx:
                    continue  # agent not visible in this frame -> stays zero

                ann_idx = frame_token_to_idx[tok]
                box = annotations.boxes[ann_idx]

                # Annotation position is in hist-frame ego-local coords
                local_x = float(box[BoundingBoxIndex.X])
                local_y = float(box[BoundingBoxIndex.Y])
                local_heading = float(box[BoundingBoxIndex.HEADING])

                # hist-local -> global
                global_x = cos_hist * local_x - sin_hist * local_y + hist_x
                global_y = sin_hist * local_x + cos_hist * local_y + hist_y
                global_heading = local_heading + hist_h

                # global -> current-local
                dx = global_x - cur_x
                dy = global_y - cur_y
                agent_x = cos_cur * dx - sin_cur * dy
                agent_y = sin_cur * dx + cos_cur * dy
                agent_heading = global_heading - cur_h

                # Velocity: hist-local -> global -> current-local
                raw_vx = float(annotations.velocity_3d[ann_idx][0])
                raw_vy = float(annotations.velocity_3d[ann_idx][1])
                # hist-local to global
                gvx = cos_hist * raw_vx - sin_hist * raw_vy
                gvy = sin_hist * raw_vx + cos_hist * raw_vy
                # global to current-local
                agent_vx = cos_cur * gvx - sin_cur * gvy
                agent_vy = sin_cur * gvx + cos_cur * gvy

                # Acceleration: approximate from velocity if we have previous frame
                # For simplicity, set to 0 — the encoder can learn from velocity diffs
                agent_ax, agent_ay = 0.0, 0.0

                history[agent_row, t_out] = [
                    agent_x, agent_y,
                    agent_vx, agent_vy,
                    agent_ax, agent_ay,
                    agent_heading,
                ]
        return torch.from_numpy(history)
    def _extract_intersection_features(
            self,
            scene: Scene,
            scenario: NavSimScenario,
            current_frame_idx: int, 
            max_intersection: int = 4,
    )-> torch.Tensor:
        """Extract intersection context features for Group C.

        Queries the nuPlan map API for nearby intersections and computes
        geometric features relative to the ego vehicle.

        Feature vector per intersection: [in_intersection, approach_dist,
        turn_angle, right_of_way_proxy, dist_to_center]

        Args:
            scene:             NAVSIM Scene object.
            scenario:          NavSimScenario with map_api access.
            current_frame_idx: Index of the planning frame.
            max_intersections: Maximum number of intersection tokens (K).

        Returns:
            features: (K, 5) float32, zero-padded if fewer intersections found.
        """
        features = np.zeros((max_intersection, 5), dtype= np.float32)

        try:
            map_api = scenario.map_api
        except Exception:
            return torch.from_numpy(features)

        ego = scene.frames[current_frame_idx].ego_status
        ego_x, ego_y, ego_h = ego.ego_pose

        ego_point = Point2D(ego_x, ego_y)

        # query nearby intersections + lane connectors (which represent 
        # the traversable paths *through& intersection)
        try:
            nearby = map_api.get_proximal_map_objects(
                ego_point, 50.0,
                [SemanticMapLayer.INTERSECTION, SemanticMapLayer.ROADBLOCK_CONNECTOR],
            )
        
        except Exception:
            return torch.from_numpy(features)
        
        intersections = nearby.get(SemanticMapLayer.INTERSECTION, [])
        connectors = nearby.get(SemanticMapLayer.ROADBLOCK_CONNECTOR, [])

        if not intersections:
            return torch.from_numpy(features)
        
        # sort intersections by distance to ego
        scored = []
        ego_shapely = ShapelyPoint(ego_x, ego_y)

        for ix in intersections:
            centroid = ix.polygon.centroid
            dist = np.sqrt((centroid.x - ego_x) ** 2 + (centroid.y - ego_y) ** 2)
            in_intersection = float(ix.polygon.contains(ego_shapely))

            # approach distance: distance from ego to intersection boundary
            # negative if inside (use 0 for inside)
            approach_dist = ix.polygon.distance(ego_shapely) if not in_intersection else 0.0

            # turn angle: angle from ego heading to intersection centroid
            dx = centroid.x - ego_x
            dy = centroid.y - ego_y
            angle_to_center = np.arctan2(dy, dx)
            turn_angle = angle_to_center - ego_h

            # normalize to [-pi, pi]
            turn_angle = (turn_angle + np.pi) % (2 * np.pi) - np.pi

            # right of way proxy: number of connectors entering this intersection
            # more conntector -> more complex -> lower implicit priority
            # normalize to [0, 1] range
            connector_count = sum(
                1 for c in connectors
                if ix.polygon.intersects(c.polygon)
            )

            row_proxy = 1.0 / max(connector_count, 1.0)

            dist_to_center = dist
            scored.append((
                    dist,
                    np.array([in_intersection, approach_dist, turn_angle, row_proxy, dist_to_center],
                            dtype=np.float32),
            ))

        scored.sort(key = lambda t: t[0])

        for i, (_, feat) in enumerate(scored[:max_intersection]):
            features[i] = feat

        
        return torch.from_numpy(features)
        
    def _extract_goal_features(
            self,
            scene: Scene,
            scenario: NavSimScenario,
            current_frame_idx: int, 
            max_goals: int = 1,
    )-> torch.Tensor:
        """
        Extract goal/route features for Group C.

        Derives a goal representation from the route roadblock IDs stored
        in each frame.  The "goal" is approximated as the centerline endpoint
        of the last on-route roadblock, expressed in the current ego frame.

        Feature vector: [goal_rel_x, goal_rel_y, dist_to_goal,
                        heading_to_goal, route_length_remaining]

        Args:
            scene:             NAVSIM Scene object.
            scenario:          NavSimScenario with map_api access.
            current_frame_idx: Index of the planning frame.
            max_goals:         Number of goal tokens (K).  Typically 1.

        Returns:
            features: (K, 5) float32, zero-padded if route unavailable.
        
        """
        features = np.zeros((max_goals, 5), dtype=np.float32)

        current_frame = scene.frames[current_frame_idx]
        route_ids = current_frame.roadblock_ids

        if not route_ids:
            return torch.from_numpy(features)
        
        try:
            map_api = scenario.map_api
        except Exception:
            return torch.from_numpy(features)
        
        ego = current_frame.ego_status
        ego_x, ego_y, ego_h = ego.ego_pose
        cos_h = np.cos(-ego_h)
        sin_h = np.sin(-ego_h)

        # Walk the route roadblocks to find the goal point (end of last
        # roadblock's centerline) and accumulate route length.
        goal_x_global, goal_y_global = None, None
        route_length = 0.0

        for rb_id in route_ids:
            # Try both roadblock and roadblock-connector layers
            roadblock = None
            for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
                try:
                    roadblock = map_api.get_map_object(rb_id, layer)
                    if roadblock is not None:
                        break
                except Exception:
                    continue

            if roadblock is None:
                continue

            # Pick the first interior lane as representative centerline
            lanes = list(roadblock.interior_edges)
            if not lanes:
                continue

            lane = lanes[0]
            discrete_path = lane.baseline_path.discrete_path
            if not discrete_path:
                continue

            # Accumulate length
            route_length += lane.baseline_path.length

            # Update goal to the endpoint of this (last) roadblock's lane
            endpoint = discrete_path[-1]
            goal_x_global = endpoint.x
            goal_y_global = endpoint.y

        if goal_x_global is None:
            return torch.from_numpy(features)

        # Transform goal to ego frame
        dx = goal_x_global - ego_x
        dy = goal_y_global - ego_y
        goal_rel_x = cos_h * dx - sin_h * dy
        goal_rel_y = sin_h * dx + cos_h * dy

        dist_to_goal = np.sqrt(goal_rel_x ** 2 + goal_rel_y ** 2)
        heading_to_goal = np.arctan2(goal_rel_y, goal_rel_x)

        features[0] = [
            goal_rel_x,
            goal_rel_y,
            dist_to_goal,
            heading_to_goal,
            route_length,
        ]

        return torch.from_numpy(features)
    
    def _extract_traffic_control_features(
            self,
            scene: Scene,
            scenario: NavSimScenario,
            current_frame_idx: int, 
            max_lights: int = 8,
    )-> torch.Tensor:        
        """Extract traffic control features for Group C.

        Reads the traffic light status list from the current frame,
        resolves each lane_connector_id to a spatial position via the
        map API, and encodes features relative to ego.

        Feature vector per light: [rel_x, rel_y, is_red, distance, bearing]

        Args:
            scene:             NAVSIM Scene object.
            scenario:          NavSimScenario with map_api access.
            current_frame_idx: Index of the planning frame.
            max_lights:        Maximum number of traffic light tokens (K).

        Returns:
            features: (K, 5) float32, zero-padded if fewer lights found.
        """
        features = np.zeros((max_lights, 5), dtype= np.float32)

        current_frame = scene.frames[current_frame_idx]
        traffic_lights = current_frame.traffic_lights

        if not traffic_lights:
            return torch.from_numpy(features)
        
        try:
            map_api = scenario.map_api
        except Exception:
            return torch.from_numpy(features)
        

        ego = current_frame.ego_status
        ego_x, ego_y, ego_h = ego.ego_pose
        cos_h = np.cos(-ego_h)
        sin_h = np.sin(-ego_h)

        entries = []  # (distance, feature_row)

        for lane_connector_id, is_red in traffic_lights:
            lc_id = str(lane_connector_id)

            # Resolve the lane connector to get its spatial position
            try:
                lane_connector = map_api.get_map_object(
                    lc_id, SemanticMapLayer.LANE_CONNECTOR,
                )
            except Exception:
                continue

            if lane_connector is None:
                continue

            # Use the polygon centroid as the light's representative position
            try:
                centroid = lane_connector.polygon.centroid
                lc_x = centroid.x
                lc_y = centroid.y
            except Exception:
                continue

            # Transform to ego frame
            dx = lc_x - ego_x
            dy = lc_y - ego_y
            rel_x = cos_h * dx - sin_h * dy
            rel_y = sin_h * dx + cos_h * dy

            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)
            bearing = np.arctan2(rel_y, rel_x)

            entries.append((
                distance,
                np.array([rel_x, rel_y, float(is_red), distance, bearing],
                        dtype=np.float32),
            ))

        # Sort by distance, keep closest
        entries.sort(key=lambda t: t[0])

        for i, (_, feat) in enumerate(entries[:max_lights]):
            features[i] = feat

        return torch.from_numpy(features)
    def _extract_pedestrian_features(
        self,
        scene: Scene,
        scenario: NavSimScenario,
        current_frame_idx: int,
        max_pedestrians: int = 10,
        crosswalk_radius: float = 15.0,
    ) -> torch.Tensor:
        """Extract pedestrian-specific features for Group C.

        Filters annotations to pedestrians only and computes features
        relevant to pedestrian interaction reasoning: position in ego
        frame, velocity, and crosswalk proximity from the map API.

        Feature vector per pedestrian: [rel_x, rel_y, speed,
                                        heading_alignment, crosswalk_proximity]

        heading_alignment: dot-product of pedestrian velocity direction
            and the ego->ped vector, normalized.  Positive means the
            pedestrian is moving *toward* the ego's path.

        crosswalk_proximity: inverse-distance signal, clamped to [0, 1].
            1.0 = on a crosswalk, 0.0 = far from any crosswalk.

        Args:
            scene:             NAVSIM Scene object.
            scenario:          NavSimScenario with map_api access.
            current_frame_idx: Index of the planning frame.
        max_pedestrians:   Maximum pedestrian tokens (K).
            crosswalk_radius:  Map query radius for crosswalks (meters).

        Returns:
            features: (K, 5) float32, zero-padded if fewer pedestrians.
        """
        features = np.zeros((max_pedestrians, 5), dtype=np.float32)

        current_frame = scene.frames[current_frame_idx]
        annotations = current_frame.annotations

        if annotations is None or len(annotations.boxes) == 0:
            return torch.from_numpy(features)

        # Try to get map API for crosswalk queries
        map_api = None
        try:
            map_api = scenario.map_api
        except Exception:
            pass

        ego = current_frame.ego_status
        ego_x, ego_y, ego_h = ego.ego_pose
        cos_h = np.cos(ego_h)
        sin_h = np.sin(ego_h)

        candidates = []  # (distance, feature_row)

        for i, name in enumerate(annotations.names):
            if name != 'pedestrian':
                continue

            box = annotations.boxes[i]
            # Position in ego frame (annotations are ego-local)
            rel_x = float(box[BoundingBoxIndex.X])
            rel_y = float(box[BoundingBoxIndex.Y])

            vx = float(annotations.velocity_3d[i][0])
            vy = float(annotations.velocity_3d[i][1])
            speed = np.sqrt(vx ** 2 + vy ** 2)

            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)

            # Heading alignment: does ped move toward ego's forward path?
            # ego->ped vector is (rel_x, rel_y), ped velocity is (vx, vy)
            # Negative dot product means approaching ego's position
            norm_dist = max(distance, 1e-6)
            norm_speed = max(speed, 1e-6)
            heading_alignment = -(vx * rel_x + vy * rel_y) / (norm_dist * norm_speed)
            # Clamp to [-1, 1] for clean input
            heading_alignment = float(np.clip(heading_alignment, -1.0, 1.0))

            # Crosswalk proximity via map API
            crosswalk_proximity = 0.0
            if map_api is not None:
                try:
                    # Transform ped position to global for map query
                    ped_global_x = ego_x + cos_h * rel_x - sin_h * rel_y
                    ped_global_y = ego_y + sin_h * rel_x + cos_h * rel_y

                    from nuplan.common.actor_state.state_representation import Point2D
                    ped_point = Point2D(ped_global_x, ped_global_y)

                    nearby = map_api.get_proximal_map_objects(
                        ped_point, crosswalk_radius,
                        [SemanticMapLayer.CROSSWALK],
                    )
                    crosswalks = nearby.get(SemanticMapLayer.CROSSWALK, [])

                    if crosswalks:
                        from shapely.geometry import Point as ShapelyPoint
                        ped_shapely = ShapelyPoint(ped_global_x, ped_global_y)

                        min_dist = min(
                            cw.polygon.distance(ped_shapely) for cw in crosswalks
                        )
                        # Invert to proximity: 0 distance -> 1.0, far -> 0.0
                        crosswalk_proximity = float(
                            np.clip(1.0 - min_dist / crosswalk_radius, 0.0, 1.0)
                        )
                except Exception:
                    pass

            candidates.append((
                distance,
                np.array([rel_x, rel_y, speed, heading_alignment,
                        crosswalk_proximity], dtype=np.float32),
            ))

        # Sort by distance, keep closest
        candidates.sort(key=lambda t: t[0])

        for i, (_, feat) in enumerate(candidates[:max_pedestrians]):
            features[i] = feat

        return torch.from_numpy(features)
    
    
    def _process_scene(self, scene: Scene, token: str) -> Dict:
        """Process scene to extract all modalities."""
        current_frame_idx = self.history_length - 1
        current_frame = scene.frames[current_frame_idx]
        current_ego = current_frame.ego_status
        ref_x, ref_y, ref_heading = current_ego.ego_pose

        
        # Helpers                                                              
        # the pose is already relavtive mean that we dont need to convert to ego frame relative.
        def to_relative(pose_x: float, pose_y: float, pose_heading: float):
            """Convert absolute pose to current-ego-frame-relative."""
            dx = pose_x - ref_x
            dy = pose_y - ref_y
            cos_h = np.cos(-ref_heading)
            sin_h = np.sin(-ref_heading)
            rel_x = cos_h * dx - sin_h * dy
            rel_y = sin_h * dx + cos_h * dy
            rel_heading = pose_heading - ref_heading
            return rel_x, rel_y, rel_heading

        # Build scenario once — required by group-C feature extractors
        scenario = NavSimScenario(
            scene,
            map_root=self.map_root,
            map_version=self.map_version,
        )

        
        # 1. Camera images                                                     
        
        camera_images = None
        if self.num_cameras > 0:
            camera_images = self._load_camera_images(current_frame)
        if camera_images is None:
            camera_images = torch.zeros(max(self.num_cameras, 1), 3, 224, 448)

        
        # 2. LiDAR BEV  (always at target resolution)                         
        
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_bev = torch.zeros(2, *self.bev_size)

        
        # 3. Camera BEV  (UniAD features or LiDAR-based fallback)             
        
        if self.use_uniad_bev:
            camera_bev = self._load_uniad_bev(token)

            if camera_bev is None:
                # Fallback: tile LiDAR to expected channel count / spatial size
                target_size = self.bev_size if self.interpolate_bev else self.uniad_bev_size
                if target_size == self.bev_size:
                    camera_bev = lidar_bev.repeat(self.bev_channels // 2, 1, 1)
                else:
                    lidar_downsampled = F.interpolate(
                        lidar_bev.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
                    camera_bev = lidar_downsampled.repeat(self.bev_channels // 2, 1, 1)
        else:
            camera_bev = lidar_bev.repeat(16, 1, 1)

        
        # 4. BEV labels  (always at target resolution)                        
        
        if self.extract_labels and self.label_extractor is not None:
            try:
                labels = self.label_extractor.extract_all_labels(scene, current_frame_idx)
                labels_tensor = {
                    k: torch.from_numpy(v).float() for k, v in labels.items()
                }
            except Exception as e:
                print(f"Label extraction failed for {token}: {e}")
                labels_tensor = self._get_empty_labels()
        else:
            labels_tensor = {}

        
        # 5. Ego agent state  (ego frame: position = origin, heading = 0)     
        
        if self.compute_acceleration and self.history_length > 1:
            prev_ego = scene.frames[current_frame_idx - 1].ego_status
            dt = 0.1  # 10 Hz
            ax = (current_ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
            ay = (current_ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
            agent_state_ego = torch.tensor(
                [[0.0, 0.0,
                current_ego.ego_velocity[0], current_ego.ego_velocity[1],
                ax, ay, 0.0]],
                dtype=torch.float32,
            )
        else:
            agent_state_ego = torch.tensor(
                [[0.0, 0.0,
                current_ego.ego_velocity[0], current_ego.ego_velocity[1],
                0.0]],
                dtype=torch.float32,
            )

        
        # 6. Ego agent history  (poses relative to current ego frame)         
        
        history_states = []
        for i in range(self.history_length):
            hist_ego = scene.frames[i].ego_status
            rel_x, rel_y, rel_h = to_relative(
                hist_ego.ego_pose[0], hist_ego.ego_pose[1], hist_ego.ego_pose[2]
            )
            state = torch.tensor(
                [[rel_x, rel_y,
                hist_ego.ego_velocity[0], hist_ego.ego_velocity[1],
                rel_h]],
                dtype=torch.float32,
            )
            history_states.append(state)
        # shape: [1, history_length, 5]
        agent_history_ego = torch.cat(history_states, dim=0).unsqueeze(0)

        
        # 7. Ground-truth trajectory  (relative to current ego frame)         
        
        future_traj = scene.get_future_trajectory(num_trajectory_frames=self.future_horizon)
        poses = future_traj.poses  # already ego-relative [x, y, heading]

        waypoints = []
        for i, (x, y, heading) in enumerate(poses):
            # poses are already in ego frame — do NOT call to_relative()
            if i > 0:
                prev_x, prev_y, _ = poses[i - 1]
                dt = 0.5
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            else:
                vx = current_ego.ego_velocity[0]
                vy = current_ego.ego_velocity[1]
            waypoints.append([x, y, vx, vy, heading])

        gt_trajectory = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)

        
        # 8. Multi-agent states + history                                      
        
        multi_agent_states, track_tokens = self._extract_multi_agent_states(
            scene, current_frame_idx, max_agents=self.max_agents,
        )
        multi_agent_history = self._extract_multi_agent_history(
            scene, current_frame_idx, track_tokens,
            max_agents=self.max_agents, num_history=self.history_length,
        )

        
        # 9. Group-C context features                                         
        
        intersection_features = self._extract_intersection_features(
            scene, scenario, current_frame_idx, max_intersection=4,
        )
        goal_features = self._extract_goal_features(
            scene, scenario, current_frame_idx, max_goals=1,
        )
        traffic_control_features = self._extract_traffic_control_features(
            scene, scenario, current_frame_idx, max_lights=8,
        )
        pedestrian_features = self._extract_pedestrian_features(
            scene, scenario, current_frame_idx,
            max_pedestrians=10, crosswalk_radius=15.0,
        )

        
        # 10. Return                                                           
        
        return {
            # perception
            'camera_images':          camera_images,          # [num_cameras, 3, H, W]
            'camera_bev':             camera_bev,             # [C, H_bev, W_bev]
            'lidar_bev':              lidar_bev,              # [2, H, W]
            'labels':                 labels_tensor,          # Dict[str, Tensor[H, W]]
            # ego
            'agent_states':           agent_state_ego,        # [1, 5] or [1, 7]
            'agent_history':          agent_history_ego,      # [1, history_length, 5]
            'gt_trajectory':          gt_trajectory,          # [1, future_horizon, 5]
            # multi-agent
            'multi_agent_states':     multi_agent_states,     # [N, 5]
            'multi_agent_history':    multi_agent_history,    # [N, T_hist, 7]
            # context
            'intersection_features':  intersection_features,  # [4, 5]
            'goal_features':          goal_features,          # [1, 5]
            'traffic_control_features': traffic_control_features,  # [8, 5]
            'pedestrian_features':    pedestrian_features,    # [10, 5]
            # meta
            'token':                  token,
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

# santity check

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    # config - edit these two lines to match your environment
    MAP_ROOT        = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
    UNIAD_CACHE_DIR = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/bev_cache/uniad_features"
    DATA_SPLIT = "mini"
    TEST_INDICES = [0, 1, 2] # which data item to work

    os.environ.setdefault('NUPLAN_MAPS_ROOT',   MAP_ROOT)
    os.environ.setdefault('OPENSCENE_DATA_ROOT', str(Path(MAP_ROOT).parent))

    # helper
    W = 70

    def header(title: str):
        print("\n" + "=" * W)
        print(f" {title}")
        print("=" * W)

    def check(label: str, cond: bool, detail: str = ""):
        mask = "OK" if cond else "FAIL"
        msg = f" [{mask}] {label}"

        if detail:
            msg += f" ({detail})"
        
        print(msg)
        return cond
    
    def tensor_stats(t: torch.Tensor) -> str:
        return (f"shape = {tuple(t.shape)}  dtype= {t.dtype}"
                f"min = {t.min():.3f} max = {t.max():.3f}"
                f"nan = {torch.isnan(t).sum().item()}"             
                )
    
    all_ok = True

    # 1. Environments

    header("1. Environment")
    all_ok &= check("MAP_ROOT exists", Path(MAP_ROOT).exists(),     MAP_ROOT)
    all_ok &= check("UNIAD_CACHE_DIR exists", Path(UNIAD_CACHE_DIR).exists(), UNIAD_CACHE_DIR)

    # 2. Dataset construction  

    header("2 · Dataset construction")
    try:
        dataset = NavsimDataset(
            data_split      = DATA_SPLIT,
            bev_size        = (200, 200),
            bev_range       = 50.0,
            extract_labels  = True,
            map_root        = MAP_ROOT,
            map_version     = "nuplan-maps-v1.0",
            compute_acceleration = False,
            use_uniad_bev   = True,
            uniad_cache_dir = UNIAD_CACHE_DIR,
            interpolate_bev = False,
        )
        all_ok &= check("Dataset created", True, f"{len(dataset)} scenes")
    except Exception as exc:
        check("Dataset created", False, str(exc))
        print("\n  Cannot continue without a dataset. Exiting.")
        sys.exit(1)

    all_ok &= check("Dataset non-empty", len(dataset) > 0)

    # 3. Single - smaple smoke test
    header("3 · Single-sample smoke te" \
    "st  (index 0)")
    sample = dataset[0]

    EXPECTED_KEYS = {
        'camera_images', 'camera_bev', 'lidar_bev', 'labels',
        'agent_states', 'agent_history', 'gt_trajectory',
        'multi_agent_states', 'multi_agent_history',
        'intersection_features', 'goal_features',
        'traffic_control_features', 'pedestrian_features',
        'token',
    }
    missing = EXPECTED_KEYS - sample.keys()
    all_ok &= check("All keys present", not missing,
                    f"missing: {missing}" if missing else "")
    
    # shape checks
    header("4 · Tensor shapes")

    NC  = dataset.num_cameras
    BEV = dataset.bev_size           # (200, 200)
    C   = dataset.bev_channels       # UniAD channels
    UNI = dataset.uniad_bev_size     # (64, 64)
    H   = dataset.history_length     # 4
    F   = dataset.future_horizon     # 8
    N   = dataset.max_agents         # 32
    dim = 7 if dataset.compute_acceleration else 5

    shape_checks = [
        ("camera_images",          sample['camera_images'].shape,          (NC, 3, 224, 448)),
        ("camera_bev",             sample['camera_bev'].shape,             (C, *UNI)),
        ("lidar_bev",              sample['lidar_bev'].shape,              (2, *BEV)),
        ("agent_states",           sample['agent_states'].shape,           (1, dim)),
        ("agent_history",          sample['agent_history'].shape,          (1, H, 5)),
        ("gt_trajectory",          sample['gt_trajectory'].shape,          (1, F, 5)),
        ("multi_agent_states",     sample['multi_agent_states'].shape,     (N, 5)),
        ("multi_agent_history",    sample['multi_agent_history'].shape,    (N, H, 7)),
        ("intersection_features",  sample['intersection_features'].shape,  (4, 5)),
        ("goal_features",          sample['goal_features'].shape,          (1, 5)),
        ("traffic_control_features", sample['traffic_control_features'].shape, (8, 5)),
        ("pedestrian_features",    sample['pedestrian_features'].shape,    (10, 5)),
    ]

    for name, got, expected in shape_checks:
        all_ok &= check(f"{name:<30}", got == expected,
                        f"got {got}  expected {expected}")
        
    # 5 BEV label channels

    header("5 · BEV label channels")
    labels = sample['labels']
    all_ok &= check("12 label channels", len(labels) == 12, f"got {len(labels)}")
    for k, v in labels.items():
        ok = (v.shape == BEV) and torch.isfinite(v).all().item()
        all_ok &= check(f"  {k:<28}", ok, tensor_stats(v))

    # 6. coordinate-system sanity

    header("6 · Coordinate-system sanity")

    ego_x, ego_y = sample['agent_states'][0, 0].item(), sample['agent_states'][0, 1].item()
    ego_h        = sample['agent_states'][0, 4].item()
    all_ok &= check("Ego x == 0",       abs(ego_x) < 1e-4, f"x={ego_x:.6f}")
    all_ok &= check("Ego y == 0",       abs(ego_y) < 1e-4, f"y={ego_y:.6f}")
    all_ok &= check("Ego heading == 0", abs(ego_h) < 1e-4, f"h={ego_h:.6f}")

    hist = sample['agent_history'][0]           # [T, 5]
    last_x, last_y = hist[-1, 0].item(), hist[-1, 1].item()
    all_ok &= check("History last-step x ≈ 0", abs(last_x) < 1e-3, f"x={last_x:.6f}")
    all_ok &= check("History last-step y ≈ 0", abs(last_y) < 1e-3, f"y={last_y:.6f}")

    oldest_x = hist[0, 0].item()
    all_ok &= check("History oldest frame behind ego (x < 0)", oldest_x < 0,
                    f"oldest_x={oldest_x:.3f}")
    
    # 7. GT trajectory sanity

    header("7 · GT trajectory sanity")

    traj = sample['gt_trajectory'][0]           # [T, 5]
    all_ok &= check("Trajectory finite",        torch.isfinite(traj).all().item())
    all_ok &= check("First wp x > 0 (forward)", traj[0, 0].item() > 0,
                    f"x={traj[0, 0].item():.3f}")

    x_seq = traj[:, 0]
    mono  = (x_seq[1:] > x_seq[:-1]).all().item()
    all_ok &= check("x coords monotonically increasing", mono,
                    "straight-ahead drive expected")
    
    # 8. Multi-sample reproducibility 

    header("8 · Multi-sample reproducibility")
    for idx in TEST_INDICES:
        try:
            s = dataset[idx]
            ok = (
                s['camera_bev'].shape   == (C, *UNI) and
                s['lidar_bev'].shape    == (2, *BEV) and
                s['gt_trajectory'].shape == (1, F, 5) and
                torch.isfinite(s['gt_trajectory']).all().item()
            )
            all_ok &= check(f"Sample {idx}", ok,
                            f"token={s['token']}")
        except Exception as exc:
            all_ok &= check(f"Sample {idx}", False, str(exc))

    # 9. Detailed tensor report for sample 0 

    header("9 · Tensor statistics (sample 0)")
    for key in ['camera_bev', 'lidar_bev', 'agent_states',
                'agent_history', 'gt_trajectory',
                'multi_agent_states', 'multi_agent_history']:
        t = sample[key]
        print(f"  {key:<30}  {tensor_stats(t)}")

    # 10. NaN / Inf guard across all test samples   

    header("10 · NaN / Inf guard")
    TENSOR_KEYS = [
        'camera_bev', 'lidar_bev', 'agent_states', 'agent_history',
        'gt_trajectory', 'multi_agent_states', 'multi_agent_history',
        'intersection_features', 'goal_features',
        'traffic_control_features', 'pedestrian_features',
    ]
    for idx in TEST_INDICES:
        s = dataset[idx]
        for key in TENSOR_KEYS:
            t = s[key]
            finite = torch.isfinite(t).all().item()
            all_ok &= check(f"sample[{idx}]['{key}'] finite", finite)

    
    # Summary    

    header("Summary")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see [FAIL] lines above")
    print("=" * W + "\n")
    sys.exit(0 if all_ok else 1)