"""
EnhancedNavsimDataset - Full-Featured Dataset
==============================================
Provides all available features including:
- Raw LiDAR points + BEV
- All 8 camera images
- 12-channel BEV labels
- Full 7D agent states (with acceleration)
- Multi-agent data (nearby agents)
- Vector maps (optional)
- Route information (optional)
- Difficulty metrics
"""

import torch
import numpy as np
from pathlib import Path
import os
import warnings
from typing import Dict, List, Optional, Any

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .base import BaseNavsimDataset
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType

# Import your existing utilities
try:
    from navsim_utilize.enhancenavsim import (
        BEVLabelExtractor,
        NavsimScenarioBuilder,
        RouteExtractor,
        DifficultyAnalyzer,
        TrajectoryConfig,
    )
    from navsim_utilize.vectormapfeature import VectorMapExtractor
except ImportError:
    warnings.warn("Could not import navsim utilities. Some features may not work.")
    BEVLabelExtractor = None
    NavsimScenarioBuilder = None
    RouteExtractor = None
    VectorMapExtractor = None

class EnhancedNavsimDataset(BaseNavsimDataset):
    """
    Enhanced NAVSIM dataset with all features.
    
    Provides:
    - Raw LiDAR point clouds
    - LiDAR BEV
    - 8 camera images (all views)
    - BEV labels (12 channels)
    - Agent state (7D with acceleration)
    - Agent history (variable length)
    - Nearby agents (multi-agent)
    - GT trajectory
    - Vector maps (optional)
    - Route information (optional)
    - Difficulty metrics
    
    Physical constraints:
    - Max batch size: 8 (heavier data)
    - Memory: ~200 MB/sample
    """
    
    def __init__(
        self,
        data_split: str = "mini",
        bev_size: tuple = (200, 200),
        bev_range: float = 50.0,
        trajectory_sampling: Optional[TrajectorySampling] = None,
        extract_labels: bool = True,
        extract_route_info: bool = True,
        extract_vector_maps: bool = True,
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        sensor_config: SensorConfig = None,
    ):
        super().__init__()
        
        self.data_split = data_split
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.extract_labels = extract_labels
        self.extract_route_info = extract_route_info
        self.extract_vector_maps = extract_vector_maps
        
        # Paths
        self.data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        # Map setup
        if map_root is None:
            map_root = os.environ.get('NUPLAN_MAPS_ROOT')
            if map_root is None:
                raise ValueError("Map root not specified!")
        self.map_root = map_root
        self.map_version = map_version
        
        # Trajectory sampling
        if trajectory_sampling is None:
            self.trajectory_sampling = TrajectoryConfig.PLANNING_TRAJECTORY_SAMPLING
        else:
            self.trajectory_sampling = trajectory_sampling
        
        # Initialize components
        self._init_scene_loader()
        self._init_extractors()
        
        print(f"✓ Initialized {self.__class__.__name__}: {len(self)} scenes")

        # Make sure sensor_config is passed to scene loading/ for check sensor loading when we do have 8 cameras full, will consider to change in SensorConfig if we have 12 cameras set up.
        if sensor_config is None:
            sensor_config = SensorConfig.build_all_sensors(include=True)
        
        self.sensor_config = sensor_config
        
        # Debug print
        print(f"Dataset initialized with sensor_config: {self.sensor_config}")

    
    def _build_contract(self) -> DataContract:
        """Declare what EnhancedNavsimDataset provides."""
        
        builder = ContractBuilder(dataset_name="EnhancedNavsimDataset")
        
        # =====================================================================
        # Raw Sensors (High Quality)
        # =====================================================================
        builder.add_feature(
            FeatureType.LIDAR_POINTS,
            shape=(-1, 3),  # Variable number of points
            dtype="float32",
            description="Raw LiDAR point clouds",
        )
        
        builder.add_feature(
            FeatureType.LIDAR_BEV,
            shape=(2, *self.bev_size),
            dtype="float32",
            description="Rasterized LiDAR BEV (density + height)",
        )
        
        builder.add_feature(
            FeatureType.CAMERA_IMAGES,
            shape=(8, 3, 900, 1600),  # 8 cameras
            dtype="float32",
            description="All 8 camera views",
        )
        
        # =====================================================================
        # Processed Features
        # =====================================================================
        if self.extract_labels:
            builder.add_feature(
                FeatureType.BEV_LABELS,
                shape=(12, *self.bev_size),
                dtype="float32",
                description="HD map semantic labels (12 channels)",
            )
        
        if self.extract_vector_maps:
            builder.add_feature(
                FeatureType.VECTOR_MAP,
                shape=(-1,),  # Variable structure
                dtype="object",
                description="Structured vector map features",
            )
        
        # =====================================================================
        # Agent Features (Full 7D)
        # =====================================================================
        builder.add_feature(
            FeatureType.AGENT_STATE,
            shape=(1, 7),
            dtype="float32",
            description="Agent state with acceleration [x, y, vx, vy, ax, ay, heading]",
        )
        
        # History length from trajectory sampling
        history_len = int(
            self.trajectory_sampling.time_horizon / 
            self.trajectory_sampling.interval_length
        )
        
        builder.add_feature(
            FeatureType.AGENT_HISTORY,
            shape=(1, history_len, 7),
            dtype="float32",
            description="Agent trajectory history",
        )
        
        builder.add_feature(
            FeatureType.AGENT_NEARBY,
            shape=(1, 10, 7),  # Up to 10 nearby agents
            dtype="float32",
            description="Nearby agents (multi-agent)",
        )
        
        # =====================================================================
        # Ground Truth
        # =====================================================================
        builder.add_feature(
            FeatureType.GT_TRAJECTORY,
            shape=(1, self.trajectory_sampling.num_poses, 5),
            dtype="float32",
            description="Ground truth future trajectory",
        )
        
        # =====================================================================
        # Mission/Route
        # =====================================================================
        if self.extract_route_info:
            builder.add_feature(
                FeatureType.ROUTE,
                shape=(-1,),
                dtype="object",
                description="Navigation route information",
            )
        
        # =====================================================================
        # Difficulty/Metadata
        # =====================================================================
        builder.add_feature(
            FeatureType.DIFFICULTY,
            shape=(-1,),
            dtype="object",
            description="Scene difficulty metrics",
        )
        
        # =====================================================================
        # Physical Constraints
        # =====================================================================
        builder.set_physical_limits(
            max_batch_size=8,  # Heavier data
            memory_footprint_mb=200.0,
        )
        
        # =====================================================================
        # Semantic Info
        # =====================================================================
        builder.set_semantic_info(
            num_cameras=8,
            bev_channels=12 if self.extract_labels else 0,
            agent_state_dim=7,
            history_length=history_len,
            has_acceleration=True,
            has_nearby_agents=True,
            has_vector_maps=self.extract_vector_maps,
        )
        
        return builder.build()
    
    def _init_scene_loader(self):
        """Initialize NAVSIM scene loader with all sensors."""
        sensor_config = SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=True
        )
        
        num_history = int(
            self.trajectory_sampling.time_horizon / 
            self.trajectory_sampling.interval_length
        )
        
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=max(4, num_history // 4),
            num_future_frames=self.trajectory_sampling.num_poses,
        )
        
        self.scene_loader = SceneLoader(
            data_path=self.data_root / 'mini_navsim_logs' / self.data_split,
            original_sensor_path=self.data_root / 'mini_sensor_blobs' / self.data_split,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
    
    def _init_extractors(self):
        """Initialize feature extractors."""
        # BEV Labels
        if self.extract_labels and BEVLabelExtractor:
            self.label_extractor = BEVLabelExtractor(
                self.bev_size, self.bev_range,
                map_root=self.map_root,
                map_version=self.map_version
            )
        else:
            self.label_extractor = None
        
        # Scenario builder
        if NavsimScenarioBuilder:
            self.scenario_builder = NavsimScenarioBuilder(
                map_root=self.map_root,
                map_version=self.map_version
            )
        else:
            self.scenario_builder = None
        
        # Route extractor
        if self.extract_route_info and RouteExtractor:
            self.route_extractor = RouteExtractor()
        else:
            self.route_extractor = None
        
        # Vector map extractor
        if self.extract_vector_maps and VectorMapExtractor:
            self.vector_map_extractor = VectorMapExtractor(
                map_root=self.map_root,
                map_version=self.map_version,
                max_points_per_lane=20,
                feature_dim=16,
                max_crosswalks=10
            )
            self._map_api_cache = {}
        else:
            self.vector_map_extractor = None

    def __len__(self):
        return len(self.scene_loader.tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with all features."""
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        
        # Build scenario
        if self.scenario_builder:
            scenario = self.scenario_builder.build_scenario(
                scene, self.trajectory_sampling
            )
        else:
            scenario = None
        
        return self._process_scene(scene, scenario, token)
    
    def _process_scene(
        self, 
        scene: Scene, 
        scenario: Optional[NavSimScenario],
        token: str
    ) -> Dict[str, Any]:
        """Process scene to extract all features."""
        
        # current_frame_idx = scenario.database_interval - 1 if scenario else len(scene.frames) // 2
        # current_frame = scene.frames[current_frame_idx]
        # Ensure we get a valid integer index
        if scenario:
            # database_interval is 0.5, so this would be -0.5, rounded to 0
            current_frame_idx = max(0, int(scenario.database_interval - 1))
        else:
            current_frame_idx = len(scene.frames) // 2

        current_frame = scene.frames[current_frame_idx]
        
        # =====================================================================
        # Raw Sensors
        # =====================================================================
        
        # LiDAR
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T
            lidar_original = torch.from_numpy(lidar_pc).float()
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_original = torch.zeros(0, 3)
            lidar_bev = torch.zeros(2, *self.bev_size)
        
        # Cameras (all 8 views)
        camera_images = self._extract_camera_images(scene, scenario)
        
        # =====================================================================
        # BEV Labels
        # =====================================================================
        if self.extract_labels and self.label_extractor:
            try:
                labels = self.label_extractor.extract_all_labels(
                    scene, current_frame_idx
                )
                labels_tensor = {
                    k: torch.from_numpy(v).float() 
                    for k, v in labels.items()
                }
            except Exception as e:
                warnings.warn(f"Label extraction failed: {e}")
                labels_tensor = self._get_empty_labels()
        else:
            labels_tensor = {}
        
        # =====================================================================
        # Agent States (Full 7D)
        # =====================================================================
        if scenario:
            ego_state = scenario.initial_ego_state
            agent_states = torch.tensor([[
                ego_state.rear_axle.x,
                ego_state.rear_axle.y,
                ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
                ego_state.rear_axle.heading,
            ]], dtype=torch.float32)
        else:
            ego = current_frame.ego_status
            # Compute acceleration from velocity
            if current_frame_idx > 0:
                prev_ego = scene.frames[current_frame_idx - 1].ego_status
                dt = 0.1
                ax = (ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
                ay = (ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
            else:
                ax, ay = 0.0, 0.0
            
            agent_states = torch.tensor([[
                ego.ego_pose[0], ego.ego_pose[1],
                ego.ego_velocity[0], ego.ego_velocity[1],
                ax, ay, ego.ego_pose[2]
            ]], dtype=torch.float32)
        
        # =====================================================================
        # Agent History
        # =====================================================================
        if scenario:
            agent_history = self._extract_agent_history_from_scenario(scene, scenario)
        else:
            agent_history = self._extract_agent_history_simple(scene, current_frame_idx)
        
        # =====================================================================
        # Nearby Agents (Multi-agent)
        # =====================================================================
        if scenario:
            nearby_agents = self._extract_nearby_agents(scene, scenario)
        else:
            nearby_agents = torch.zeros(1, 10, 7)
        
        # =====================================================================
        # Ground Truth Trajectory
        # =====================================================================
        if scenario:
            gt_trajectory = self._extract_gt_trajectory(scene, scenario)
        else:
            gt_trajectory = torch.zeros(1, self.trajectory_sampling.num_poses, 5)
        
        # =====================================================================
        # Route Information
        # =====================================================================
        if self.extract_route_info and self.route_extractor and scenario:
            try:
                route_info = self.route_extractor.extract_route_info(scenario)
            except:
                route_info = {}
        else:
            route_info = {}
        
        # =====================================================================
        # Vector Map
        # =====================================================================
        if self.extract_vector_maps and self.vector_map_extractor:
            try:
                ego_pose = current_frame.ego_status.ego_pose
                map_name = scene.scene_metadata.map_name
                
                if map_name not in self._map_api_cache:
                    self._map_api_cache[map_name] = (
                        self.vector_map_extractor.get_map_api(map_name)
                    )
                
                map_api = self._map_api_cache[map_name]
                vector_map_features = self.vector_map_extractor.extract(
                    map_api=map_api,
                    ego_pose=ego_pose,
                    radius=self.bev_range
                )
            except Exception as e:
                warnings.warn(f"Vector map extraction failed: {e}")
                vector_map_features = None
        else:
            vector_map_features = None
        
        # =====================================================================
        # Difficulty Metrics
        # =====================================================================
        if scenario and DifficultyAnalyzer:
            try:
                difficulty = DifficultyAnalyzer.compute_difficulty(
                    scenario, route_info
                )
                difficulty_dict = {
                    'score': difficulty.difficulty_score,
                    'level': difficulty.difficulty_level.value,
                    'num_agents': difficulty.num_agents,
                    'ego_speed': difficulty.ego_speed,
                }
            except:
                difficulty_dict = {}
        else:
            difficulty_dict = {}
        
        # =====================================================================
        # Assemble Sample
        # =====================================================================
        return {
            # Raw sensors
            'lidar_original': lidar_original,
            'lidar_bev': lidar_bev,
            'camera_images': camera_images,
            
            # BEV labels
            'labels': labels_tensor,
            
            # Agent data
            'agent_states': agent_states,
            'agent_history': agent_history,
            'nearby_agents': nearby_agents,
            
            # Ground truth
            'gt_trajectory': gt_trajectory,
            
            # Mission
            'route_info': route_info,
            
            # Map
            'vector_map': vector_map_features,
            
            # Metadata
            'difficulty': difficulty_dict,
            'token': token,
        }
    
    # def _extract_camera_images(self, frame: Frame) -> Dict[str, torch.Tensor]:
    #     """Extract all 8 camera images."""
    #     camera_images = {}
    #     camera_keys = [
    #         ('cam_f0', 'front'),
    #         ('cam_l0', 'front_left'),
    #         ('cam_l1', 'side_left'),
    #         ('cam_l2', 'back_left'),
    #         ('cam_r0', 'front_right'),
    #         ('cam_r1', 'side_right'),
    #         ('cam_r2', 'back_right'),
    #         ('cam_b0', 'back'),
    #     ]
        
    #     for cam_attr, cam_name in camera_keys:
    #         if hasattr(frame, cam_attr):
    #             cam_data = getattr(frame, cam_attr)
    #             if cam_data is not None and hasattr(cam_data, 'image'):
    #                 img = cam_data.image
    #                 if img is not None and isinstance(img, np.ndarray):
    #                     if len(img.shape) == 3 and img.shape[2] == 3:
    #                         img = np.transpose(img, (2, 0, 1))
    #                     camera_images[cam_name] = torch.from_numpy(img).float() / 255.0
        
    #     return camera_images
    def _extract_camera_images(self, scene: Scene, scenario: NavSimScenario) -> List[torch.Tensor]:
        """Extract camera images from current frame."""
        initial_idx = scenario._initial_frame_idx
        current_frame = scene.frames[initial_idx]
        cameras = current_frame.cameras
        
        print(f"\n=== Debug Camera Extraction ===")
        print(f"Initial frame index: {initial_idx}")
        print(f"Total frames in scene: {len(scene.frames)}")
        
        camera_list = []
        camera_names = ['cam_f0', 'cam_l0', 'cam_l1', 'cam_l2', 'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0']
        
        for cam_name in camera_names:
            camera = getattr(cameras, cam_name)
            print(f"{cam_name}: image={'exists' if camera.image is not None else 'None'}")
            
            if camera.image is not None:
                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(camera.image).float() / 255.0
                # Permute from (H, W, C) to (C, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
                camera_list.append(img_tensor)
            else:
                # Don't add empty placeholders - this helps debug
                pass
        
        print(f"Total cameras extracted: {len(camera_list)}")
        return camera_list
        
    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR to BEV."""
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
            height[y_indices[i], x_indices[i]] = max(
                height[y_indices[i], x_indices[i]], z[i]
            )
        
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()
    
    def _extract_agent_history_from_scenario(
        self, 
        scene: Scene, 
        scenario: NavSimScenario
    ) -> torch.Tensor:
        """Extract agent history from scenario."""
        # history_length = int(scenario.database_interval)
        # history_states = []
        
        # for i in range(history_length):
        #     ego = scene.frames[i].ego_status
        #     if i > 0:
        #         prev_ego = scene.frames[i-1].ego_status
        #         dt = 0.1
        #         ax = (ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
        #         ay = (ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
        #     else:
        #         ax, ay = 0.0, 0.0
            
        #     state = torch.tensor([
        #         ego.ego_pose[0], ego.ego_pose[1],
        #         ego.ego_velocity[0], ego.ego_velocity[1],
        #         ax, ay, ego.ego_pose[2]
        #     ], dtype=torch.float32)
        #     history_states.append(state)

        # Get history frames (all frames before and including initial frame)
        num_history_frames = scenario._scene_data.num_history_frames
        initial_idx = scenario._initial_frame_idx

        history_states = []

        # Iterate through history frames
        for i in range(num_history_frames):
            frame_idx = initial_idx - (num_history_frames - 1 - i)  # Calculate actual frame index
            frame = scene.frames[frame_idx]  # Get the frame object
            
            # Now access ego_status from the frame object
            ego = frame.ego_status
            
            # Check if there's a previous frame (use frame_idx, not frame)
            if frame_idx > 0:
                prev_ego = scene.frames[frame_idx - 1].ego_status
                dt = 0.1
                ax = (ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
                ay = (ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
            else:
                ax, ay = 0.0, 0.0
            
            state = torch.tensor([
                ego.ego_pose[0], ego.ego_pose[1],
                ego.ego_velocity[0], ego.ego_velocity[1],
                ax, ay, ego.ego_pose[2]
            ], dtype=torch.float32)
            
            history_states.append(state)

        return torch.stack(history_states).unsqueeze(0)
            
    def _extract_agent_history_simple(
        self, 
        scene: Scene, 
        current_idx: int
    ) -> torch.Tensor:
        """Simple agent history extraction."""
        history_length = min(4, current_idx + 1)
        history_states = []
        
        for i in range(history_length):
            idx = current_idx - history_length + i + 1
            ego = scene.frames[idx].ego_status
            
            if idx > 0:
                prev_ego = scene.frames[idx-1].ego_status
                dt = 0.1
                ax = (ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
                ay = (ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
            else:
                ax, ay = 0.0, 0.0
            
            state = torch.tensor([
                ego.ego_pose[0], ego.ego_pose[1],
                ego.ego_velocity[0], ego.ego_velocity[1],
                ax, ay, ego.ego_pose[2]
            ], dtype=torch.float32)
            history_states.append(state)
        
        return torch.stack(history_states).unsqueeze(0)
    
    def _extract_agent_history_from_scenario(self, scene: Scene, scenario: NavSimScenario):
        """Extract agent history states from scenario."""
        
        num_history_frames = scenario._scene_data.num_history_frames
        initial_idx = scenario._initial_frame_idx
        
        history_states = []
        
        # Iterate through history frames
        for i in range(num_history_frames):
            frame_idx = initial_idx - (num_history_frames - 1 - i)  # Calculate actual frame index
            
            # Get ego state using scenario method
            ego_state = scenario.get_ego_state_at_iteration(frame_idx)
            
            # Extract position from rear_axle
            x = ego_state.rear_axle.x
            y = ego_state.rear_axle.y
            heading = ego_state.rear_axle.heading
            
            # Extract velocity from dynamic_car_state
            vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
            vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
            
            # Extract acceleration from dynamic_car_state
            ax = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
            ay = ego_state.dynamic_car_state.rear_axle_acceleration_2d.y
            
            state = torch.tensor([
                x, y, vx, vy, ax, ay, heading
            ], dtype=torch.float32)
            
            history_states.append(state)
        
        return torch.stack(history_states).unsqueeze(0)
    
    def _extract_nearby_agents(
        self, 
        scene: Scene,
        scenario: NavSimScenario, 
        max_agents: int = 10
    ) -> torch.Tensor:
        """Extract nearby agents from the current frame."""
        
        # Get the current frame (initial frame for planning)
        initial_idx = scenario._initial_frame_idx
        current_frame = scene.frames[initial_idx]
        
        # Get annotations from the frame
        annotations = current_frame.annotations
        
        if annotations is None or len(annotations.boxes) == 0:
            return torch.zeros(1, max_agents, 7)
        
        # Get ego position from scenario
        ego_pos = np.array([
            scenario.initial_ego_state.rear_axle.x,
            scenario.initial_ego_state.rear_axle.y
        ])
        
        agents_data = []
        
        # Iterate through all annotated objects
        for i in range(len(annotations.boxes)):
            # boxes format: [x, y, z, length, width, height, heading]
            # or [x, y, z, width, length, height, heading] depending on convention
            box = annotations.boxes[i]
            
            # Extract position (x, y are typically first two elements)
            agent_pos = np.array([box[0], box[1]])
            distance = np.linalg.norm(agent_pos - ego_pos)
            
            # Extract velocity (vx, vy from velocity_3d)
            velocity = annotations.velocity_3d[i] if annotations.velocity_3d is not None else [0.0, 0.0, 0.0]
            vx = velocity[0]
            vy = velocity[1]
            
            # Extract heading (typically last element in box)
            heading = box[6] if len(box) > 6 else 0.0
            
            # Extract dimensions (width and length)
            # Box format varies, but typically: [x, y, z, length, width, height, heading]
            # or [x, y, z, width, length, height, heading]
            width = box[4] if len(box) > 4 else 2.0
            length = box[3] if len(box) > 3 else 4.5
            
            agents_data.append((distance, [
                box[0],      # x
                box[1],      # y
                vx,          # velocity x
                vy,          # velocity y
                heading,     # heading
                width,       # width
                length,      # length
            ]))
        
        # Sort by distance and take closest agents
        agents_data.sort(key=lambda x: x[0])
        agents_data = agents_data[:max_agents]
        
        # Pad with zeros if needed
        while len(agents_data) < max_agents:
            agents_data.append((0, [0, 0, 0, 0, 0, 0, 0]))
        
        agents_array = np.array([data for _, data in agents_data])
        return torch.from_numpy(agents_array).float().unsqueeze(0)
    
    # def _extract_gt_trajectory(self, scenario: NavSimScenario) -> torch.Tensor:
    #     """Extract ground truth trajectory."""
    #     trajectory = scenario.get_expert_ego_trajectory()
    #     waypoints = []
        
    #     for state in trajectory.trajectory:
    #         waypoints.append([
    #             state.rear_axle.x,
    #             state.rear_axle.y,
    #             state.dynamic_car_state.rear_axle_velocity_2d.x,
    #             state.dynamic_car_state.rear_axle_velocity_2d.y,
    #             state.rear_axle.heading,
    #         ])
        
    #     return torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
    
    # def _extract_gt_trajectory(self, scene: Scene, scenario: NavSimScenario) -> torch.Tensor:
    #     """Extract ground truth trajectory from future frames."""
    #     initial_idx = scenario._initial_frame_idx
    #     num_future_frames = scenario._scene_data.num_future_frames
        
    #     waypoints = []
        
    #     # Get the initial frame's ego pose for coordinate transformation
    #     initial_ego_pose = scene.frames[initial_idx].ego_status.ego_pose
        
    #     # Iterate through future frames
    #     for i in range(1, num_future_frames + 1):
    #         frame_idx = initial_idx + i
    #         if frame_idx >= len(scene.frames):
    #             break
                
    #         frame = scene.frames[frame_idx]
    #         ego_status = frame.ego_status
            
    #         # Transform to local coordinates relative to initial frame
    #         # (assuming ego_pose is in global coordinates)
    #         local_x = ego_status.ego_pose[0] - initial_ego_pose[0]
    #         local_y = ego_status.ego_pose[1] - initial_ego_pose[1]
            
    #         waypoints.append([
    #             local_x,
    #             local_y,
    #             ego_status.ego_velocity[0],
    #             ego_status.ego_velocity[1],
    #             ego_status.ego_pose[2],
    #         ])
        
    #     return torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
    def _extract_gt_trajectory(self, scene: Scene, scenario: NavSimScenario) -> torch.Tensor:
        """Extract ground truth trajectory using scene's built-in method."""
        # Use scene's built-in method - already handles coordinate transformation
        trajectory = scene.get_future_trajectory()
        
        waypoints = []
        dt = trajectory.trajectory_sampling.interval_length  # 0.5 seconds
        
        for i, pose in enumerate(trajectory.poses):
            # Compute velocity from position differences
            if i > 0:
                prev_pose = trajectory.poses[i-1]
                vx = (pose[0] - prev_pose[0]) / dt
                vy = (pose[1] - prev_pose[1]) / dt
            else:
                # First point - could use ego's current velocity or 0
                vx, vy = 0.0, 0.0
            
            waypoints.append([
                pose[0],   # x (already in local coordinates)
                pose[1],   # y (already in local coordinates)
                vx,        # velocity x
                vy,        # velocity y
                pose[2],   # heading (already relative)
            ])
        
        return torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
    
    def _get_empty_labels(self) -> Dict[str, torch.Tensor]:
        """Return empty labels."""
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