"""
PhaseNavsimDataset - Multi-Phase Training Dataset
==================================================
Supports progressive feature extraction across 3 phases:

Phase 0 (Core): All immediately available features
  - All sensors (LiDAR, cameras)
  - BEV labels
  - Agent states & history
  - Vector maps
  - Route info

Phase 1 (Pretrained): Features from pretrained models
  - Weather detection
  - Road surface analysis
  - Occlusion prediction
  - Behavioral models

Phase 2 (Custom): Features from custom models
  - Agent behavior classification
  - Intersection context
  - Risk assessment

Curriculum learning: Start with Phase 0, progressively add Phase 1 & 2.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import warnings
from enum import Enum

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .base import BaseNavsimDataset
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType

# Import utilities
try:
    from navsim_utilize.enhancenavsim import (
        BEVLabelExtractor,
        NavsimScenarioBuilder,
        RouteExtractor,
        DifficultyAnalyzer,
        DifficultyLevel,
        TrajectoryConfig,
    )
    from navsim_utilize.vectormapfeature import VectorMapExtractor
except ImportError:
    warnings.warn("Could not import navsim utilities")
    BEVLabelExtractor = None
    NavsimScenarioBuilder = None
    RouteExtractor = None
    DifficultyAnalyzer = None
    DifficultyLevel = None
    VectorMapExtractor = None

# =============================================================================
# Phase Configuration
# =============================================================================

class ExtractionPhase(Enum):
    """Feature extraction phases for curriculum learning."""
    PHASE_0_CORE = "phase_0_core"              # Available now
    PHASE_1_PRETRAINED = "phase_1_pretrained"  # Requires pretrained models
    PHASE_2_CUSTOM = "phase_2_custom"          # Requires custom training

# =============================================================================
# PhaseNavsimDataset
# =============================================================================

class PhaseNavsimDataset(BaseNavsimDataset):
    """
    Multi-phase NAVSIM dataset for curriculum learning.
    
    Supports progressive feature extraction:
    - Phase 0: Core features (sensors, maps, agents)
    - Phase 1: Pretrained model features (placeholders)
    - Phase 2: Custom model features (placeholders)
    
    Physical constraints:
    - Max batch size: 4 (most comprehensive)
    - Memory: ~250 MB/sample (all phases)
    """
    
    def __init__(
        self,
        # Basic settings
        data_split: str = "mini",
        bev_size: tuple = (200, 200),
        bev_range: float = 50.0,
        
        # Phase configurations
        enable_phase_0: bool = True,
        enable_phase_1: bool = False,
        enable_phase_2: bool = False,
        
        # Caching
        use_cache: bool = True,
        cache_root: Optional[Path] = None,
        force_recompute: bool = False,
        
        # Phase 0 features
        trajectory_sampling: Optional[TrajectorySampling] = None,
        difficulty_filter: Optional['DifficultyLevel'] = None,
        extract_labels: bool = True,
        extract_route_info: bool = True,
        extract_vector_maps: bool = True,
        
        # Map settings
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        
        # Sensor configuration
        sensor_config: SensorConfig = None,
    ):
        super().__init__()
        
        self.data_split = data_split
        self.bev_size = bev_size
        self.bev_range = bev_range
        
        # Phase enables
        self.enable_phase_0 = enable_phase_0
        self.enable_phase_1 = enable_phase_1
        self.enable_phase_2 = enable_phase_2
        
        # Feature extraction flags
        self.extract_labels = extract_labels
        self.extract_route_info = extract_route_info
        self.extract_vector_maps = extract_vector_maps
        self.difficulty_filter = difficulty_filter
        
        # Caching
        self.use_cache = use_cache
        self.force_recompute = force_recompute
        
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
        
        # Sensor configuration
        if sensor_config is None:
            sensor_config = SensorConfig.build_all_sensors(include=True)
        self.sensor_config = sensor_config
        
        # Cache setup
        if cache_root is None:
            cache_root = self.data_root / 'cache' / 'navsim_phase'
        self.cache_root = Path(cache_root)
        
        difficulty_suffix = f"_{difficulty_filter.value}" if difficulty_filter else ""
        self.phase_0_cache = self.cache_root / 'phase_0_core' / f"{data_split}{difficulty_suffix}"
        self.phase_1_cache = self.cache_root / 'phase_1_pretrained' / f"{data_split}{difficulty_suffix}"
        self.phase_2_cache = self.cache_root / 'phase_2_custom' / f"{data_split}{difficulty_suffix}"
        
        if use_cache:
            self.phase_0_cache.mkdir(parents=True, exist_ok=True)
            if enable_phase_1:
                self.phase_1_cache.mkdir(parents=True, exist_ok=True)
            if enable_phase_2:
                self.phase_2_cache.mkdir(parents=True, exist_ok=True)
        
        # Initialize
        self._init_scene_loader()
        self._init_extractors()
        self._compute_and_filter_difficulties()
        
        print(f"✓ Initialized {self.__class__.__name__}: {len(self)} scenes")
        print(f"  Phase 0: {'✓' if enable_phase_0 else '✗'}")
        print(f"  Phase 1: {'✓' if enable_phase_1 else '✗'}")
        print(f"  Phase 2: {'✓' if enable_phase_2 else '✗'}")
        print(f"  Dataset initialized with sensor_config: {self.sensor_config}")
    
    def _build_contract(self) -> DataContract:
        """Declare what PhaseNavsimDataset provides."""
        
        builder = ContractBuilder(dataset_name="PhaseNavsimDataset")
        
        # =====================================================================
        # Phase 0 Features (if enabled)
        # =====================================================================
        if self.enable_phase_0:
            # Raw sensors
            builder.add_feature(
                FeatureType.LIDAR_POINTS,
                shape=(-1, 3),
                dtype="float32",
                description="Raw LiDAR point clouds (Phase 0)",
            )
            
            builder.add_feature(
                FeatureType.LIDAR_BEV,
                shape=(2, *self.bev_size),
                dtype="float32",
                description="Rasterized LiDAR BEV (Phase 0)",
            )
            
            builder.add_feature(
                FeatureType.CAMERA_IMAGES,
                shape=(8, 3, 900, 1600),
                dtype="float32",
                description="All 8 camera views (Phase 0)",
            )
            
            # BEV labels
            if self.extract_labels:
                builder.add_feature(
                    FeatureType.BEV_LABELS,
                    shape=(12, *self.bev_size),
                    dtype="float32",
                    description="HD map semantic labels (Phase 0)",
                )
            
            # Vector maps
            if self.extract_vector_maps:
                builder.add_feature(
                    FeatureType.VECTOR_MAP,
                    shape=(-1,),
                    dtype="object",
                    description="Structured vector map (Phase 0)",
                )
            
            # Agent features
            history_len = int(
                self.trajectory_sampling.time_horizon / 
                self.trajectory_sampling.interval_length
            )
            
            builder.add_feature(
                FeatureType.AGENT_STATE,
                shape=(1, 7),
                dtype="float32",
                description="Agent state with acceleration (Phase 0)",
            )
            
            builder.add_feature(
                FeatureType.AGENT_HISTORY,
                shape=(1, history_len, 7),
                dtype="float32",
                description="Agent trajectory history (Phase 0)",
            )
            
            builder.add_feature(
                FeatureType.AGENT_NEARBY,
                shape=(1, 10, 7),
                dtype="float32",
                description="Nearby agents (Phase 0)",
            )
            
            # Ground truth
            builder.add_feature(
                FeatureType.GT_TRAJECTORY,
                shape=(1, self.trajectory_sampling.num_poses, 5),
                dtype="float32",
                description="Ground truth trajectory (Phase 0)",
            )
            
            # Route
            if self.extract_route_info:
                builder.add_feature(
                    FeatureType.ROUTE,
                    shape=(-1,),
                    dtype="object",
                    description="Navigation route (Phase 0)",
                )
            
            # Difficulty
            builder.add_feature(
                FeatureType.DIFFICULTY,
                shape=(-1,),
                dtype="object",
                description="Scene difficulty metrics (Phase 0)",
            )
        
        # =====================================================================
        # Phase 1 & 2 Features (placeholders - not in contract yet)
        # =====================================================================
        # These will be added when pretrained/custom models are ready
        # For now, they're just in the sample dict but not in contract
        
        # =====================================================================
        # Physical Constraints
        # =====================================================================
        memory_mb = 250.0 if (self.enable_phase_1 or self.enable_phase_2) else 200.0
        
        builder.set_physical_limits(
            max_batch_size=4,  # Most comprehensive dataset
            memory_footprint_mb=memory_mb,
        )
        
        # =====================================================================
        # Semantic Info
        # =====================================================================
        builder.set_semantic_info(
            num_cameras=8 if self.enable_phase_0 else 0,
            bev_channels=12 if (self.enable_phase_0 and self.extract_labels) else 0,
            agent_state_dim=7 if self.enable_phase_0 else 5,
            history_length=history_len if self.enable_phase_0 else 4,
            has_acceleration=self.enable_phase_0,
            has_nearby_agents=self.enable_phase_0,
            has_vector_maps=self.enable_phase_0 and self.extract_vector_maps,
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
        if self.enable_phase_0:
            # BEV labels
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
        else:
            self.label_extractor = None
            self.scenario_builder = None
            self.route_extractor = None
            self.vector_map_extractor = None
    
    def _compute_and_filter_difficulties(self):
        """Compute difficulties and apply filtering."""
        all_tokens = self.scene_loader.tokens
        
        if self.enable_phase_0 and DifficultyAnalyzer:
            self.difficulty_scores = {}
            
            # Create default difficulty class
            from dataclasses import dataclass
            from typing import Any
            @dataclass
            class DefaultDifficulty:
                difficulty_score: float = 0.0
                difficulty_level: Any = None
                num_agents: int = 0
                ego_speed: float = 0.0
            
            for token in all_tokens:
                try:
                    scene = self.scene_loader.get_scene_from_token(token)
                    scenario = self.scenario_builder.build_scenario(
                        scene, self.trajectory_sampling
                    )
                    route_info = (
                        self.route_extractor.extract_route_info(scenario)
                        if self.extract_route_info and self.route_extractor
                        else {}
                    )
                    
                    # Try to compute difficulty, but be prepared for failures
                    try:
                        difficulty = DifficultyAnalyzer.compute_difficulty(
                            scenario, route_info
                        )
                        self.difficulty_scores[token] = difficulty
                    except (AttributeError, KeyError, IndexError) as e:
                        # If difficulty computation fails, compute a simple version
                        # based on available data
                        initial_idx = scenario._initial_frame_idx
                        current_frame = scene.frames[initial_idx]
                        annotations = current_frame.annotations
                        
                        num_agents = len(annotations.boxes) if annotations and annotations.boxes is not None else 0
                        ego_speed = np.linalg.norm([
                            scenario.initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                            scenario.initial_ego_state.dynamic_car_state.rear_axle_velocity_2d.y
                        ])
                        
                        # Create simple difficulty
                        simple_diff = DefaultDifficulty(
                            difficulty_score=0.0,
                            difficulty_level=None,
                            num_agents=num_agents,
                            ego_speed=ego_speed
                        )
                        self.difficulty_scores[token] = simple_diff
                        
                except Exception as e:
                    # Complete failure - use default
                    self.difficulty_scores[token] = DefaultDifficulty()
            
            # Apply filter
            if self.difficulty_filter is not None:
                filtered_tokens = [
                    token for token in all_tokens
                    if (self.difficulty_scores[token].difficulty_level == self.difficulty_filter
                        if self.difficulty_scores[token].difficulty_level is not None
                        else False)
                ]
                self.scene_tokens = filtered_tokens
                if len(filtered_tokens) > 0:
                    print(f"  Filtered to {len(filtered_tokens)} {self.difficulty_filter.value} scenes")
                else:
                    print(f"  Warning: No scenes matched difficulty filter {self.difficulty_filter.value}")
                    print(f"  Using all {len(all_tokens)} scenes instead")
                    self.scene_tokens = all_tokens
            else:
                self.scene_tokens = all_tokens
        else:
            self.scene_tokens = all_tokens
            self.difficulty_scores = {}
    
    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with enabled phases."""
        token = self.scene_tokens[idx]
        
        sample = {'metadata': {'token': token, 'scene_split': self.data_split}}
        
        # Phase 0: Core data
        if self.enable_phase_0:
            phase_0_data = self._get_phase_0(token)
            sample['phase_0'] = phase_0_data
        
        # Phase 1: Pretrained models (placeholders)
        if self.enable_phase_1:
            phase_1_data = self._get_phase_1(token)
            sample['phase_1'] = phase_1_data
        
        # Phase 2: Custom models (placeholders)
        if self.enable_phase_2:
            phase_2_data = self._get_phase_2(token)
            sample['phase_2'] = phase_2_data
        
        return sample
    
    def _get_phase_0(self, token: str) -> Dict[str, Any]:
        """Extract Phase 0 features (all core features)."""
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_0_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load scene
        scene = self.scene_loader.get_scene_from_token(token)
        
        # Build scenario
        scenario = None
        if self.scenario_builder:
            scenario = self.scenario_builder.build_scenario(
                scene, self.trajectory_sampling
            )
        
        # Process scene
        phase_0_data = self._process_scene(scene, scenario, token)
        
        # Cache
        if self.use_cache:
            cache_file = self.phase_0_cache / f'{token}.pt'
            torch.save(phase_0_data, cache_file)
        
        return phase_0_data
    
    def _process_scene(
        self, 
        scene: Scene, 
        scenario: Optional[NavSimScenario],
        token: str
    ) -> Dict[str, Any]:
        """Process scene to extract all Phase 0 features (based on EnhancedNavsimDataset)."""
        
        # Ensure we get a valid integer index
        if scenario:
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
        difficulty = self.difficulty_scores.get(token)
        difficulty_dict = {}
        if difficulty:
            difficulty_dict = {
                'score': difficulty.difficulty_score,
                'level': difficulty.difficulty_level.value if difficulty.difficulty_level else 'unknown',
                'num_agents': difficulty.num_agents,
                'ego_speed': difficulty.ego_speed,
            }
        
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
    
    def _get_phase_1(self, token: str) -> Dict[str, Any]:
        """Get Phase 1 features (pretrained models - placeholders)."""
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_1_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # PLACEHOLDER: Pretrained model features
        phase_1_data = {
            'environmental': {
                'weather_condition': 'clear',
                'weather_confidence': 0.0,
                'visibility_factor': 1.0,
                'road_surface_condition': 'dry',
                'friction_coefficient': 1.0,
                'occlusion_map': torch.zeros(16, *self.bev_size),
            },
            'behavioral_pretrained': {
                'pedestrian_crossing_intentions': torch.zeros(0),
                'vehicle_lane_change_intentions': torch.zeros(0, 3),
            },
        }
        
        # Cache
        if self.use_cache:
            cache_file = self.phase_1_cache / f'{token}.pt'
            torch.save(phase_1_data, cache_file)
        
        return phase_1_data
    
    def _get_phase_2(self, token: str) -> Dict[str, Any]:
        """Get Phase 2 features (custom models - placeholders)."""
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_2_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # PLACEHOLDER: Custom model features
        phase_2_data = {
            'behavioral_custom': {
                'agent_behaviors': torch.zeros(0),
                'agent_aggressiveness': torch.zeros(0),
            },
            'contextual': {
                'is_in_intersection': False,
                'turn_intention': 'straight',
                'traffic_density': 0.0,
            },
            'safety': {
                'overall_safety_score': 1.0,
                'collision_risk': 0.0,
            },
        }
        
        # Cache
        if self.use_cache:
            cache_file = self.phase_2_cache / f'{token}.pt'
            torch.save(phase_2_data, cache_file)
        
        return phase_2_data
    
    # =========================================================================
    # Helper Methods (from EnhancedNavsimDataset)
    # =========================================================================
    
    def _extract_camera_images(self, scene: Scene, scenario: NavSimScenario) -> List[torch.Tensor]:
        """Extract camera images from current frame."""
        initial_idx = scenario._initial_frame_idx if scenario else len(scene.frames) // 2
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
        """Extract agent history states from scenario."""
        
        num_history_frames = scenario._scene_data.num_history_frames
        initial_idx = scenario._initial_frame_idx
        
        history_states = []
        
        # Iterate through history frames
        for i in range(num_history_frames):
            frame_idx = initial_idx - (num_history_frames - 1 - i)
            
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
            box = annotations.boxes[i]
            
            # Extract position
            agent_pos = np.array([box[0], box[1]])
            distance = np.linalg.norm(agent_pos - ego_pos)
            
            # Extract velocity
            velocity = annotations.velocity_3d[i] if annotations.velocity_3d is not None else [0.0, 0.0, 0.0]
            vx = velocity[0]
            vy = velocity[1]
            
            # Extract heading
            heading = box[6] if len(box) > 6 else 0.0
            
            # Extract dimensions
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