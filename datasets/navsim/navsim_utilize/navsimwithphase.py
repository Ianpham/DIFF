"""
Phase NAVSIM Dataset - Complete Integration
==============================================

Combines:
1. Phase-based feature organization (Phase 0/1/2)
2. Complete NAVSIM integration (NavSimScenario, TrajectorySampling, Routes)
3. Difficulty analysis & curriculum learning
4. Vector map extraction

Strategy:
- Phase 0: All IMMEDIATELY AVAILABLE features from EnhancedNavsimDataset
- Phase 1: Pretrained model features (placeholders)
- Phase 2: Custom model features (placeholders)

Author: Phase version combining both approaches
"""

import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
import warnings
from dataclasses import dataclass
from enum import Enum

# NAVSIM imports
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Import shared components from EnhancedNavsimDataset
# NOTE: These should be imported from your actual module
# For now, we'll assume they exist in the same directory
try:
    from navsim_utilize.enhancenavsim import (
        BEVLabelExtractor,
        NavsimScenarioBuilder,
        RouteExtractor,
        DifficultyAnalyzer,
        ScenarioDifficulty,
        DifficultyLevel,
        TrajectoryConfig,
    )
    from navsim_utilize.vectormapfeature import VectorMapExtractor
except ImportError:
    # Fallback - you'll need to include these classes
    print("   Warning: Could not import shared components. Include them in this file.")



# Phase Configuration (from Phase dataset)


@dataclass
class PhaseConfig:
    """Configuration for each feature extraction phase."""
    enabled: bool = False
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    force_recompute: bool = False


class ExtractionPhase(Enum):
    """Feature extraction phases."""
    PHASE_0_CORE = "phase_0_core"              # Available now - ALL EnhancedNavsimDataset features
    PHASE_1_PRETRAINED = "phase_1_pretrained"  # Requires pretrained models
    PHASE_2_CUSTOM = "phase_2_custom"          # Requires custom training



# Phase DATASET - Complete Integration


class PhaseNavsimDataset(Dataset):
    """
    Phase NAVSIM dataset combining phase structure with complete features.
    
    Phase 0 (CORE) now includes:
      All camera images + LiDAR
      12 BEV semantic labels
      NavSimScenario integration
      Route & mission goals
      GT trajectories with TrajectorySampling
      Difficulty metrics
      Vector map features
      Agent states & history
    
    Phase 1 (PRETRAINED) - placeholders for:
    - Weather detection
    - Road surface detection
    - Pedestrian crossing intention
    - Lane change intention
    - Occlusion prediction
    
    Phase 2 (CUSTOM) - placeholders for:
    - Agent behavior classification
    - Intersection context
    - Risk assessment
    """
    
    def __init__(
        self,
        # Basic settings
        data_split: str = "mini",
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: float = 50.0,
        
        # Phase configurations
        enable_phase_0: bool = True,   # Core data (recommended)
        enable_phase_1: bool = False,  # Pretrained models
        enable_phase_2: bool = False,  # Custom models
        
        # Caching
        use_cache: bool = True,
        cache_root: Optional[Path] = None,
        force_recompute: bool = False,
        
        # NAVSIM features (from EnhancedNavsimDataset)
        trajectory_sampling: TrajectorySampling = None,
        difficulty_filter: Optional[DifficultyLevel] = None,
        extract_labels: bool = True,
        extract_route_info: bool = True,
        extract_vector_maps: bool = True,
        
        # Map settings
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        
        # Vector map settings
        max_points_per_lane: int = 20,
        vector_map_feature_dim: int = 16,
        max_crosswalks: int = 10,
    ):
        """Initialize Phase dataset."""
        
        # Paths
        self.data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        self.data_split = data_split
        self.bev_size = bev_size
        self.bev_range = bev_range
        
        # Phase configurations
        self.enable_phase_0 = enable_phase_0
        self.enable_phase_1 = enable_phase_1
        self.enable_phase_2 = enable_phase_2
        
        # Feature extraction flags
        self.extract_labels = extract_labels
        self.extract_route_info = extract_route_info
        self.extract_vector_maps = extract_vector_maps
        self.difficulty_filter = difficulty_filter
        
        # Trajectory sampling
        if trajectory_sampling is None:
            self.trajectory_sampling = TrajectoryConfig.PLANNING_TRAJECTORY_SAMPLING
        else:
            self.trajectory_sampling = trajectory_sampling
        
        # Map setup
        if map_root is None:
            map_root = os.environ.get('NUPLAN_MAPS_ROOT')
            if map_root is None:
                raise ValueError("Map root not specified!")
        self.map_root = map_root
        self.map_version = map_version
        
        # Cache setup
        if cache_root is None:
            cache_root = self.data_root / 'cache' / 'navsim_Phase'
        
        self.cache_root = Path(cache_root)
        self.use_cache = use_cache
        self.force_recompute = force_recompute
        
        # Phase cache directories
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
        
        print("=" * 70)
        print(f"  Phase NAVSIM Dataset - {data_split.upper()}")
        print("=" * 70)
        print(f"Phase 0 (Core):       {'  ENABLED' if enable_phase_0 else '  DISABLED'}")
        print(f"  - Labels:           {' ' if extract_labels else ' '}")
        print(f"  - Routes:           {' ' if extract_route_info else ' '}")
        print(f"  - Vector maps:      {' ' if extract_vector_maps else ' '}")
        print(f"Phase 1 (Pretrained): {'  ENABLED' if enable_phase_1 else '  DISABLED'}")
        print(f"Phase 2 (Custom):     {'  ENABLED' if enable_phase_2 else '  DISABLED'}")
        print(f"Difficulty filter:    {difficulty_filter.value if difficulty_filter else 'None'}")
        print(f"Cache:                {'  ENABLED' if use_cache else '  DISABLED'}")
        print("=" * 70)
        
        # Initialize NAVSIM scene loader
        self._init_scene_loader()
        
        # Initialize extractors
        self._init_extractors()
        
        # Compute difficulties & filter
        self._compute_and_filter_difficulties()
        
        print(f"  Loaded {len(self)} scenes")
        print("=" * 70)
    
    def _init_scene_loader(self):
        """Initialize NAVSIM scene loader."""
        sensor_config = SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=True
        )
        
        num_history = int(self.trajectory_sampling.time_horizon / self.trajectory_sampling.interval_length)
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=max(4, num_history // 4),
            num_future_frames=self.trajectory_sampling.num_poses,
        )
        
        self.scene_loader = SceneLoader(
            data_path=self.data_root / f'mini_navsim_logs' / self.data_split,
            original_sensor_path=self.data_root / f'mini_sensor_blobs' / self.data_split,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
    
    def _init_extractors(self):
        """Initialize all feature extractors."""
        
        # Phase 0 extractors (always if enabled)
        if self.enable_phase_0:
            # BEV Labels
            if self.extract_labels:
                self.label_extractor = BEVLabelExtractor(
                    self.bev_size, self.bev_range, 
                    map_root=self.map_root, 
                    map_version=self.map_version
                )
            else:
                self.label_extractor = None
            
            # Scenario builder
            self.scenario_builder = NavsimScenarioBuilder(
                map_root=self.map_root,
                map_version=self.map_version
            )
            
            # Route extractor
            if self.extract_route_info:
                self.route_extractor = RouteExtractor()
            else:
                self.route_extractor = None
            
            # Vector map extractor
            if self.extract_vector_maps:
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
        
        # Phase 1 & 2 extractors (lazy loading)
        self._phase_1_extractor = None
        self._phase_2_extractor = None
    
    def _compute_and_filter_difficulties(self):
        """Compute difficulties and apply filtering."""
        all_tokens = self.scene_loader.tokens
        
        if self.enable_phase_0:
            print("\n  Computing scenario difficulties...")
            self.difficulty_scores = {}
            
            for token in all_tokens:
                scene = self.scene_loader.get_scene_from_token(token)
                
                try:
                    scenario = self.scenario_builder.build_scenario(scene, self.trajectory_sampling)
                    route_info = self.route_extractor.extract_route_info(scenario) if self.extract_route_info else {}
                    difficulty = DifficultyAnalyzer.compute_difficulty(scenario, route_info)
                    self.difficulty_scores[token] = difficulty
                except Exception as e:
                    warnings.warn(f"Difficulty computation failed for {token}: {e}")
                    # Default difficulty
                    self.difficulty_scores[token] = ScenarioDifficulty(
                        num_agents=0, ego_speed=0.0, ego_acceleration=0.0,
                        min_distance_to_agents=50.0, road_curvature=0.0,
                        traffic_density=0.0, has_intersection=False,
                        has_lane_change=False
                    )
            
            # Apply difficulty filter
            if self.difficulty_filter is not None:
                filtered_tokens = [
                    token for token in all_tokens
                    if self.difficulty_scores[token].difficulty_level == self.difficulty_filter
                ]
                self.scene_tokens = filtered_tokens
                print(f"  Filtered to {len(filtered_tokens)} {self.difficulty_filter.value} scenes")
            else:
                self.scene_tokens = all_tokens
            
            # Print stats
            self._print_difficulty_stats()
        else:
            # No difficulty computation
            self.scene_tokens = all_tokens
            self.difficulty_scores = {}
    
    def _print_difficulty_stats(self):
        """Print difficulty distribution."""
        if not self.difficulty_scores:
            return
        
        levels = {level: 0 for level in DifficultyLevel}
        scores = []
        
        for diff in self.difficulty_scores.values():
            levels[diff.difficulty_level] += 1
            scores.append(diff.difficulty_score)
        
        print("\n  Difficulty Distribution:")
        for level, count in levels.items():
            pct = count / len(self.difficulty_scores) * 100
            print(f"  {level.value:8s}: {count:4d} ({pct:5.1f}%)")
        
        if scores:
            print(f"\n  Avg score: {np.mean(scores):.3f}")
            print(f"  Std score: {np.std(scores):.3f}")
    
    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample with all enabled phases."""
        token = self.scene_tokens[idx]
        
        # Initialize sample
        sample = {'metadata': {'token': token, 'scene_split': self.data_split}}
        
        # Phase 0: Core data (ALL EnhancedNavsimDataset features)
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
    
    # =========================================================================
    # Phase 0: Complete Core Features (from EnhancedNavsimDataset)
    # =========================================================================
    
    def _get_phase_0(self, token: str) -> Dict:
        """
        Extract Phase 0 features - ALL AVAILABLE features from EnhancedNavsimDataset.
        
          CHANGE FROM ORIGINAL PHASE DATASET:
        - Now includes NavSimScenario, routes, difficulty, vector maps
        - No longer placeholders - all features are extracted
        """
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_0_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load scene
        scene = self.scene_loader.get_scene_from_token(token)
        
        # Build scenario
        scenario = self.scenario_builder.build_scenario(scene, self.trajectory_sampling)
        
        # Extract route info
        if self.extract_route_info and self.route_extractor:
            route_info = self.route_extractor.extract_route_info(scenario)
        else:
            route_info = {}
        
        # Current frame
        current_frame_idx = scenario.database_interval - 1
        current_frame = scene.frames[current_frame_idx]
        
        # Camera images (all 8 cameras)
        camera_images = self._extract_camera_images(current_frame)
        
        # LiDAR data
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T
            lidar_original = torch.from_numpy(lidar_pc).float()
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_original = torch.zeros(0, 3)
            lidar_bev = torch.zeros(2, *self.bev_size)
        
        camera_bev = lidar_bev.repeat(32, 1, 1)
        
        # BEV Labels (12 channels)
        if self.extract_labels and self.label_extractor:
            try:
                labels = self.label_extractor.extract_all_labels(scene, current_frame_idx)
                labels_tensor = {k: torch.from_numpy(v).float() for k, v in labels.items()}
            except Exception as e:
                warnings.warn(f"Label extraction failed: {e}")
                labels_tensor = self._get_empty_labels_tensor()
        else:
            labels_tensor = {}
        
        # GT Trajectory (using TrajectorySampling)
        gt_trajectory = self._extract_gt_trajectory(scenario)
        
        # Agent states & history
        ego_state = scenario.initial_ego_state
        agent_states = torch.tensor([[
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            ego_state.rear_axle.heading,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ]], dtype=torch.float32)
        
        agent_history = self._extract_agent_history(scene, scenario)
        nearby_agents = self._extract_nearby_agents(scenario)
        
        # Vector map features
        vector_map_features = None
        if self.extract_vector_maps and self.vector_map_extractor:
            try:
                ego_pose = current_frame.ego_status.ego_pose
                map_name = scene.scene_metadata.map_name
                
                if map_name not in self._map_api_cache:
                    self._map_api_cache[map_name] = self.vector_map_extractor.get_map_api(map_name)
                map_api = self._map_api_cache[map_name]
                
                vector_map_features = self.vector_map_extractor.extract(
                    map_api=map_api,
                    ego_pose=ego_pose,
                    radius=self.bev_range
                )
            except Exception as e:
                warnings.warn(f"Vector map extraction failed: {e}")
                vector_map_features = None
        
        # Difficulty metrics
        difficulty = self.difficulty_scores.get(token)
        
        # Assemble Phase 0 data
        phase_0_data = {
            # ========== RAW SENSOR DATA ==========
            'camera_images': camera_images,
            'lidar_original': lidar_original,
            'camera_bev': camera_bev,
            'lidar_bev': lidar_bev,
            
            # ========== BEV LABELS (12 channels) ==========
            'labels': labels_tensor,
            
            # ========== AGENT STATES ==========
            'agent_states': agent_states,
            'agent_history': agent_history,
            'nearby_agents': nearby_agents,
            
            # ========== GROUND TRUTH ==========
            'gt_trajectory': gt_trajectory,
            
            # ========== ROUTE & MISSION ==========
            'route_info': route_info,
            
            # ========== VECTOR MAP ==========
            'vector_map': vector_map_features,
            
            # ========== DIFFICULTY METRICS ==========
            'difficulty': {
                'score': difficulty.difficulty_score if difficulty else 0.0,
                'level': difficulty.difficulty_level.value if difficulty else 'unknown',
                'num_agents': difficulty.num_agents if difficulty else 0,
                'ego_speed': difficulty.ego_speed if difficulty else 0.0,
                'min_distance': difficulty.min_distance_to_agents if difficulty else 50.0,
                'traffic_density': difficulty.traffic_density if difficulty else 0.0,
            } if difficulty else {},
            
            # ========== METADATA ==========
            'scene_metadata': {
                'log_name': scene.scene_metadata.log_name,
                'map_name': scene.scene_metadata.map_name,
                'timestamp': current_frame.timestamp,
            },
        }
        
        # Cache
        if self.use_cache:
            cache_file = self.phase_0_cache / f'{token}.pt'
            torch.save(phase_0_data, cache_file)
        
        return phase_0_data
    
    # =========================================================================
    # Phase 1 & 2: Unchanged (Placeholders)
    # =========================================================================
    
    def _get_phase_1(self, token: str) -> Dict:
        """Phase 1 features (pretrained models) - UNCHANGED from original."""
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_1_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # PLACEHOLDER: Same as original phase dataset
        phase_1_data = {
            'environmental': {
                'weather_condition': 'clear',
                'weather_confidence': 0.0,
                'weather_intensity': 0.0,
                'visibility_factor': 1.0,
                'road_surface_condition': 'dry',
                'road_surface_confidence': 0.0,
                'friction_coefficient': 1.0,
                'braking_distance_multiplier': 1.0,
                'occlusion_map': torch.zeros(16, *self.bev_size),
                'visibility_map': torch.ones(16, *self.bev_size),
                'occluded_regions': torch.zeros(16, *self.bev_size),
            },
            'behavioral_pretrained': {
                'pedestrian_crossing_intentions': torch.zeros(0),
                'pedestrian_positions': torch.zeros(0, 2),
                'pedestrian_velocities': torch.zeros(0, 2),
                'vehicle_lane_change_intentions': torch.zeros(0, 3),
                'vehicle_lane_change_confidence': torch.zeros(0),
                'vehicle_predicted_trajectories': torch.zeros(0, 16, 2),
            },
        }
        
        # Cache
        if self.use_cache:
            cache_file = self.phase_1_cache / f'{token}.pt'
            torch.save(phase_1_data, cache_file)
        
        return phase_1_data
    
    def _get_phase_2(self, token: str) -> Dict:
        """Phase 2 features (custom models) - UNCHANGED from original."""
        
        # Check cache
        if self.use_cache and not self.force_recompute:
            cache_file = self.phase_2_cache / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # PLACEHOLDER: Same as original phase dataset
        phase_2_data = {
            'behavioral_custom': {
                'agent_behaviors': torch.zeros(0),
                'agent_behavior_names': [],
                'agent_aggressiveness': torch.zeros(0),
                'agent_following_behavior': torch.zeros(0),
                'agent_lane_keeping_score': torch.zeros(0),
            },
            'contextual': {
                'is_in_intersection': False,
                'intersection_type': None,
                'turn_intention': 'straight',
                'approaching_lanes': 0,
                'time_to_intersection': float('inf'),
                'traffic_density': 0.0,
                'local_traffic_flow': torch.zeros(2),
                'min_ttc': float('inf'),
                'collision_risk': 0.0,
            },
            'safety': {
                'overall_safety_score': 1.0,
                'recommended_speed_multiplier': 1.0,
                'increased_following_distance': 1.0,
                'emergency_brake_probability': 0.0,
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
    
    def _extract_camera_images(self, frame: Frame) -> Dict[str, torch.Tensor]:
        """Extract all camera images."""
        camera_images = {}
        camera_keys = [
            ('cam_f0', 'front'), ('cam_l0', 'front_left'),
            ('cam_l1', 'side_left'), ('cam_l2', 'back_left'),
            ('cam_r0', 'front_right'), ('cam_r1', 'side_right'),
            ('cam_r2', 'back_right'), ('cam_b0', 'back'),
        ]
        
        for cam_attr, cam_name in camera_keys:
            if hasattr(frame, cam_attr):
                cam_data = getattr(frame, cam_attr)
                if cam_data is not None and hasattr(cam_data, 'image'):
                    img = cam_data.image
                    if img is not None and isinstance(img, np.ndarray):
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            img = np.transpose(img, (2, 0, 1))
                            camera_images[cam_name] = torch.from_numpy(img).float() / 255.0
        
        return camera_images
    
    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR to BEV (density + height)."""
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
    
    def _extract_gt_trajectory(self, scenario: NavSimScenario) -> torch.Tensor:
        """Extract GT trajectory."""
        trajectory = scenario.get_expert_trajectory()
        waypoints = []
        for state in trajectory.trajectory:
            waypoints.append([
                state.rear_axle.x, state.rear_axle.y,
                state.dynamic_car_state.rear_axle_velocity_2d.x,
                state.dynamic_car_state.rear_axle_velocity_2d.y,
                state.rear_axle.heading,
            ])
        return torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
    
    def _extract_agent_history(self, scene: Scene, scenario: NavSimScenario) -> torch.Tensor:
        """Extract agent history."""
        history_length = scenario.database_interval
        history_states = []
        
        for i in range(history_length):
            ego = scene.frames[i].ego_status
            state = torch.tensor([
                ego.ego_pose[0], ego.ego_pose[1],
                ego.ego_velocity[0], ego.ego_velocity[1],
                ego.ego_pose[2],
                ego.ego_acceleration[0], ego.ego_acceleration[1]
            ], dtype=torch.float32)
            history_states.append(state)
        
        return torch.stack(history_states).unsqueeze(0)
    
    def _extract_nearby_agents(self, scenario: NavSimScenario, max_agents: int = 10) -> torch.Tensor:
        """Extract nearby agents."""
        observation = scenario.get_ego_state_at_iteration(0)
        detections = observation.observation.detections
        
        if detections is None or len(detections) == 0:
            return torch.zeros(1, max_agents, 7)
        
        ego_pos = np.array([
            scenario.initial_ego_state.rear_axle.x,
            scenario.initial_ego_state.rear_axle.y
        ])
        
        agents_data = []
        for detection in detections:
            agent_pos = np.array([detection.center.x, detection.center.y])
            distance = np.linalg.norm(agent_pos - ego_pos)
            
            agents_data.append((distance, [
                detection.center.x, detection.center.y,
                detection.velocity.x if hasattr(detection, 'velocity') else 0.0,
                detection.velocity.y if hasattr(detection, 'velocity') else 0.0,
                detection.center.heading,
                detection.box.width,
                detection.box.length,
            ]))
        
        agents_data.sort(key=lambda x: x[0])
        agents_data = agents_data[:max_agents]
        
        while len(agents_data) < max_agents:
            agents_data.append((0, [0, 0, 0, 0, 0, 0, 0]))
        
        agents_array = np.array([data for _, data in agents_data])
        return torch.from_numpy(agents_array).float().unsqueeze(0)
    
    def _get_empty_labels_tensor(self) -> Dict[str, torch.Tensor]:
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



# Collate Function (combines both approaches)


def Phase_collate_fn(batch):
    """Collate function for Phase dataset."""
    collated = {'metadata': [item['metadata'] for item in batch]}
    
    # Collate Phase 0 (if present)
    if 'phase_0' in batch[0]:
        collated['phase_0'] = _collate_phase_0([item['phase_0'] for item in batch])
    
    # Collate Phase 1 (if present)
    if 'phase_1' in batch[0]:
        collated['phase_1'] = _collate_phase_1([item['phase_1'] for item in batch])
    
    # Collate Phase 2 (if present)
    if 'phase_2' in batch[0]:
        collated['phase_2'] = _collate_phase_2([item['phase_2'] for item in batch])
    
    return collated


def _collate_phase_0(batch):
    """Collate Phase 0 data."""
    # Camera images
    camera_images = {}
    if batch[0]['camera_images']:
        for cam_name in batch[0]['camera_images'].keys():
            camera_images[cam_name] = torch.stack([item['camera_images'][cam_name] for item in batch])
    
    # LiDAR (list - variable size)
    lidar_original = [item['lidar_original'] for item in batch]
    
    # BEV data
    camera_bev = torch.stack([item['camera_bev'] for item in batch])
    lidar_bev = torch.stack([item['lidar_bev'] for item in batch])
    
    # Labels
    labels = {}
    if batch[0]['labels']:
        for key in batch[0]['labels'].keys():
            labels[key] = torch.stack([item['labels'][key] for item in batch])
    
    # Vector maps
    vector_map = None
    if batch[0]['vector_map'] is not None:
        # Use vector map collation from enhanced dataset
        try:
            from navsim_utilize.vectormapfeature import pad_and_stack, pad_and_stack_2d, pad_and_stack_2d_mixed
            
            valid_features = [item['vector_map'] for item in batch if item['vector_map'] is not None]
            
            if len(valid_features) > 0:
                max_lanes = max(f.num_lanes for f in valid_features)
                max_crosswalks = max(f.num_crosswalks for f in valid_features)
                
                # Pad to match batch size
                while len(valid_features) < len(batch):
                    valid_features.append(valid_features[0])
                
                vector_map = {
                    'lane_polylines': pad_and_stack([f.lane_polylines for f in valid_features], max_lanes),
                    'lane_features': pad_and_stack([f.lane_features for f in valid_features], max_lanes),
                    'lane_masks': pad_and_stack([f.lane_masks for f in valid_features], max_lanes),
                    'connectivity': pad_and_stack_2d([f.connectivity_matrix for f in valid_features], max_lanes),
                    # Add other vector map features as needed
                }
        except Exception as e:
            warnings.warn(f"Vector map collation failed: {e}")
            vector_map = None
    
    return {
        'camera_images': camera_images,
        'lidar_original': lidar_original,
        'camera_bev': camera_bev,
        'lidar_bev': lidar_bev,
        'labels': labels,
        'agent_states': torch.cat([item['agent_states'] for item in batch], dim=0),
        'agent_history': torch.cat([item['agent_history'] for item in batch], dim=0),
        'gt_trajectory': torch.cat([item['gt_trajectory'] for item in batch], dim=0),
        'nearby_agents': torch.cat([item['nearby_agents'] for item in batch], dim=0),
        'route_info': [item['route_info'] for item in batch],
        'vector_map': vector_map,
        'difficulty': [item['difficulty'] for item in batch],
        'scene_metadata': [item['scene_metadata'] for item in batch],
    }


def _collate_phase_1(batch):
    """Collate Phase 1 data (placeholders)."""
    return {
        'environmental': {
            'weather_condition': [item['environmental']['weather_condition'] for item in batch],
            'weather_confidence': torch.tensor([item['environmental']['weather_confidence'] for item in batch]),
            # ... other environmental features
        },
        'behavioral_pretrained': {
            k: [item['behavioral_pretrained'][k] for item in batch]
            for k in batch[0]['behavioral_pretrained'].keys()
        },
    }


def _collate_phase_2(batch):
    """Collate Phase 2 data (placeholders)."""
    return {
        'behavioral_custom': {
            k: [item['behavioral_custom'][k] for item in batch]
            for k in batch[0]['behavioral_custom'].keys()
        },
        'contextual': {
            k: [item['contextual'][k] for item in batch]
            for k in batch[0]['contextual'].keys()
        },
        'safety': {
            k: torch.tensor([item['safety'][k] for item in batch])
            for k in batch[0]['safety'].keys()
        },
    }



# Main / Testing


if __name__ == "__main__":
    print("=" * 70)
    print("  Phase NAVSIM Dataset Test")
    print("=" * 70)
    
    dataset = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=False,
        enable_phase_2=False,
        difficulty_filter=None,
        extract_labels=True,
        extract_route_info=True,
        extract_vector_maps=True,
        use_cache=False,
    )
    
    print(f"\n  Created dataset: {len(dataset)} scenes")
    
    # Get sample
    sample = dataset[0]
    
    print("\n📦 Sample structure:")
    print(f"  Metadata: {list(sample['metadata'].keys())}")
    
    if 'phase_0' in sample:
        print(f"\n  Phase 0 keys: {list(sample['phase_0'].keys())}")
        print(f"    Camera images: {len(sample['phase_0']['camera_images'])} cameras")
        print(f"    Labels: {len(sample['phase_0']['labels'])} channels")
        print(f"    GT trajectory: {sample['phase_0']['gt_trajectory'].shape}")
        print(f"    Vector map: {sample['phase_0']['vector_map'] is not None}")
        print(f"    Difficulty: {sample['phase_0']['difficulty']}")
    
    print("\n  Test complete!")