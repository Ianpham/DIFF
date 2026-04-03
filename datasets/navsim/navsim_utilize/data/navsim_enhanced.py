"""
EnhancedNavsimDataset - Full-Featured Dataset
==============================================
Provides all available features including:
- Raw LiDAR points + BEV
- All 8 camera images
- 12-channel BEV labels
- Full 7D agent states (with acceleration), ego-frame-relative
- Multi-agent data (nearby agents) with full history
- Group-C context features (intersection, goal, traffic, pedestrian)
- Vector maps (optional)
- Route information (optional)
- Difficulty metrics
"""

import torch
import numpy as np
from pathlib import Path
import os
import warnings
from typing import Dict, List, Optional, Any, Tuple

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import SemanticMapLayer
from shapely.geometry import Point as ShapelyPoint

from data.base import BaseNavsimDataset
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
from datasets.navsim.navsim.common.enums import BoundingBoxIndex

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
    DifficultyAnalyzer = None
    VectorMapExtractor = None


class EnhancedNavsimDataset(BaseNavsimDataset):
    """
    Enhanced NAVSIM dataset with all features.

    Provides:
    - Raw LiDAR point clouds
    - LiDAR BEV
    - 8 camera images (all views)
    - BEV labels (12 channels)
    - Agent state (7D with acceleration), ego-frame-relative
    - Agent history (ego-frame-relative)
    - Multi-agent states + history (32 agents, ego-frame-relative)
    - GT trajectory (ego-frame-relative)
    - Group-C context features (intersection, goal, traffic lights, pedestrians)
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
        max_agents: int = 32,
    ):
        super().__init__()

        self.data_split = data_split
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.extract_labels = extract_labels
        self.extract_route_info = extract_route_info
        self.extract_vector_maps = extract_vector_maps
        self.max_agents = max_agents
        self.history_length = 4
        self.future_horizon = 8

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

        # Initialize components
        self._init_scene_loader()
        self._init_extractors()

        print(f"  Initialized {self.__class__.__name__}: {len(self)} scenes")
        print(f"  Dataset initialized with sensor_config: {self.sensor_config}")

    # =========================================================================
    # Contract
    # =========================================================================

    def _build_contract(self) -> DataContract:
        """Declare what EnhancedNavsimDataset provides."""

        builder = ContractBuilder(dataset_name="EnhancedNavsimDataset")

        builder.add_feature(
            FeatureType.LIDAR_POINTS,
            shape=(-1, 3),
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
            shape=(8, 3, 900, 1600),
            dtype="float32",
            description="All 8 camera views",
        )

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
                shape=(-1,),
                dtype="object",
                description="Structured vector map features",
            )

        history_len = int(
            self.trajectory_sampling.time_horizon /
            self.trajectory_sampling.interval_length
        )

        builder.add_feature(
            FeatureType.AGENT_STATE,
            shape=(1, 7),
            dtype="float32",
            description="Agent state (ego-frame): [0,0, vx,vy, ax,ay, 0]",
        )

        builder.add_feature(
            FeatureType.AGENT_HISTORY,
            shape=(1, history_len, 7),
            dtype="float32",
            description="Agent trajectory history (ego-frame-relative)",
        )

        builder.add_feature(
            FeatureType.AGENT_NEARBY,
            shape=(self.max_agents, 5),
            dtype="float32",
            description="Multi-agent states [x,y,vx,vy,heading] (ego-frame-relative)",
        )

        builder.add_feature(
            FeatureType.GT_TRAJECTORY,
            shape=(1, self.trajectory_sampling.num_poses, 5),
            dtype="float32",
            description="Ground truth future trajectory (ego-frame-relative)",
        )

        if self.extract_route_info:
            builder.add_feature(
                FeatureType.ROUTE,
                shape=(-1,),
                dtype="object",
                description="Navigation route information",
            )

        builder.add_feature(
            FeatureType.DIFFICULTY,
            shape=(-1,),
            dtype="object",
            description="Scene difficulty metrics",
        )

        builder.set_physical_limits(
            max_batch_size=8,
            memory_footprint_mb=200.0,
        )

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

    # =========================================================================
    # Initialisation helpers
    # =========================================================================

    def _init_scene_loader(self):
        """Initialize NAVSIM scene loader with all 8 cameras."""
        sensor_config = SensorConfig(
            cam_f0=True, cam_l0=True, cam_l1=True, cam_l2=True,
            cam_r0=True, cam_r1=True, cam_r2=True, cam_b0=True,
            lidar_pc=True,
        )

        num_history = int(
            self.trajectory_sampling.time_horizon /
            self.trajectory_sampling.interval_length
        )

        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=max(self.history_length, num_history // 4),
            num_future_frames=self.trajectory_sampling.num_poses,
        )

        self.scene_loader = SceneLoader(
            data_path=self.data_root / 'mini_navsim_logs' / self.data_split,
            original_sensor_path=self.data_root / 'mini_sensor_blobs' / self.data_split,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

    def _init_extractors(self):
        """Initialize feature extractors."""
        # BEV Labels
        if self.extract_labels and BEVLabelExtractor:
            self.label_extractor = BEVLabelExtractor(
                self.bev_size, self.bev_range,
                map_root=self.map_root,
                map_version=self.map_version,
            )
        else:
            self.label_extractor = None

        # Scenario builder
        if NavsimScenarioBuilder:
            self.scenario_builder = NavsimScenarioBuilder(
                map_root=self.map_root,
                map_version=self.map_version,
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
                max_crosswalks=10,
            )
            self._map_api_cache = {}
        else:
            self.vector_map_extractor = None

    # =========================================================================
    # Dataset interface
    # =========================================================================

    def __len__(self):
        return len(self.scene_loader.tokens)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        token = self.scene_loader.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)

        scenario = None
        if self.scenario_builder:
            scenario = self.scenario_builder.build_scenario(
                scene, self.trajectory_sampling
            )

        return self._process_scene(scene, scenario, token)

    # =========================================================================
    # Core processing
    # =========================================================================

    def _process_scene(
        self,
        scene: Scene,
        scenario: Optional[NavSimScenario],
        token: str,
    ) -> Dict[str, Any]:
        """
        Process scene — all outputs in current-ego-frame-relative coordinates.

        Ego vehicle sits at origin (0, 0) with heading 0.  All agent positions,
        history, and the GT trajectory are expressed relative to this frame.
        """

        if scenario:
            current_frame_idx = max(0, int(scenario.database_interval - 1))
        else:
            current_frame_idx = len(scene.frames) // 2

        current_frame = scene.frames[current_frame_idx]
        current_ego   = current_frame.ego_status
        ref_x, ref_y, ref_heading = current_ego.ego_pose

        def to_relative(pose_x: float, pose_y: float, pose_heading: float):
            """Convert global pose to current-ego-frame-relative."""
            dx    = pose_x - ref_x
            dy    = pose_y - ref_y
            cos_h = np.cos(-ref_heading)
            sin_h = np.sin(-ref_heading)
            rel_x = cos_h * dx - sin_h * dy
            rel_y = sin_h * dx + cos_h * dy
            rel_h = pose_heading - ref_heading
            return rel_x, rel_y, rel_h

        # Build a NavSimScenario for group-C map queries if needed
        if scenario is not None:
            scenario_ctx = scenario
        else:
            scenario_ctx = NavSimScenario(
                scene,
                map_root=self.map_root,
                map_version=self.map_version,
            )

        # ------------------------------------------------------------------
        # 1. LiDAR
        # ------------------------------------------------------------------
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc       = current_frame.lidar.lidar_pc[:3, :].T
            lidar_original = torch.from_numpy(lidar_pc).float()
            lidar_bev      = self._rasterize_lidar(lidar_pc)
        else:
            lidar_original = torch.zeros(0, 3)
            lidar_bev      = torch.zeros(2, *self.bev_size)

        # ------------------------------------------------------------------
        # 2. Cameras (all 8 views)
        # ------------------------------------------------------------------
        camera_images = self._extract_camera_images(scene, current_frame_idx)

        # ------------------------------------------------------------------
        # 3. BEV labels
        # ------------------------------------------------------------------
        if self.extract_labels and self.label_extractor:
            try:
                labels = self.label_extractor.extract_all_labels(scene, current_frame_idx)
                labels_tensor = {k: torch.from_numpy(v).float() for k, v in labels.items()}
            except Exception as e:
                warnings.warn(f"Label extraction failed for {token}: {e}")
                labels_tensor = self._get_empty_labels()
        else:
            labels_tensor = {}

        # ------------------------------------------------------------------
        # 4. Ego agent state — ego frame: position = (0,0), heading = 0
        # ------------------------------------------------------------------
        if current_frame_idx > 0:
            prev_ego = scene.frames[current_frame_idx - 1].ego_status
            dt = 0.1
            ax = (current_ego.ego_velocity[0] - prev_ego.ego_velocity[0]) / dt
            ay = (current_ego.ego_velocity[1] - prev_ego.ego_velocity[1]) / dt
        else:
            ax, ay = 0.0, 0.0

        agent_states = torch.tensor(
            [[0.0, 0.0,
              current_ego.ego_velocity[0], current_ego.ego_velocity[1],
              ax, ay, 0.0]],
            dtype=torch.float32,
        )  # [1, 7]

        # ------------------------------------------------------------------
        # 5. Ego agent history — relative to current ego frame
        #    Always use self.history_length steps (oldest -> newest).
        #    Frames that fall before scene start are zero-padded.
        # ------------------------------------------------------------------
        history_states = []
        for i in range(self.history_length):
            frame_idx = current_frame_idx - (self.history_length - 1 - i)
            if frame_idx < 0:
                history_states.append(torch.zeros(7))
                continue
            h_ego = scene.frames[frame_idx].ego_status
            rel_x, rel_y, rel_h = to_relative(*h_ego.ego_pose)
            if frame_idx > 0:
                p_ego = scene.frames[frame_idx - 1].ego_status
                h_ax  = (h_ego.ego_velocity[0] - p_ego.ego_velocity[0]) / 0.1
                h_ay  = (h_ego.ego_velocity[1] - p_ego.ego_velocity[1]) / 0.1
            else:
                h_ax, h_ay = 0.0, 0.0
            history_states.append(torch.tensor(
                [rel_x, rel_y,
                 h_ego.ego_velocity[0], h_ego.ego_velocity[1],
                 h_ax, h_ay, rel_h],
                dtype=torch.float32,
            ))
        agent_history = torch.stack(history_states).unsqueeze(0)  # [1, T, 7]

        # ------------------------------------------------------------------
        # 6. Multi-agent states + history (replaces old _extract_nearby_agents)
        # ------------------------------------------------------------------
        multi_agent_states, track_tokens = self._extract_multi_agent_states(
            scene, current_frame_idx, max_agents=self.max_agents,
        )
        multi_agent_history = self._extract_multi_agent_history(
            scene, current_frame_idx, track_tokens,
            max_agents=self.max_agents, num_history=self.history_length,
        )

        # ------------------------------------------------------------------
        # 7. GT trajectory — already ego-relative from scene helper
        # ------------------------------------------------------------------
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=self.trajectory_sampling.num_poses
        )
        poses = future_traj.poses
        waypoints = []
        for i, (x, y, heading) in enumerate(poses):
            if i > 0:
                prev_x, prev_y, _ = poses[i - 1]
                vx = (x - prev_x) / 0.5
                vy = (y - prev_y) / 0.5
            else:
                vx = current_ego.ego_velocity[0]
                vy = current_ego.ego_velocity[1]
            waypoints.append([x, y, vx, vy, heading])
        gt_trajectory = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)

        # ------------------------------------------------------------------
        # 8. Route information
        # ------------------------------------------------------------------
        route_info = {}
        if self.extract_route_info and self.route_extractor and scenario:
            try:
                route_info = self.route_extractor.extract_route_info(scenario)
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 9. Vector map
        # ------------------------------------------------------------------
        vector_map_features = None
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
                    radius=self.bev_range,
                )
            except Exception as e:
                warnings.warn(f"Vector map extraction failed for {token}: {e}")

        # ------------------------------------------------------------------
        # 10. Group-C context features
        # ------------------------------------------------------------------
        intersection_features     = self._extract_intersection_features(
            scene, scenario_ctx, current_frame_idx, max_intersection=4,
        )
        goal_features             = self._extract_goal_features(
            scene, scenario_ctx, current_frame_idx, max_goals=1,
        )
        traffic_control_features  = self._extract_traffic_control_features(
            scene, scenario_ctx, current_frame_idx, max_lights=8,
        )
        pedestrian_features       = self._extract_pedestrian_features(
            scene, scenario_ctx, current_frame_idx,
            max_pedestrians=10, crosswalk_radius=15.0,
        )

        # ------------------------------------------------------------------
        # 11. Difficulty metrics
        # ------------------------------------------------------------------
        difficulty_dict = {}
        if scenario and DifficultyAnalyzer:
            try:
                difficulty = DifficultyAnalyzer.compute_difficulty(scenario, route_info)
                difficulty_dict = {
                    'score':      difficulty.difficulty_score,
                    'level':      difficulty.difficulty_level.value,
                    'num_agents': difficulty.num_agents,
                    'ego_speed':  difficulty.ego_speed,
                }
            except Exception:
                pass

        # ------------------------------------------------------------------
        # 12. Assemble
        # ------------------------------------------------------------------
        return {
            # raw sensors
            'lidar_original':             lidar_original,        # [N, 3]
            'lidar_bev':                  lidar_bev,             # [2, H, W]
            'camera_images':              camera_images,         # List[Tensor[3,H,W]]
            # labels
            'labels':                     labels_tensor,
            # ego
            'agent_states':               agent_states,          # [1, 7]
            'agent_history':              agent_history,         # [1, T, 7]
            'gt_trajectory':              gt_trajectory,         # [1, F, 5]
            # multi-agent
            'multi_agent_states':         multi_agent_states,    # [N, 5]
            'multi_agent_history':        multi_agent_history,   # [N, T, 7]
            # context
            'intersection_features':      intersection_features,       # [4, 5]
            'goal_features':              goal_features,               # [1, 5]
            'traffic_control_features':   traffic_control_features,    # [8, 5]
            'pedestrian_features':        pedestrian_features,         # [10, 5]
            # mission / map
            'route_info':                 route_info,
            'vector_map':                 vector_map_features,
            # meta
            'difficulty':                 difficulty_dict,
            'token':                      token,
        }

    # =========================================================================
    # Multi-agent extraction (ported from NavsimDataset)
    # =========================================================================

    def _extract_multi_agent_states(
        self,
        scene: Scene,
        current_frame_idx: int,
        max_agents: int = 32,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract unified multi-agent state tensor with ego as agent 0.

        All coordinates are in the ego frame of the current timestep.

        Returns:
            agent_states:  (N, 5)  float32  [x, y, vx, vy, heading]
            track_tokens:  List[str] of length N  (track_tokens[0] = 'ego')
        """
        DYNAMIC_AGENT_TYPES = {'vehicle', 'pedestrian', 'bicycle'}

        current_frame = scene.frames[current_frame_idx]
        ego           = current_frame.ego_status
        annotations   = current_frame.annotations

        # Ego is at origin in its own frame
        ego_state = np.array(
            [0.0, 0.0, ego.ego_velocity[0], ego.ego_velocity[1], 0.0],
            dtype=np.float32,
        )

        nearby = []  # (distance, state_array, track_token)

        if annotations is not None and len(annotations.boxes) > 0:
            for i in range(len(annotations.boxes)):
                if annotations.names[i] not in DYNAMIC_AGENT_TYPES:
                    continue
                box = annotations.boxes[i]
                x       = float(box[BoundingBoxIndex.X])
                y       = float(box[BoundingBoxIndex.Y])
                heading = float(box[BoundingBoxIndex.HEADING])
                vx      = float(annotations.velocity_3d[i][0])
                vy      = float(annotations.velocity_3d[i][1])
                dist    = np.sqrt(x ** 2 + y ** 2)
                nearby.append((
                    dist,
                    np.array([x, y, vx, vy, heading], dtype=np.float32),
                    annotations.track_tokens[i],
                ))

        nearby.sort(key=lambda t: t[0])
        nearby = nearby[: max_agents - 1]

        states = np.zeros((max_agents, 5), dtype=np.float32)
        states[0] = ego_state
        track_tokens = ['ego']

        for idx, (_, state_arr, tok) in enumerate(nearby):
            states[idx + 1] = state_arr
            track_tokens.append(tok)

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

        Transforms each historical agent position from the frame's local
        ego coordinates into the *current* frame's ego coordinate system.

        Returns:
            history: (N, T_hist, 7)  float32
                     [x, y, vx, vy, ax, ay, heading]  oldest -> newest
        """
        DYNAMIC_AGENT_TYPES = {'vehicle', 'pedestrian', 'bicycle'}

        N = max_agents
        T = num_history
        history = np.zeros((N, T, 7), dtype=np.float32)

        cur_ego = scene.frames[current_frame_idx].ego_status
        cur_x, cur_y, cur_h = cur_ego.ego_pose
        cos_cur = np.cos(-cur_h)
        sin_cur = np.sin(-cur_h)

        start_idx    = max(0, current_frame_idx - T + 1)
        frame_indices = list(range(start_idx, current_frame_idx + 1))
        time_offset   = T - len(frame_indices)

        for t_local, frame_idx in enumerate(frame_indices):
            t_out  = t_local + time_offset
            frame  = scene.frames[frame_idx]
            h_ego  = frame.ego_status
            hist_x, hist_y, hist_h = h_ego.ego_pose
            cos_hist = np.cos(hist_h)
            sin_hist = np.sin(hist_h)

            # Ego position in current frame
            dx = hist_x - cur_x
            dy = hist_y - cur_y
            ego_in_cur_x = cos_cur * dx - sin_cur * dy
            ego_in_cur_y = sin_cur * dx + cos_cur * dy
            ego_heading_in_cur = hist_h - cur_h

            # Ego velocity rotated to current frame
            hvx, hvy = h_ego.ego_velocity
            gvx = cos_hist * hvx - sin_hist * hvy
            gvy = sin_hist * hvx + cos_hist * hvy
            ego_vx_cur = cos_cur * gvx - sin_cur * gvy
            ego_vy_cur = sin_cur * gvx + cos_cur * gvy

            # Ego acceleration rotated to current frame
            hax, hay = h_ego.ego_acceleration
            gax = cos_hist * hax - sin_hist * hay
            gay = sin_hist * hax + cos_hist * hay
            ego_ax_cur = cos_cur * gax - sin_cur * gay
            ego_ay_cur = sin_cur * gax + cos_cur * gay

            history[0, t_out] = [
                ego_in_cur_x, ego_in_cur_y,
                ego_vx_cur, ego_vy_cur,
                ego_ax_cur, ego_ay_cur,
                ego_heading_in_cur,
            ]

            # Other agents
            annotations = frame.annotations
            if annotations is None or len(annotations.boxes) == 0:
                continue

            frame_token_to_idx = {}
            for i, name in enumerate(annotations.names):
                if name in DYNAMIC_AGENT_TYPES:
                    frame_token_to_idx[annotations.track_tokens[i]] = i

            for agent_row in range(1, N):
                # guard for indext out of list range
                if agent_row >= len(track_tokens):
                    continue                

                tok = track_tokens[agent_row] if agent_row < len(track_tokens) else ''
                if tok == '' or tok not in frame_token_to_idx:
                    continue

                ann_idx = frame_token_to_idx[tok]
                box     = annotations.boxes[ann_idx]

                local_x       = float(box[BoundingBoxIndex.X])
                local_y       = float(box[BoundingBoxIndex.Y])
                local_heading = float(box[BoundingBoxIndex.HEADING])

                # hist-local -> global
                global_x       = cos_hist * local_x - sin_hist * local_y + hist_x
                global_y       = sin_hist * local_x + cos_hist * local_y + hist_y
                global_heading = local_heading + hist_h

                # global -> current-local
                dx = global_x - cur_x
                dy = global_y - cur_y
                agent_x       = cos_cur * dx - sin_cur * dy
                agent_y       = sin_cur * dx + cos_cur * dy
                agent_heading = global_heading - cur_h

                # Velocity hist-local -> global -> current-local
                raw_vx = float(annotations.velocity_3d[ann_idx][0])
                raw_vy = float(annotations.velocity_3d[ann_idx][1])
                gvx = cos_hist * raw_vx - sin_hist * raw_vy
                gvy = sin_hist * raw_vx + cos_hist * raw_vy
                agent_vx = cos_cur * gvx - sin_cur * gvy
                agent_vy = sin_cur * gvx + cos_cur * gvy

                history[agent_row, t_out] = [
                    agent_x, agent_y,
                    agent_vx, agent_vy,
                    0.0, 0.0,  # acceleration approximated as zero
                    agent_heading,
                ]

        return torch.from_numpy(history)

    # =========================================================================
    # Group-C context feature extractors (ported from NavsimDataset)
    # =========================================================================

    def _extract_intersection_features(
        self,
        scene: Scene,
        scenario: NavSimScenario,
        current_frame_idx: int,
        max_intersection: int = 4,
    ) -> torch.Tensor:
        """
        Extract intersection context features.

        Feature vector per intersection:
            [in_intersection, approach_dist, turn_angle, right_of_way_proxy, dist_to_center]

        Returns: (K, 5) float32
        """
        features = np.zeros((max_intersection, 5), dtype=np.float32)

        try:
            map_api = scenario.map_api
        except Exception:
            return torch.from_numpy(features)

        ego = scene.frames[current_frame_idx].ego_status
        ego_x, ego_y, ego_h = ego.ego_pose
        ego_point   = Point2D(ego_x, ego_y)
        ego_shapely = ShapelyPoint(ego_x, ego_y)

        try:
            nearby = map_api.get_proximal_map_objects(
                ego_point, 50.0,
                [SemanticMapLayer.INTERSECTION, SemanticMapLayer.ROADBLOCK_CONNECTOR],
            )
        except Exception:
            return torch.from_numpy(features)

        intersections = nearby.get(SemanticMapLayer.INTERSECTION, [])
        connectors    = nearby.get(SemanticMapLayer.ROADBLOCK_CONNECTOR, [])

        if not intersections:
            return torch.from_numpy(features)

        scored = []
        for ix in intersections:
            centroid       = ix.polygon.centroid
            in_intersection = float(ix.polygon.contains(ego_shapely))
            approach_dist  = ix.polygon.distance(ego_shapely) if not in_intersection else 0.0
            dx             = centroid.x - ego_x
            dy             = centroid.y - ego_y
            turn_angle     = (np.arctan2(dy, dx) - ego_h + np.pi) % (2 * np.pi) - np.pi
            connector_count = sum(1 for c in connectors if ix.polygon.intersects(c.polygon))
            row_proxy      = 1.0 / max(connector_count, 1.0)
            dist           = np.sqrt(dx ** 2 + dy ** 2)
            scored.append((dist, np.array(
                [in_intersection, approach_dist, turn_angle, row_proxy, dist],
                dtype=np.float32,
            )))

        scored.sort(key=lambda t: t[0])
        for i, (_, feat) in enumerate(scored[:max_intersection]):
            features[i] = feat

        return torch.from_numpy(features)

    def _extract_goal_features(
        self,
        scene: Scene,
        scenario: NavSimScenario,
        current_frame_idx: int,
        max_goals: int = 1,
    ) -> torch.Tensor:
        """
        Extract goal/route features.

        Feature vector: [goal_rel_x, goal_rel_y, dist_to_goal, heading_to_goal,
                         route_length_remaining]

        Returns: (K, 5) float32
        """
        features = np.zeros((max_goals, 5), dtype=np.float32)

        current_frame = scene.frames[current_frame_idx]
        route_ids     = current_frame.roadblock_ids
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

        goal_x_global, goal_y_global = None, None
        route_length = 0.0

        for rb_id in route_ids:
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
            lanes = list(roadblock.interior_edges)
            if not lanes:
                continue
            lane          = lanes[0]
            discrete_path = lane.baseline_path.discrete_path
            if not discrete_path:
                continue
            route_length  += lane.baseline_path.length
            endpoint       = discrete_path[-1]
            goal_x_global  = endpoint.x
            goal_y_global  = endpoint.y

        if goal_x_global is None:
            return torch.from_numpy(features)

        dx          = goal_x_global - ego_x
        dy          = goal_y_global - ego_y
        goal_rel_x  = cos_h * dx - sin_h * dy
        goal_rel_y  = sin_h * dx + cos_h * dy
        dist        = np.sqrt(goal_rel_x ** 2 + goal_rel_y ** 2)
        heading     = np.arctan2(goal_rel_y, goal_rel_x)

        features[0] = [goal_rel_x, goal_rel_y, dist, heading, route_length]
        return torch.from_numpy(features)

    def _extract_traffic_control_features(
        self,
        scene: Scene,
        scenario: NavSimScenario,
        current_frame_idx: int,
        max_lights: int = 8,
    ) -> torch.Tensor:
        """
        Extract traffic control features.

        Feature vector per light: [rel_x, rel_y, is_red, distance, bearing]

        Returns: (K, 5) float32
        """
        features = np.zeros((max_lights, 5), dtype=np.float32)

        current_frame  = scene.frames[current_frame_idx]
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

        entries = []
        for lane_connector_id, is_red in traffic_lights:
            try:
                lc = map_api.get_map_object(
                    str(lane_connector_id), SemanticMapLayer.LANE_CONNECTOR,
                )
            except Exception:
                continue
            if lc is None:
                continue
            try:
                centroid = lc.polygon.centroid
                lc_x, lc_y = centroid.x, centroid.y
            except Exception:
                continue

            dx      = lc_x - ego_x
            dy      = lc_y - ego_y
            rel_x   = cos_h * dx - sin_h * dy
            rel_y   = sin_h * dx + cos_h * dy
            dist    = np.sqrt(rel_x ** 2 + rel_y ** 2)
            bearing = np.arctan2(rel_y, rel_x)

            entries.append((dist, np.array(
                [rel_x, rel_y, float(is_red), dist, bearing], dtype=np.float32,
            )))

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
        """
        Extract pedestrian-specific features.

        Feature vector per pedestrian:
            [rel_x, rel_y, speed, heading_alignment, crosswalk_proximity]

        Returns: (K, 5) float32
        """
        features      = np.zeros((max_pedestrians, 5), dtype=np.float32)
        current_frame = scene.frames[current_frame_idx]
        annotations   = current_frame.annotations

        if annotations is None or len(annotations.boxes) == 0:
            return torch.from_numpy(features)

        map_api = None
        try:
            map_api = scenario.map_api
        except Exception:
            pass

        ego = current_frame.ego_status
        ego_x, ego_y, ego_h = ego.ego_pose
        cos_h = np.cos(ego_h)
        sin_h = np.sin(ego_h)

        candidates = []
        for i, name in enumerate(annotations.names):
            if name != 'pedestrian':
                continue

            box   = annotations.boxes[i]
            rel_x = float(box[BoundingBoxIndex.X])
            rel_y = float(box[BoundingBoxIndex.Y])
            vx    = float(annotations.velocity_3d[i][0])
            vy    = float(annotations.velocity_3d[i][1])
            speed = np.sqrt(vx ** 2 + vy ** 2)
            dist  = np.sqrt(rel_x ** 2 + rel_y ** 2)

            norm_dist  = max(dist, 1e-6)
            norm_speed = max(speed, 1e-6)
            heading_alignment = float(np.clip(
                -(vx * rel_x + vy * rel_y) / (norm_dist * norm_speed), -1.0, 1.0,
            ))

            crosswalk_proximity = 0.0
            if map_api is not None:
                try:
                    ped_global_x = ego_x + cos_h * rel_x - sin_h * rel_y
                    ped_global_y = ego_y + sin_h * rel_x + cos_h * rel_y
                    ped_point    = Point2D(ped_global_x, ped_global_y)
                    nearby       = map_api.get_proximal_map_objects(
                        ped_point, crosswalk_radius, [SemanticMapLayer.CROSSWALK],
                    )
                    crosswalks = nearby.get(SemanticMapLayer.CROSSWALK, [])
                    if crosswalks:
                        ped_shapely = ShapelyPoint(ped_global_x, ped_global_y)
                        min_dist    = min(cw.polygon.distance(ped_shapely) for cw in crosswalks)
                        crosswalk_proximity = float(
                            np.clip(1.0 - min_dist / crosswalk_radius, 0.0, 1.0)
                        )
                except Exception:
                    pass

            candidates.append((dist, np.array(
                [rel_x, rel_y, speed, heading_alignment, crosswalk_proximity],
                dtype=np.float32,
            )))

        candidates.sort(key=lambda t: t[0])
        for i, (_, feat) in enumerate(candidates[:max_pedestrians]):
            features[i] = feat

        return torch.from_numpy(features)

    # =========================================================================
    # Sensor helpers
    # =========================================================================

    def _extract_camera_images(
        self,
        scene: Scene,
        current_frame_idx: int,
    ) -> List[torch.Tensor]:
        """Extract all 8 camera images from the current frame."""
        current_frame = scene.frames[current_frame_idx]
        cameras       = current_frame.cameras

        camera_list  = []
        camera_names = [
            'cam_f0', 'cam_l0', 'cam_l1', 'cam_l2',
            'cam_r0', 'cam_r1', 'cam_r2', 'cam_b0',
        ]

        for cam_name in camera_names:
            camera = getattr(cameras, cam_name, None)
            if camera is not None and camera.image is not None:
                img_tensor = torch.from_numpy(camera.image).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
                camera_list.append(img_tensor)

        return camera_list

    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR point cloud to BEV (density + height channels)."""
        H, W = self.bev_size
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2] if point_cloud.shape[1] >= 3 else np.zeros_like(x)

        x_min, x_max = -self.bev_range, self.bev_range
        y_min, y_max = -self.bev_range, self.bev_range

        xi = ((x - x_min) / (x_max - x_min) * W).astype(int)
        yi = ((y - y_min) / (y_max - y_min) * H).astype(int)

        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi, yi, z = xi[valid], yi[valid], z[valid]

        density = np.zeros((H, W), dtype=np.float32)
        height  = np.zeros((H, W), dtype=np.float32)
        for k in range(len(xi)):
            density[yi[k], xi[k]] += 1
            height[yi[k], xi[k]]   = max(height[yi[k], xi[k]], z[k])

        density = np.clip(density / 10.0, 0, 1)
        height  = np.clip((height + 2) / 5.0, 0, 1)

        return torch.from_numpy(np.stack([density, height])).float()

    # =========================================================================
    # Fallback / utility
    # =========================================================================

    def _get_empty_labels(self) -> Dict[str, torch.Tensor]:
        """Return zero-filled label dict at target resolution."""
        return {
            'drivable_area':       torch.zeros(self.bev_size, dtype=torch.float32),
            'lane_boundaries':     torch.zeros(self.bev_size, dtype=torch.float32),
            'lane_dividers':       torch.zeros(self.bev_size, dtype=torch.float32),
            'vehicle_occupancy':   torch.zeros(self.bev_size, dtype=torch.float32),
            'pedestrian_occupancy': torch.zeros(self.bev_size, dtype=torch.float32),
            'velocity_x':          torch.zeros(self.bev_size, dtype=torch.float32),
            'velocity_y':          torch.zeros(self.bev_size, dtype=torch.float32),
            'ego_mask':            torch.zeros(self.bev_size, dtype=torch.float32),
            'traffic_lights':      torch.zeros(self.bev_size, dtype=torch.uint8),
            'vehicle_classes':     torch.zeros(self.bev_size, dtype=torch.uint8),
            'crosswalks':          torch.zeros(self.bev_size, dtype=torch.float32),
            'stop_lines':          torch.zeros(self.bev_size, dtype=torch.float32),
        }


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import torch

    # ── config — edit these to match your environment ────────────────────────
    MAP_ROOT    = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
    DATA_SPLIT  = "mini"
    TEST_INDICES = [0, 1, 2]

    os.environ.setdefault('NUPLAN_MAPS_ROOT',    MAP_ROOT)
    os.environ.setdefault('OPENSCENE_DATA_ROOT', str(Path(MAP_ROOT).parent))

    # ── helpers ───────────────────────────────────────────────────────────────
    W = 70

    def header(title: str):
        print("\n" + "=" * W)
        print(f" {title}")
        print("=" * W)

    def check(label: str, cond: bool, detail: str = "") -> bool:
        tag = "OK" if cond else "FAIL"
        msg = f"  [{tag}] {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        return cond

    def tensor_stats(t: torch.Tensor) -> str:
        return (
            f"shape={tuple(t.shape)}  dtype={t.dtype}  "
            f"min={t.min():.3f}  max={t.max():.3f}  "
            f"nan={torch.isnan(t.float()).sum().item()}"
        )

    all_ok = True

    # ── 1. Environment ────────────────────────────────────────────────────────
    header("1. Environment")
    all_ok &= check("MAP_ROOT exists", Path(MAP_ROOT).exists(), MAP_ROOT)

    # ── 2. Dataset construction ───────────────────────────────────────────────
    header("2. Dataset construction")
    try:
        dataset = EnhancedNavsimDataset(
            data_split          = DATA_SPLIT,
            bev_size            = (200, 200),
            bev_range           = 50.0,
            extract_labels      = True,
            extract_route_info  = True,
            extract_vector_maps = True,
            map_root            = MAP_ROOT,
            map_version         = "nuplan-maps-v1.0",
            max_agents          = 32,
        )
        all_ok &= check("Dataset created", True, f"{len(dataset)} scenes")
    except Exception as exc:
        check("Dataset created", False, str(exc))
        print("\n  Cannot continue without a dataset. Exiting.")
        sys.exit(1)

    all_ok &= check("Dataset non-empty", len(dataset) > 0)

    # ── 3. Single-sample smoke test ───────────────────────────────────────────
    header("3. Single-sample smoke test  (index 0)")
    sample = dataset[0]

    EXPECTED_KEYS = {
        'lidar_original', 'lidar_bev', 'camera_images', 'labels',
        'agent_states', 'agent_history', 'gt_trajectory',
        'multi_agent_states', 'multi_agent_history',
        'intersection_features', 'goal_features',
        'traffic_control_features', 'pedestrian_features',
        'route_info', 'vector_map', 'difficulty', 'token',
    }
    missing = EXPECTED_KEYS - sample.keys()
    all_ok &= check("All keys present", not missing,
                    f"missing: {missing}" if missing else "")

    # ── 4. Tensor shapes ─────────────────────────────────────────────────────
    header("4. Tensor shapes")

    BEV = dataset.bev_size        # (200, 200)
    N   = dataset.max_agents      # 32
    H   = dataset.history_length  # 4
    F   = dataset.trajectory_sampling.num_poses

    shape_checks = [
        ("lidar_bev",                  sample['lidar_bev'].shape,                  (2, *BEV)),
        ("agent_states",               sample['agent_states'].shape,               (1, 7)),
        ("agent_history",              sample['agent_history'].shape,              (1, H, 7)),
        ("gt_trajectory",              sample['gt_trajectory'].shape,              (1, F, 5)),
        ("multi_agent_states",         sample['multi_agent_states'].shape,         (N, 5)),
        ("multi_agent_history",        sample['multi_agent_history'].shape,        (N, H, 7)),
        ("intersection_features",      sample['intersection_features'].shape,      (4, 5)),
        ("goal_features",              sample['goal_features'].shape,              (1, 5)),
        ("traffic_control_features",   sample['traffic_control_features'].shape,   (8, 5)),
        ("pedestrian_features",        sample['pedestrian_features'].shape,        (10, 5)),
    ]
    for name, got, expected in shape_checks:
        all_ok &= check(f"{name:<32}", got == expected,
                        f"got {got}  expected {expected}")

    # camera_images is a List[Tensor]
    cam_imgs = sample['camera_images']
    all_ok &= check("camera_images is a list",  isinstance(cam_imgs, list))
    all_ok &= check("camera_images non-empty",  len(cam_imgs) > 0,
                    f"{len(cam_imgs)} cameras")
    if cam_imgs:
        all_ok &= check("camera image shape [3,H,W]",
                        cam_imgs[0].ndim == 3 and cam_imgs[0].shape[0] == 3,
                        str(tuple(cam_imgs[0].shape)))

    # lidar_original is variable-length
    all_ok &= check("lidar_original ndim==2",
                    sample['lidar_original'].ndim == 2 and
                    sample['lidar_original'].shape[1] == 3,
                    str(tuple(sample['lidar_original'].shape)))

    # ── 5. BEV label channels ────────────────────────────────────────────────
    header("5. BEV label channels")
    labels = sample['labels']
    all_ok &= check("12 label channels", len(labels) == 12, f"got {len(labels)}")
    for k, v in labels.items():
        ok = (v.shape == BEV) and torch.isfinite(v.float()).all().item()
        all_ok &= check(f"  {k:<30}", ok, tensor_stats(v))

    # ── 6. Coordinate-system sanity ──────────────────────────────────────────
    header("6. Coordinate-system sanity")

    ego_x = sample['agent_states'][0, 0].item()
    ego_y = sample['agent_states'][0, 1].item()
    ego_h = sample['agent_states'][0, 6].item()
    all_ok &= check("Ego x == 0",       abs(ego_x) < 1e-4, f"x={ego_x:.6f}")
    all_ok &= check("Ego y == 0",       abs(ego_y) < 1e-4, f"y={ego_y:.6f}")
    all_ok &= check("Ego heading == 0", abs(ego_h) < 1e-4, f"h={ego_h:.6f}")

    hist = sample['agent_history'][0]          # [T, 7]
    last_x = hist[-1, 0].item()
    last_y = hist[-1, 1].item()
    all_ok &= check("History last-step x ≈ 0", abs(last_x) < 1e-3, f"x={last_x:.6f}")
    all_ok &= check("History last-step y ≈ 0", abs(last_y) < 1e-3, f"y={last_y:.6f}")

    # History should not be all zeros (would mean the loop produced nothing useful)
    all_ok &= check("History is not all-zero", hist.abs().sum().item() > 1e-6)

    # If ego was moving, oldest frame should be behind (x < 0 in ego frame).
    # We check this only when the ego speed is non-trivial to avoid false FAILs
    # on stationary scenes.
    ego_speed = sample['agent_states'][0, 2:4].norm().item()
    oldest_x  = hist[0, 0].item()
    if ego_speed > 0.5:
        all_ok &= check("History oldest frame behind ego (x < 0)",
                        oldest_x < 0, f"oldest_x={oldest_x:.3f}  ego_speed={ego_speed:.2f}")
    else:
        print(f"  [SKIP] History oldest-frame check skipped (ego nearly stationary, "
              f"speed={ego_speed:.2f} m/s, oldest_x={oldest_x:.3f})")

    # ── 7. GT trajectory sanity ───────────────────────────────────────────────
    header("7. GT trajectory sanity")
    traj = sample['gt_trajectory'][0]          # [F, 5]
    all_ok &= check("Trajectory finite",       torch.isfinite(traj).all().item())
    all_ok &= check("First wp x > 0 (forward)", traj[0, 0].item() > 0,
                    f"x={traj[0, 0].item():.3f}")
    x_seq = traj[:, 0]
    all_ok &= check("x coords monotonically increasing",
                    (x_seq[1:] > x_seq[:-1]).all().item(),
                    "straight-ahead drive expected")

    # ── 8. Multi-agent sanity ─────────────────────────────────────────────────
    header("8. Multi-agent sanity")
    ma_states = sample['multi_agent_states']   # [N, 5]
    # Row 0 is ego — should be at origin
    all_ok &= check("Multi-agent ego row x == 0", abs(ma_states[0, 0].item()) < 1e-4,
                    f"x={ma_states[0, 0].item():.6f}")
    all_ok &= check("Multi-agent ego row y == 0", abs(ma_states[0, 1].item()) < 1e-4,
                    f"y={ma_states[0, 1].item():.6f}")
    all_ok &= check("Multi-agent states finite", torch.isfinite(ma_states).all().item())
    all_ok &= check("Multi-agent history finite",
                    torch.isfinite(sample['multi_agent_history']).all().item())

    # ── 9. Context features sanity ───────────────────────────────────────────
    header("9. Context features sanity")
    for key, expected_shape in [
        ('intersection_features',    (4, 5)),
        ('goal_features',            (1, 5)),
        ('traffic_control_features', (8, 5)),
        ('pedestrian_features',      (10, 5)),
    ]:
        t  = sample[key]
        ok = t.shape == expected_shape and torch.isfinite(t).all().item()
        all_ok &= check(f"{key:<32}", ok, tensor_stats(t))

    # ── 10. Multi-sample reproducibility ─────────────────────────────────────
    header("10. Multi-sample reproducibility")
    for idx in TEST_INDICES:
        try:
            s  = dataset[idx]
            ok = (
                s['lidar_bev'].shape    == (2, *BEV)   and
                s['agent_states'].shape == (1, 7)       and
                s['gt_trajectory'].shape == (1, F, 5)  and
                torch.isfinite(s['gt_trajectory']).all().item()
            )
            all_ok &= check(f"Sample {idx}", ok, f"token={s['token']}")
        except Exception as exc:
            all_ok &= check(f"Sample {idx}", False, str(exc))

    # ── 11. NaN / Inf guard ───────────────────────────────────────────────────
    header("11. NaN / Inf guard")
    TENSOR_KEYS = [
        'lidar_bev', 'agent_states', 'agent_history', 'gt_trajectory',
        'multi_agent_states', 'multi_agent_history',
        'intersection_features', 'goal_features',
        'traffic_control_features', 'pedestrian_features',
    ]
    for idx in TEST_INDICES:
        s = dataset[idx]
        for key in TENSOR_KEYS:
            t      = s[key]
            finite = torch.isfinite(t.float()).all().item()
            all_ok &= check(f"sample[{idx}]['{key}'] finite", finite)

    # ── 12. Tensor statistics report ─────────────────────────────────────────
    header("12. Tensor statistics (sample 0)")
    for key in TENSOR_KEYS:
        t = sample[key]
        print(f"  {key:<34}  {tensor_stats(t)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("Summary")
    if all_ok:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see [FAIL] lines above")
    print("=" * W + "\n")
    sys.exit(0 if all_ok else 1)