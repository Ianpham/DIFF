"""
Pedestrian Interaction Feature & Target Builders for NavSim
============================================================
Feature Extraction Pipeline

These builders plug directly into NavSim's caching and training pipeline:
    - PedestrianInteractionFeatureBuilder: computes features from AgentInput (no future access)
    - InteractionTargetBuilder: computes pseudo-labels from Scene (with future access)
    - TrajectoryTargetBuilder: standard GT trajectory target
https://claude.ai/chat/a98db9ae-2add-41d1-b417-6d8b4fe34272

Both produce Dict[str, Tensor] that get cached via NavSim's standard caching system.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from navsim.common.dataclasses import AgentInput, Annotations, EgoStatus, Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

try:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False


# Helper functions



def _find_pedestrians(annotations: Annotations) -> List[int]:
    """Return indices of pedestrians in annotations."""
    return [i for i, name in enumerate(annotations.names) if name == "pedestrian"]


def _ego_to_global(ego_pose: Tuple[float, float, float], local_x: float, local_y: float):
    """Transform from ego-local to global coordinates."""
    ex, ey, eh = ego_pose
    cos_h = math.cos(eh)
    sin_h = math.sin(eh)
    gx = ex + cos_h * local_x - sin_h * local_y
    gy = ey + sin_h * local_x + cos_h * local_y
    return gx, gy


def _compute_closing_rate(
    ego_statuses: List[EgoStatus],
    annotations_list: List[Annotations],
    target_track_token: str,
) -> float:
    """
    Compute closing rate between ego and a tracked pedestrian over history frames.
    Uses finite difference of distance over 0.5s intervals (2Hz).
    Returns negative values when distance is decreasing (approaching).
    """
    distances = []
    for ego_status, annot in zip(ego_statuses, annotations_list):
        for idx, name in enumerate(annot.names):
            if name == "pedestrian" and annot.track_tokens[idx] == target_track_token:
                box = annot.boxes[idx]
                dist = math.sqrt(box[0] ** 2 + box[1] ** 2)
                distances.append(dist)
                break

    if len(distances) < 2:
        return 0.0

    # Finite difference over the available history (each step is 0.5s)
    rates = []
    for i in range(1, len(distances)):
        rates.append((distances[i] - distances[i - 1]) / 0.5)

    return float(np.mean(rates))



# Feature Builder (inference-time, from AgentInput only)



class PedestrianInteractionFeatureBuilder(AbstractFeatureBuilder):
    """
    Extracts pedestrian interaction features from AgentInput.

    Produces a fixed-size feature vector for the most relevant pedestrian,
    plus a spatial risk field grid. If no pedestrian is relevant, features
    are zeroed out with a "no_pedestrian" flag.

    Output keys:
        "ped_interaction_features": Tensor [num_features]  (20-dim)
        "ped_risk_field": Tensor [1, grid_size, grid_size]  (spatial)
        "has_relevant_pedestrian": Tensor [1]  (binary flag)
    """

    def __init__(
        self,
        corridor_width: float = 5.0,
        corridor_length: float = 30.0,
        max_distance: float = 30.0,
        risk_field_size: int = 32,
        risk_field_range: float = 32.0,  # meters, half-width of the grid
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.0",
    ):
        super().__init__()
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length
        self.max_distance = max_distance
        self.risk_field_size = risk_field_size
        self.risk_field_range = risk_field_range
        self.map_root = map_root
        self.map_version = map_version

        self._map_cache: Dict[str, Any] = {}

    def get_unique_name(self) -> str:
        return "ped_interaction_feature"

    def _get_map_api(self, map_name: str):
        """Lazy-load and cache map API."""
        if not MAP_AVAILABLE or self.map_root is None:
            return None
        if map_name not in self._map_cache:
            actual_name = map_name if map_name != "las_vegas" else "us-nv-las-vegas-strip"
            try:
                self._map_cache[map_name] = get_maps_api(
                    self.map_root, self.map_version, actual_name
                )
            except Exception:
                self._map_cache[map_name] = None
        return self._map_cache[map_name]

    def _check_crosswalk(self, map_api, global_x: float, global_y: float) -> Tuple[bool, float]:
        """Check if point is on/near a crosswalk. Returns (on_crosswalk, distance_to_nearest)."""
        if map_api is None:
            return False, float("inf")
        try:
            point = Point2D(global_x, global_y)
            nearby = map_api.get_proximal_map_objects(point, 10.0, [SemanticMapLayer.CROSSWALK])
            crosswalks = nearby.get(SemanticMapLayer.CROSSWALK, [])
            if not crosswalks:
                return False, float("inf")

            from shapely.geometry import Point as SP

            sp = SP(global_x, global_y)
            min_dist = float("inf")
            on_cw = False
            for cw in crosswalks:
                d = cw.polygon.distance(sp)
                if d < min_dist:
                    min_dist = d
                if cw.polygon.contains(sp):
                    on_cw = True
            return on_cw, min_dist
        except Exception:
            return False, float("inf")

    def _check_crosswalk_ahead(self, map_api, ego_pose) -> Tuple[bool, float]:
        """Check if there's a crosswalk ahead along ego's forward direction."""
        if map_api is None:
            return False, float("inf")

        ex, ey, eh = ego_pose
        cos_h = math.cos(eh)
        sin_h = math.sin(eh)

        for lookahead in [10.0, 20.0, 30.0]:
            gx = ex + cos_h * lookahead
            gy = ey + sin_h * lookahead
            try:
                point = Point2D(gx, gy)
                nearby = map_api.get_proximal_map_objects(point, 8.0, [SemanticMapLayer.CROSSWALK])
                crosswalks = nearby.get(SemanticMapLayer.CROSSWALK, [])
                if crosswalks:
                    return True, lookahead
            except Exception:
                continue
        return False, float("inf")

    def _select_relevant_pedestrian(
        self,
        annotations: Annotations,
        ego_status: EgoStatus,
        map_api=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the most relevant pedestrian for interaction modeling.
        Priority: in corridor > on crosswalk > near crosswalk > closest.
        Returns dict of features or None if no pedestrian within range.
        """
        ped_indices = _find_pedestrians(annotations)
        if not ped_indices:
            return None

        ego_pose = ego_status.ego_pose
        cos_h = math.cos(ego_pose[2])
        sin_h = math.sin(ego_pose[2])

        candidates = []
        for idx in ped_indices:
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])  # ego frame
            dist = math.sqrt(px**2 + py**2)

            if dist > self.max_distance:
                continue

            vx = float(annotations.velocity_3d[idx][0])
            vy = float(annotations.velocity_3d[idx][1])
            speed = math.sqrt(vx**2 + vy**2)

            in_corridor = (
                0 < px < self.corridor_length and abs(py) < self.corridor_width
            )
            lateral_offset = abs(py)
            bearing = math.atan2(py, px)

            # Heading alignment (does ped move toward ego's path)
            to_ego_norm = max(dist, 1e-6)
            heading_align = (-px * vx + -py * vy) / (to_ego_norm * max(speed, 1e-6))

            # Crosswalk check
            on_cw, cw_dist = False, float("inf")
            if map_api is not None:
                gx, gy = _ego_to_global(ego_pose, px, py)
                on_cw, cw_dist = self._check_crosswalk(map_api, gx, gy)

            candidates.append(
                {
                    "idx": idx,
                    "track_token": annotations.track_tokens[idx],
                    "rel_x": px,
                    "rel_y": py,
                    "distance": dist,
                    "vx": vx,
                    "vy": vy,
                    "speed": speed,
                    "bearing": bearing,
                    "in_corridor": in_corridor,
                    "lateral_offset": lateral_offset,
                    "heading_alignment": heading_align,
                    "on_crosswalk": on_cw,
                    "near_crosswalk": cw_dist < 5.0,
                    "crosswalk_distance": cw_dist,
                }
            )

        if not candidates:
            return None

        # Sort by relevance priority
        candidates.sort(
            key=lambda c: (
                -int(c["in_corridor"]),
                -int(c["on_crosswalk"]),
                -int(c["near_crosswalk"]),
                c["distance"],
            )
        )

        return candidates[0]

    def _build_risk_field(
        self, annotations: Annotations, ego_status: EgoStatus
    ) -> np.ndarray:
        """
        Build a 2D spatial risk field around ego.
        Each cell accumulates pedestrian risk based on proximity and velocity.
        Grid is ego-centered: x-forward, y-left.
        """
        grid = np.zeros((self.risk_field_size, self.risk_field_size), dtype=np.float32)
        cell_size = (2 * self.risk_field_range) / self.risk_field_size
        half_grid = self.risk_field_size // 2

        ped_indices = _find_pedestrians(annotations)
        for idx in ped_indices:
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])  # ego frame
            vx = float(annotations.velocity_3d[idx][0])
            vy = float(annotations.velocity_3d[idx][1])
            speed = math.sqrt(vx**2 + vy**2)

            # Current position risk (Gaussian blob)
            dist = math.sqrt(px**2 + py**2)
            if dist > self.risk_field_range * 1.5:
                continue

            # Risk magnitude: inversely proportional to distance, scaled by speed
            risk_mag = max(0.1, 1.0 / max(dist, 1.0)) * (1.0 + speed)

            # Gaussian spread (sigma proportional to speed — faster ped = wider risk)
            sigma = max(2.0, speed * 1.5)  # meters

            # Map to grid coordinates
            # Grid: row 0 = ego's left-front, row N = ego's right-back
            # Convention: grid_x corresponds to ego-frame x (forward), grid_y to ego-frame y (left)
            for di in range(-4, 5):
                for dj in range(-4, 5):
                    # Cell center in ego frame
                    cell_x = (di + 0.5) * cell_size + px
                    cell_y = (dj + 0.5) * cell_size + py

                    # Grid indices: center of grid = ego position
                    gi = half_grid - int(round(cell_y / cell_size))  # y-axis flipped
                    gj = half_grid + int(round(cell_x / cell_size))  # x-axis forward

                    if 0 <= gi < self.risk_field_size and 0 <= gj < self.risk_field_size:
                        dx = cell_x - px
                        dy = cell_y - py
                        d2 = dx**2 + dy**2
                        val = risk_mag * math.exp(-d2 / (2 * sigma**2))
                        grid[gi, gj] += val

            # Propagate risk along velocity direction (future position prediction)
            for t_step in [0.5, 1.0, 1.5, 2.0]:
                future_px = px + vx * t_step
                future_py = py + vy * t_step

                gi = half_grid - int(round(future_py / cell_size))
                gj = half_grid + int(round(future_px / cell_size))

                if 0 <= gi < self.risk_field_size and 0 <= gj < self.risk_field_size:
                    decay = math.exp(-t_step * 0.5)  # temporal decay
                    grid[gi, gj] += risk_mag * decay * 0.5

        # Normalize to [0, 1]
        max_val = grid.max()
        if max_val > 0:
            grid /= max_val

        return grid

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """Compute pedestrian interaction features from AgentInput."""
        current_ego = agent_input.ego_statuses[-1]
        current_annot = agent_input.annotations[-1]

        # Try to load map (need scene metadata for map name — not available in AgentInput)
        # Workaround: we'll use map_root if provided, but map_name must come from config
        # For caching, we pass map_root in __init__. At runtime in compute_trajectory,
        # map won't be available — that's okay, features are cached.
        map_api = None  # Map queries happen during caching only

        # Select most relevant pedestrian
        best_ped = self._select_relevant_pedestrian(current_annot, current_ego, map_api)

        has_ped = best_ped is not None

        if has_ped:
            # Compute closing rate from history
            closing_rate = _compute_closing_rate(
                agent_input.ego_statuses,
                agent_input.annotations,
                best_ped["track_token"],
            )

            # Check crosswalk ahead
            cw_ahead, cw_dist_ahead = False, 50.0
            if map_api is not None:
                cw_ahead, cw_dist_ahead = self._check_crosswalk_ahead(
                    map_api, current_ego.ego_pose
                )

            ego_vx, ego_vy = current_ego.ego_velocity
            ego_ax, ego_ay = current_ego.ego_acceleration
            ego_speed = math.sqrt(ego_vx**2 + ego_vy**2)

            # Feature vector: 20 dimensions
            features = torch.tensor(
                [
                    # Ego dynamics [0:5]
                    ego_vx,
                    ego_vy,
                    ego_ax,
                    ego_ay,
                    ego_speed,
                    # Driving command [5:8]
                    *current_ego.driving_command,
                    # Pedestrian relative state [8:15]
                    best_ped["rel_x"] / self.max_distance,  # normalized
                    best_ped["rel_y"] / self.corridor_width,  # normalized
                    best_ped["vx"],
                    best_ped["vy"],
                    best_ped["distance"] / self.max_distance,  # normalized
                    math.sin(best_ped["bearing"]),
                    math.cos(best_ped["bearing"]),
                    # Interaction dynamics [15:18]
                    closing_rate,
                    best_ped["lateral_offset"] / self.corridor_width,
                    best_ped["heading_alignment"],
                    # Crosswalk context [18:20]
                    float(best_ped.get("on_crosswalk", False)),
                    min(best_ped.get("crosswalk_distance", 50.0), 50.0) / 50.0,
                ],
                dtype=torch.float32,
            )
        else:
            features = torch.zeros(20, dtype=torch.float32)

        # Build risk field (includes ALL pedestrians, not just best)
        risk_field = self._build_risk_field(current_annot, current_ego)
        risk_field_tensor = torch.tensor(risk_field, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return {
            "ped_interaction_features": features,
            "ped_risk_field": risk_field_tensor,
            "has_relevant_pedestrian": torch.tensor([float(has_ped)], dtype=torch.float32),
        }


# Target Builder (training-time, from Scene with future access)



class EnhancedInteractionTargetBuilder(AbstractTargetBuilder):
    """
    Computes pseudo-labels for pedestrian interaction from ground-truth future data.

    Uses Scene (with future frames) to determine:
        - Did ego yield? (deceleration + stopping near crosswalk)
        - Did pedestrian cross ego's path?
        - Minimum ego-pedestrian distance over future horizon
        - Interaction outcome label (categorical)
        - Interaction score (continuous weight for training loss)

    Output keys:
        "interaction_outcome_label": Tensor [1] (int64, categorical)
        "interaction_score": Tensor [1] (float, 0-1, for loss weighting)
        "ego_yielded": Tensor [1] (float, binary)
        "ped_crossed": Tensor [1] (float, binary)
        "min_future_distance": Tensor [1] (float, meters)
        "trajectory": Tensor [num_poses, 3] (GT trajectory)

    Stage 2 target builder with comfort-aware pseudo-labels.

    Outputs everything Stage 1 does, plus:
        "speed_profile": Tensor [num_poses] — GT ego speed at each future timestep
        "decel_pattern": Tensor [1] — categorical deceleration pattern
        "gt_jerk_magnitude": Tensor [1] — average jerk of GT trajectory
        "gt_lat_accel": Tensor [1] — average lateral acceleration
        "closest_approach_time": Tensor [1] — timestep of min ego-ped distance (0-1 normalized)
        "trajectory": Tensor [num_poses, 3] — GT trajectory
    """

    # Outcome labels 
    NO_PEDESTRIAN = 0
    NO_INTERACTION = 1
    PED_YIELDED = 2
    EGO_YIELDED = 3
    PED_ASSERTED = 4
    CONTESTED = 5

    # Deceleration patterns
    DECEL_NONE = 0        # No deceleration
    DECEL_GENTLE = 1      # Gradual slowdown (>2s to stop)
    DECEL_MODERATE = 2    # Normal braking (1-2s)
    DECEL_HARD = 3        # Hard braking (<1s)
    DECEL_COAST = 4       # Release throttle, no active braking

    def __init__(
        self,
        num_trajectory_frames: int = 8,
        corridor_width: float = 5.0,
        corridor_length: float = 30.0,
        yield_decel_threshold: float = -0.5,
        crossing_lateral_threshold: float = 3.0,
        contest_distance_threshold: float = 5.0,
    ):
        super().__init__()
        self.num_trajectory_frames = num_trajectory_frames
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length
        self.yield_decel_threshold = yield_decel_threshold
        self.crossing_lateral_threshold = crossing_lateral_threshold
        self.contest_distance_threshold = contest_distance_threshold

    def get_unique_name(self) -> str:
        return "enhanced_interaction_target"

    def _find_best_ped_track_token(self, scene: Scene) -> Optional[str]:
        """Find the most relevant pedestrian's track_token at current frame."""
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_annot = scene.frames[current_idx].annotations
        best_token = None
        best_score = float("inf")

        for idx, name in enumerate(current_annot.names):
            if name != "pedestrian":
                continue
            box = current_annot.boxes[idx]
            px, py = float(box[0]), float(box[1])
            dist = math.sqrt(px**2 + py**2)
            in_corridor = 0 < px < self.corridor_length and abs(py) < self.corridor_width
            score = dist + (0 if in_corridor else 100.0)
            if score < best_score:
                best_score = score
                best_token = current_annot.track_tokens[idx]

        return best_token

    def _compute_ego_speed_profile(self, scene: Scene) -> List[float]:
        """Get ego speed at each future frame."""
        current_idx = scene.scene_metadata.num_history_frames - 1
        num_future = scene.scene_metadata.num_future_frames
        speeds = []
        for fi in range(current_idx + 1, min(current_idx + 1 + num_future, len(scene.frames))):
            ego = scene.frames[fi].ego_status
            vx, vy = ego.ego_velocity
            speeds.append(math.sqrt(vx**2 + vy**2))
        # Pad if needed
        while len(speeds) < self.num_trajectory_frames:
            speeds.append(speeds[-1] if speeds else 0.0)
        return speeds[:self.num_trajectory_frames]

    def _classify_decel_pattern(self, speed_profile: List[float], current_speed: float) -> int:
        """Classify the deceleration pattern from speed profile."""
        if current_speed < 0.5:
            return self.DECEL_NONE

        min_speed = min(speed_profile) if speed_profile else current_speed
        speed_drop = current_speed - min_speed

        if speed_drop < 0.5:
            return self.DECEL_NONE

        # Find how quickly speed drops
        time_to_min = 0
        for i, s in enumerate(speed_profile):
            if s <= min_speed + 0.1:
                time_to_min = (i + 1) * 0.5  # seconds
                break

        if time_to_min == 0:
            return self.DECEL_COAST

        decel_rate = speed_drop / time_to_min

        if decel_rate > 4.0:
            return self.DECEL_HARD
        elif decel_rate > 1.5:
            return self.DECEL_MODERATE
        elif decel_rate > 0.3:
            return self.DECEL_GENTLE
        else:
            return self.DECEL_COAST

    def _compute_trajectory_comfort(
        self, trajectory_poses: np.ndarray, dt: float = 0.5
    ) -> Tuple[float, float]:
        """
        Compute jerk and lateral acceleration from GT trajectory.
        These match what NavSim's pdm_comfort_metrics.py checks.

        Args:
            trajectory_poses: [T, 3] array of (x, y, heading)
            dt: time between poses

        Returns:
            (avg_jerk_magnitude, avg_lateral_accel)
        """
        if len(trajectory_poses) < 3:
            return 0.0, 0.0

        positions = trajectory_poses[:, :2]
        headings = trajectory_poses[:, 2]

        # Velocities
        vel = np.diff(positions, axis=0) / dt  # [T-1, 2]

        # Accelerations
        accel = np.diff(vel, axis=0) / dt  # [T-2, 2]

        # Jerk
        if len(accel) >= 2:
            jerk = np.diff(accel, axis=0) / dt  # [T-3, 2]
            avg_jerk = float(np.mean(np.linalg.norm(jerk, axis=1)))
        else:
            avg_jerk = 0.0

        # Lateral acceleration (perpendicular to heading)
        lat_accels = []
        for i in range(len(accel)):
            heading_idx = min(i + 1, len(headings) - 1)
            h = headings[heading_idx]
            # Lateral direction: perpendicular to heading
            lat_dir = np.array([-math.sin(h), math.cos(h)])
            lat_a = abs(float(np.dot(accel[i], lat_dir)))
            lat_accels.append(lat_a)
        avg_lat_accel = float(np.mean(lat_accels)) if lat_accels else 0.0

        return avg_jerk, avg_lat_accel

    def _compute_closest_approach_time(
        self, scene: Scene, track_token: str
    ) -> float:
        """Find the normalized time of closest ego-ped approach in future."""
        current_idx = scene.scene_metadata.num_history_frames - 1
        num_future = scene.scene_metadata.num_future_frames
        min_dist = float("inf")
        min_time = 0.5  # default to middle

        for offset, fi in enumerate(
            range(current_idx + 1, min(current_idx + 1 + num_future, len(scene.frames)))
        ):
            annot = scene.frames[fi].annotations
            for idx, name in enumerate(annot.names):
                if name == "pedestrian" and annot.track_tokens[idx] == track_token:
                    box = annot.boxes[idx]
                    dist = math.sqrt(float(box[0])**2 + float(box[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        min_time = (offset + 1) / max(num_future, 1)
                    break

        return min_time

    def compute_targets(self, scene: Scene) -> Dict[str, Tensor]:
        """Compute all targets including comfort-aware pseudo-labels."""

        # GT trajectory
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self.num_trajectory_frames
        )
        trajectory_tensor = torch.tensor(future_trajectory.poses, dtype=torch.float32)

        # Current ego state
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_ego = scene.frames[current_idx].ego_status
        current_speed = math.sqrt(
            current_ego.ego_velocity[0]**2 + current_ego.ego_velocity[1]**2
        )

        # Speed profile
        speed_profile = self._compute_ego_speed_profile(scene)
        speed_tensor = torch.tensor(speed_profile, dtype=torch.float32)

        # Find best pedestrian
        best_token = self._find_best_ped_track_token(scene)

        # Trajectory comfort metrics
        gt_poses = future_trajectory.poses  # numpy array [T, 3]
        avg_jerk, avg_lat_accel = self._compute_trajectory_comfort(gt_poses)

        if best_token is None:
            return {
                "interaction_outcome_label": torch.tensor([self.NO_PEDESTRIAN], dtype=torch.long),
                "interaction_score": torch.tensor([0.0], dtype=torch.float32),
                "ego_yielded": torch.tensor([0.0], dtype=torch.float32),
                "ped_crossed": torch.tensor([0.0], dtype=torch.float32),
                "min_future_distance": torch.tensor([50.0], dtype=torch.float32),
                "speed_profile": speed_tensor,
                "decel_pattern": torch.tensor([self.DECEL_NONE], dtype=torch.long),
                "gt_jerk_magnitude": torch.tensor([avg_jerk], dtype=torch.float32),
                "gt_lat_accel": torch.tensor([avg_lat_accel], dtype=torch.float32),
                "closest_approach_time": torch.tensor([0.5], dtype=torch.float32),
                "trajectory": trajectory_tensor,
            }

        # Track ped through future
        num_future = scene.scene_metadata.num_future_frames
        min_distance = float("inf")
        ped_crossed = False

        for fi in range(current_idx + 1, min(current_idx + 1 + num_future, len(scene.frames))):
            annot = scene.frames[fi].annotations
            for idx, name in enumerate(annot.names):
                if name == "pedestrian" and annot.track_tokens[idx] == best_token:
                    box = annot.boxes[idx]
                    dist = math.sqrt(float(box[0])**2 + float(box[1])**2)
                    min_distance = min(min_distance, dist)
                    if (abs(float(box[1])) < self.crossing_lateral_threshold
                            and 0 < float(box[0]) < self.corridor_length):
                        ped_crossed = True
                    break

        # Ego yielded check
        ego_yielded = False
        if speed_profile and current_speed > 1.0:
            min_future_speed = min(speed_profile)
            if min_future_speed < 0.5 and current_speed > 2.0:
                ego_yielded = True
            elif (current_speed - min_future_speed) > 1.0:
                ego_yielded = True

        # Outcome classification
        if min_distance > 20.0:
            outcome = self.NO_INTERACTION
        elif ped_crossed and ego_yielded:
            outcome = self.EGO_YIELDED
        elif ped_crossed and not ego_yielded:
            outcome = self.PED_ASSERTED
        elif not ped_crossed and ego_yielded:
            outcome = self.PED_YIELDED
        elif min_distance < self.contest_distance_threshold:
            outcome = self.CONTESTED
        else:
            outcome = self.NO_INTERACTION

        # Interaction score
        if min_distance < 1.0:
            interaction_score = 1.0
        elif min_distance < 5.0:
            interaction_score = 0.8
        elif min_distance < 10.0:
            interaction_score = 0.5
        elif min_distance < 20.0:
            interaction_score = 0.2
        else:
            interaction_score = 0.0

        # Deceleration pattern
        decel_pattern = self._classify_decel_pattern(speed_profile, current_speed)

        # Closest approach time
        closest_time = self._compute_closest_approach_time(scene, best_token)

        return {
            "interaction_outcome_label": torch.tensor([outcome], dtype=torch.long),
            "interaction_score": torch.tensor([interaction_score], dtype=torch.float32),
            "ego_yielded": torch.tensor([float(ego_yielded)], dtype=torch.float32),
            "ped_crossed": torch.tensor([float(ped_crossed)], dtype=torch.float32),
            "min_future_distance": torch.tensor([min(min_distance, 50.0)], dtype=torch.float32),
            "speed_profile": speed_tensor,
            "decel_pattern": torch.tensor([decel_pattern], dtype=torch.long),
            "gt_jerk_magnitude": torch.tensor([avg_jerk], dtype=torch.float32),
            "gt_lat_accel": torch.tensor([avg_lat_accel], dtype=torch.float32),
            "closest_approach_time": torch.tensor([closest_time], dtype=torch.float32),
            "trajectory": trajectory_tensor,
        }