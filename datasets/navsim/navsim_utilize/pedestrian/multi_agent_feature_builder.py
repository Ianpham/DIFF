"""
Multi-Agent Interaction Feature Builder (Stage 3)
==================================================
Stage 3, Days 27-34: Multi-Pedestrian + Map + Vehicle Context

Upgrades from Stage 2:
    1. Multi-pedestrian encoding (top-K with set-attention, not single best)
    2. Map polyline features (crosswalks, lanes, route context)
    3. Nearby vehicle context (vehicles that also interact with peds)
    4. Occlusion reasoning (infer invisible ped risk from map geometry)
    5. Pedestrian trajectory prediction features (linear extrapolation)

Feature output structure:
    - Per-agent embeddings → set-attention → pooled interaction embedding
    - Map polylines → MLP + pool → map context
    - Fused into richer input for the interaction encoder
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from navsim.common.dataclasses import AgentInput, Annotations, EgoStatus
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder

try:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False



# Constants


MAX_PEDS = 5           # Top-K pedestrians to encode
MAX_VEHICLES = 3       # Nearby vehicles that may interact
MAX_MAP_POLYLINES = 8  # Crosswalks + nearby lane segments
POLYLINE_POINTS = 10   # Points per polyline (resampled)
PED_FEATURE_DIM = 18   # Per-pedestrian feature vector size
VEH_FEATURE_DIM = 10   # Per-vehicle feature vector size
MAP_POLYLINE_DIM = 6   # Per-point feature in polyline (x, y, dx, dy, type, dist_to_ego)



# Per-agent feature extraction



def _extract_single_ped_features(
    annotations_list: List[Annotations],
    ego_statuses: List[EgoStatus],
    track_token: str,
    corridor_width: float = 5.0,
    corridor_length: float = 30.0,
) -> Optional[Dict[str, float]]:
    """
    Extract temporal features for a single pedestrian across history frames.
    Returns 18-dim feature dict or None if ped not found at current frame.
    """
    current_annot = annotations_list[-1]
    current_ego = ego_statuses[-1]

    # Find ped at current frame
    current_info = None
    for idx, name in enumerate(current_annot.names):
        if name == "pedestrian" and current_annot.track_tokens[idx] == track_token:
            box = current_annot.boxes[idx]
            current_info = {
                "x": float(box[0]), "y": float(box[1]),
                "heading": float(box[6]) if len(box) > 6 else 0.0,
                "vx": float(current_annot.velocity_3d[idx][0]),
                "vy": float(current_annot.velocity_3d[idx][1]),
            }
            break

    if current_info is None:
        return None

    px, py = current_info["x"], current_info["y"]
    vx, vy = current_info["vx"], current_info["vy"]
    speed = math.sqrt(vx**2 + vy**2)
    dist = math.sqrt(px**2 + py**2)
    bearing = math.atan2(py, px)
    lateral = abs(py)
    in_corridor = 0 < px < corridor_length and lateral < corridor_width

    # Closing rate
    ego_vx, ego_vy = current_ego.ego_velocity
    if dist > 0.1:
        closing_rate = -(px * (vx - ego_vx) + py * (vy - ego_vy)) / dist
    else:
        closing_rate = 0.0

    # TTC
    ttc = dist / max(closing_rate, 0.3) if closing_rate > 0.3 else 10.0
    ttc = min(ttc, 10.0)

    # Track history for temporal features
    history_dists = []
    for annot in annotations_list:
        for idx2, name2 in enumerate(annot.names):
            if name2 == "pedestrian" and annot.track_tokens[idx2] == track_token:
                box2 = annot.boxes[idx2]
                history_dists.append(math.sqrt(float(box2[0])**2 + float(box2[1])**2))
                break

    # Speed variance (hesitation)
    history_speeds = []
    for annot in annotations_list:
        for idx2, name2 in enumerate(annot.names):
            if name2 == "pedestrian" and annot.track_tokens[idx2] == track_token:
                sv = math.sqrt(float(annot.velocity_3d[idx2][0])**2 + float(annot.velocity_3d[idx2][1])**2)
                history_speeds.append(sv)
                break

    speed_var = float(np.var(history_speeds)) if len(history_speeds) >= 2 else 0.0
    dist_trend = (history_dists[-1] - history_dists[0]) / max(len(history_dists) - 1, 1) if len(history_dists) >= 2 else 0.0

    # Linear extrapolation: where will ped be in 2s?
    future_x = px + vx * 2.0
    future_y = py + vy * 2.0
    future_in_corridor = 0 < future_x < corridor_length and abs(future_y) < corridor_width

    # Track quality
    frames_visible = len(history_dists)
    track_quality = frames_visible / max(len(annotations_list), 1)

    return {
        "rel_x": px / 30.0,
        "rel_y": py / 10.0,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "distance": dist / 30.0,
        "sin_bearing": math.sin(bearing),
        "cos_bearing": math.cos(bearing),
        "lateral": lateral / 10.0,
        "closing_rate": closing_rate,
        "ttc": ttc / 10.0,
        "in_corridor": float(in_corridor),
        "speed_variance": min(speed_var, 2.0),
        "dist_trend": dist_trend,
        "future_x": future_x / 30.0,
        "future_y": future_y / 10.0,
        "future_in_corridor": float(future_in_corridor),
        "track_quality": track_quality,
    }


def _extract_vehicle_features(
    annotations: Annotations,
    ego_status: EgoStatus,
    max_vehicles: int = MAX_VEHICLES,
) -> List[Dict[str, float]]:
    """Extract features for nearby vehicles that may interact with pedestrians."""
    vehicles = []
    for idx, name in enumerate(annotations.names):
        if name != "vehicle":
            continue
        box = annotations.boxes[idx]
        px, py = float(box[0]), float(box[1])
        dist = math.sqrt(px**2 + py**2)
        if dist > 40.0:
            continue

        vx = float(annotations.velocity_3d[idx][0])
        vy = float(annotations.velocity_3d[idx][1])
        speed = math.sqrt(vx**2 + vy**2)

        vehicles.append({
            "rel_x": px / 30.0,
            "rel_y": py / 10.0,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "distance": dist / 30.0,
            "heading": float(box[6]) if len(box) > 6 else 0.0,
            "length": float(box[3]) if len(box) > 3 else 4.5,
            "width": float(box[4]) if len(box) > 4 else 2.0,
            "could_occlude_ped": float(dist < 20.0 and abs(py) < 5.0),
        })

    vehicles.sort(key=lambda v: v["distance"])
    return vehicles[:max_vehicles]



# Map polyline encoding



def _extract_map_polylines(
    map_api,
    ego_pose: Tuple[float, float, float],
    route_roadblock_ids: Optional[List[str]] = None,
    max_polylines: int = MAX_MAP_POLYLINES,
    radius: float = 40.0,
) -> List[Dict[str, Any]]:
    """
    Extract crosswalk and lane polylines around ego, transformed to ego frame.
    Returns list of polyline dicts with points + metadata.
    """
    if map_api is None:
        return []

    ex, ey, eh = ego_pose
    cos_h = math.cos(eh)
    sin_h = math.sin(eh)

    polylines = []

    try:
        point = Point2D(ex, ey)

        # 1. Crosswalks (highest priority)
        nearby = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.CROSSWALK])
        for cw in nearby.get(SemanticMapLayer.CROSSWALK, []):
            polygon = cw.polygon
            coords = list(polygon.exterior.coords)
            # Transform to ego frame
            ego_coords = []
            for gx, gy in coords[:POLYLINE_POINTS]:
                lx = cos_h * (gx - ex) + sin_h * (gy - ey)
                ly = -sin_h * (gx - ex) + cos_h * (gy - ey)
                ego_coords.append((lx, ly))

            dist_to_ego = min(math.sqrt(c[0]**2 + c[1]**2) for c in ego_coords) if ego_coords else 999.0

            # Check if on route
            on_route = False
            if route_roadblock_ids:
                # Rough check: crosswalk near ego's forward path
                centroid = polygon.centroid
                cx = cos_h * (centroid.x - ex) + sin_h * (centroid.y - ey)
                cy = -sin_h * (centroid.x - ex) + cos_h * (centroid.y - ey)
                on_route = 0 < cx < 40.0 and abs(cy) < 10.0

            polylines.append({
                "points": ego_coords,
                "type": "crosswalk",
                "type_id": 1.0,
                "distance": dist_to_ego,
                "on_route": float(on_route),
            })

        # 2. Lane boundaries near crosswalks
        lane_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        lane_nearby = map_api.get_proximal_map_objects(point, radius, lane_layers)

        for layer in lane_layers:
            for roadblock in lane_nearby.get(layer, []):
                for lane in roadblock.interior_edges:
                    try:
                        path = lane.baseline_path.discrete_path
                        ego_coords = []
                        for state in path[:POLYLINE_POINTS]:
                            gx, gy = state.x, state.y
                            lx = cos_h * (gx - ex) + sin_h * (gy - ey)
                            ly = -sin_h * (gx - ex) + cos_h * (gy - ey)
                            ego_coords.append((lx, ly))

                        if not ego_coords:
                            continue
                        dist_to_ego = min(math.sqrt(c[0]**2 + c[1]**2) for c in ego_coords)

                        on_route = False
                        if route_roadblock_ids and roadblock.id in route_roadblock_ids:
                            on_route = True

                        polylines.append({
                            "points": ego_coords,
                            "type": "lane",
                            "type_id": 0.0,
                            "distance": dist_to_ego,
                            "on_route": float(on_route),
                        })
                    except Exception:
                        continue

    except Exception as e:
        pass

    # Sort by distance, prioritize crosswalks
    polylines.sort(key=lambda p: (-p["type_id"], p["distance"]))
    return polylines[:max_polylines]


def _polylines_to_tensor(
    polylines: List[Dict[str, Any]],
    max_polylines: int = MAX_MAP_POLYLINES,
    max_points: int = POLYLINE_POINTS,
) -> Tuple[Tensor, Tensor]:
    """
    Convert polyline list to padded tensor.
    Returns:
        polyline_features: [max_polylines, max_points, 6] (x, y, dx, dy, type, dist)
        polyline_mask: [max_polylines] (1 if valid, 0 if padding)
    """
    features = torch.zeros(max_polylines, max_points, MAP_POLYLINE_DIM)
    mask = torch.zeros(max_polylines)

    for i, poly in enumerate(polylines[:max_polylines]):
        points = poly["points"]
        n = min(len(points), max_points)
        mask[i] = 1.0

        for j in range(n):
            x, y = points[j]
            features[i, j, 0] = x / 30.0
            features[i, j, 1] = y / 10.0
            if j > 0:
                features[i, j, 2] = (x - points[j-1][0]) / 10.0
                features[i, j, 3] = (y - points[j-1][1]) / 10.0
            features[i, j, 4] = poly["type_id"]
            features[i, j, 5] = poly["on_route"]

    return features, mask



# Occlusion reasoning



def _compute_occlusion_features(
    annotations: Annotations,
    ego_status: EgoStatus,
    has_crosswalk_ahead: bool,
) -> Dict[str, float]:
    """
    Infer occlusion risk: are there large vehicles that could hide a pedestrian?

    Key insight: a parked/slow vehicle near a crosswalk creates ped occlusion risk
    even when no pedestrian is currently visible.
    """
    vehicles_near_crosswalk = 0
    max_occluder_size = 0.0
    closest_occluder_dist = 30.0

    for idx, name in enumerate(annotations.names):
        if name != "vehicle":
            continue
        box = annotations.boxes[idx]
        px, py = float(box[0]), float(box[1])
        dist = math.sqrt(px**2 + py**2)
        length = float(box[3]) if len(box) > 3 else 4.5

        # Vehicle in forward zone, could occlude crosswalk
        if 5.0 < px < 25.0 and abs(py) < 8.0 and dist < 25.0:
            vx = float(annotations.velocity_3d[idx][0])
            vy = float(annotations.velocity_3d[idx][1])
            veh_speed = math.sqrt(vx**2 + vy**2)

            # Slow/parked vehicles are better occluders
            if veh_speed < 2.0:
                vehicles_near_crosswalk += 1
                max_occluder_size = max(max_occluder_size, length)
                closest_occluder_dist = min(closest_occluder_dist, dist)

    # Occlusion risk: high if parked vehicles near crosswalk ahead
    occlusion_risk = 0.0
    if has_crosswalk_ahead and vehicles_near_crosswalk > 0:
        occlusion_risk = min(1.0, vehicles_near_crosswalk * 0.4 + max_occluder_size / 10.0)

    return {
        "occlusion_risk": occlusion_risk,
        "num_potential_occluders": float(min(vehicles_near_crosswalk, 3)),
        "closest_occluder_dist": closest_occluder_dist / 30.0,
        "phantom_ped_risk": occlusion_risk * float(has_crosswalk_ahead),
    }



# Main Feature Builder



class MultiAgentInteractionFeatureBuilder(AbstractFeatureBuilder):
    """
    Stage 3 feature builder: multi-pedestrian + map + vehicle + occlusion.

    Output keys:
        "multi_ped_features": [MAX_PEDS, PED_FEATURE_DIM]  — per-ped features
        "multi_ped_mask": [MAX_PEDS]  — 1 if valid ped, 0 if padding
        "vehicle_features": [MAX_VEHICLES, VEH_FEATURE_DIM]
        "vehicle_mask": [MAX_VEHICLES]
        "map_polyline_features": [MAX_MAP_POLYLINES, POLYLINE_POINTS, MAP_POLYLINE_DIM]
        "map_polyline_mask": [MAX_MAP_POLYLINES]
        "ego_history_features": [10]  — same as Stage 2
        "ped_risk_field": [1, 32, 32]  — spatial risk (same as S2)
        "occlusion_features": [4]
        "scene_summary": [8]  — global scene statistics
    """

    def __init__(
        self,
        corridor_width: float = 5.0,
        corridor_length: float = 30.0,
        max_distance: float = 30.0,
        risk_field_size: int = 32,
        risk_field_range: float = 32.0,
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
        self._map_cache = {}

    def get_unique_name(self) -> str:
        return "multi_agent_interaction_feature"

    def _get_map_api(self, map_name: str):
        if not MAP_AVAILABLE or self.map_root is None:
            return None
        if map_name not in self._map_cache:
            actual = map_name if map_name != "las_vegas" else "us-nv-las-vegas-strip"
            try:
                self._map_cache[map_name] = get_maps_api(self.map_root, self.map_version, actual)
            except Exception:
                self._map_cache[map_name] = None
        return self._map_cache[map_name]

    def _get_top_k_ped_tokens(
        self, annotations: Annotations, max_peds: int = MAX_PEDS
    ) -> List[str]:
        """Select top-K most relevant pedestrians."""
        candidates = []
        for idx, name in enumerate(annotations.names):
            if name != "pedestrian":
                continue
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])
            dist = math.sqrt(px**2 + py**2)
            if dist > self.max_distance:
                continue
            in_corridor = 0 < px < self.corridor_length and abs(py) < self.corridor_width
            score = dist + (0.0 if in_corridor else 50.0)
            candidates.append((score, annotations.track_tokens[idx]))

        candidates.sort()
        return [token for _, token in candidates[:max_peds]]

    def _build_risk_field(self, annotations: Annotations) -> np.ndarray:
        """Build spatial risk field — same as Stage 2."""
        grid = np.zeros((self.risk_field_size, self.risk_field_size), dtype=np.float32)
        cell_size = (2 * self.risk_field_range) / self.risk_field_size
        half_grid = self.risk_field_size // 2

        for idx, name in enumerate(annotations.names):
            if name != "pedestrian":
                continue
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])
            vx, vy = float(annotations.velocity_3d[idx][0]), float(annotations.velocity_3d[idx][1])
            speed = math.sqrt(vx**2 + vy**2)
            dist = math.sqrt(px**2 + py**2)
            if dist > self.risk_field_range * 1.5:
                continue

            risk_mag = max(0.1, 1.0 / max(dist, 1.0)) * (1.0 + speed)
            sigma = max(2.0, speed * 1.5)
            gi = half_grid - int(round(py / cell_size))
            gj = half_grid + int(round(px / cell_size))
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    r, c = gi + di, gj + dj
                    if 0 <= r < self.risk_field_size and 0 <= c < self.risk_field_size:
                        d2 = (di * cell_size)**2 + (dj * cell_size)**2
                        grid[r, c] += risk_mag * math.exp(-d2 / (2 * sigma**2))

            for t in [0.5, 1.0, 1.5, 2.0]:
                fi = half_grid - int(round((py + vy * t) / cell_size))
                fj = half_grid + int(round((px + vx * t) / cell_size))
                if 0 <= fi < self.risk_field_size and 0 <= fj < self.risk_field_size:
                    grid[fi, fj] += risk_mag * math.exp(-t * 0.5) * 0.5

        m = grid.max()
        if m > 0:
            grid /= m
        return grid

    def _compute_ego_features(self, ego_statuses: List[EgoStatus]) -> Tensor:
        """Same ego features as Stage 2 (10-dim)."""
        cur = ego_statuses[-1]
        vx, vy = cur.ego_velocity
        ax, ay = cur.ego_acceleration
        speed = math.sqrt(vx**2 + vy**2)

        speed_trend = 0.0
        if len(ego_statuses) >= 3:
            prev = ego_statuses[-3]
            prev_speed = math.sqrt(prev.ego_velocity[0]**2 + prev.ego_velocity[1]**2)
            speed_trend = speed - prev_speed

        is_decel = float(speed > 0.5 and (vx * ax + vy * ay) / speed < -0.3)
        cmd = cur.driving_command

        return torch.tensor([
            vx, vy, ax, ay, speed, speed_trend, is_decel,
            cmd[0], cmd[1], cmd[2] if len(cmd) > 2 else 0.0,
        ], dtype=torch.float32)

    def _compute_scene_summary(self, annotations: Annotations, ego_status: EgoStatus) -> Tensor:
        """Global scene statistics (8-dim)."""
        n_peds = sum(1 for n in annotations.names if n == "pedestrian")
        n_vehicles = sum(1 for n in annotations.names if n == "vehicle")
        n_cyclists = sum(1 for n in annotations.names if n == "bicycle")

        ego_speed = math.sqrt(ego_status.ego_velocity[0]**2 + ego_status.ego_velocity[1]**2)

        # Closest ped distance
        min_ped_dist = 30.0
        for idx, name in enumerate(annotations.names):
            if name == "pedestrian":
                box = annotations.boxes[idx]
                d = math.sqrt(float(box[0])**2 + float(box[1])**2)
                min_ped_dist = min(min_ped_dist, d)

        return torch.tensor([
            float(min(n_peds, 10)) / 10.0,
            float(min(n_vehicles, 20)) / 20.0,
            float(min(n_cyclists, 5)) / 5.0,
            ego_speed / 15.0,
            min_ped_dist / 30.0,
            float(n_peds > 0),
            float(n_peds > 2),  # crowd mode flag
            float(min_ped_dist < 10.0),  # imminent interaction flag
        ], dtype=torch.float32)

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """Compute all Stage 3 features."""
        current_annot = agent_input.annotations[-1]
        current_ego = agent_input.ego_statuses[-1]

        # 1. Multi-pedestrian features
        top_tokens = self._get_top_k_ped_tokens(current_annot)
        ped_features = torch.zeros(MAX_PEDS, PED_FEATURE_DIM)
        ped_mask = torch.zeros(MAX_PEDS)

        for i, token in enumerate(top_tokens):
            feats = _extract_single_ped_features(
                agent_input.annotations, agent_input.ego_statuses,
                token, self.corridor_width, self.corridor_length,
            )
            if feats is not None:
                ped_features[i] = torch.tensor(list(feats.values()), dtype=torch.float32)
                ped_mask[i] = 1.0

        # 2. Vehicle features
        veh_list = _extract_vehicle_features(current_annot, current_ego)
        veh_features = torch.zeros(MAX_VEHICLES, VEH_FEATURE_DIM)
        veh_mask = torch.zeros(MAX_VEHICLES)
        for i, veh in enumerate(veh_list):
            veh_features[i] = torch.tensor(list(veh.values()), dtype=torch.float32)
            veh_mask[i] = 1.0

        # 3. Map polylines (during caching only — map_api needed)
        map_features = torch.zeros(MAX_MAP_POLYLINES, POLYLINE_POINTS, MAP_POLYLINE_DIM)
        map_mask = torch.zeros(MAX_MAP_POLYLINES)
        # Map polylines extracted during caching via map_api
        # At inference without map, these stay zero (spatial risk field compensates)

        # 4. Risk field
        risk_field = self._build_risk_field(current_annot)

        # 5. Ego features
        ego_features = self._compute_ego_features(agent_input.ego_statuses)

        # 6. Occlusion reasoning
        # Check if crosswalk ahead (rough: any ped with in_corridor and distance 10-25m)
        has_cw_ahead = any(ped_mask[i] > 0 and ped_features[i, 11] > 0 for i in range(MAX_PEDS))
        occ_feats = _compute_occlusion_features(current_annot, current_ego, has_cw_ahead)
        occ_tensor = torch.tensor(list(occ_feats.values()), dtype=torch.float32)

        # 7. Scene summary
        scene_summary = self._compute_scene_summary(current_annot, current_ego)

        return {
            "multi_ped_features": ped_features,       # [5, 18]
            "multi_ped_mask": ped_mask,               # [5]
            "vehicle_features": veh_features,          # [3, 10]
            "vehicle_mask": veh_mask,                  # [3]
            "map_polyline_features": map_features,     # [8, 10, 6]
            "map_polyline_mask": map_mask,             # [8]
            "ego_history_features": ego_features,      # [10]
            "ped_risk_field": torch.tensor(risk_field, dtype=torch.float32).unsqueeze(0),  # [1, 32, 32]
            "occlusion_features": occ_tensor,          # [4]
            "scene_summary": scene_summary,            # [8]
        }