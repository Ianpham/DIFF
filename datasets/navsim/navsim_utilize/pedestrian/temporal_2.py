"""
Temporal Interaction Feature Builder (Stage 2)
===============================================
Stage 2, Days 13-17: Temporal Robustness

Upgrades from Stage 1's single-frame feature builder:
    1. State-difference encoding across all 4 history frames (2s at 2Hz)
    2. Temporal derivatives of pairwise features (Δdist, Δlateral, Δclosing_rate)
    3. Pedestrian track persistence validation
    4. Intent trajectory features (speed variance = hesitation signal)
    5. Ego history encoding (speed trend, deceleration pattern)

Key insight from our discussion: with only 4 frames at 2Hz, a learned temporal
encoder can't distinguish "decelerating" from "noise". Instead we explicitly
compute dynamics as handcrafted features — trading learned temporal features
for engineered ones because the data budget demands it.

Drops in as a replacement for PedestrianInteractionFeatureBuilder.
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



# Pedestrian track history extraction



def _extract_ped_history(
    annotations_list: List[Annotations],
    ego_statuses: List[EgoStatus],
    track_token: str,
) -> List[Optional[Dict[str, float]]]:
    """
    Extract per-frame state for a specific pedestrian across all history frames.
    Returns list of dicts (one per frame), None if ped not visible in that frame.
    All positions are in the ego frame of THAT frame (not current frame).
    """
    history = []
    for annot, ego in zip(annotations_list, ego_statuses):
        found = False
        for idx, name in enumerate(annot.names):
            if name == "pedestrian" and annot.track_tokens[idx] == track_token:
                box = annot.boxes[idx]
                vx = float(annot.velocity_3d[idx][0])
                vy = float(annot.velocity_3d[idx][1])
                speed = math.sqrt(vx**2 + vy**2)

                history.append({
                    "x": float(box[0]),
                    "y": float(box[1]),
                    "heading": float(box[6]) if len(box) > 6 else 0.0,
                    "vx": vx,
                    "vy": vy,
                    "speed": speed,
                    "length": float(box[3]) if len(box) > 3 else 0.5,
                    "width": float(box[4]) if len(box) > 4 else 0.5,
                })
                found = True
                break
        if not found:
            history.append(None)
    return history


def _compute_state_difference_features(
    ped_history: List[Optional[Dict[str, float]]],
) -> Dict[str, float]:
    """
    Compute the state-difference encoding from the pedestrian track history.
    This replaces temporal convolution — we explicitly compute the dynamics.

    From our discussion:
    - Current state at t₀: position, heading, velocity, size → 9 features
    - Velocity change (t₀ vs t₋₁): Δvx, Δvy → 2 features
    - Speed trend: speed(t₀) - speed(t₋₂) over 1.0s → 1 feature
    - Position displacement: cumulative from t₋₃ to t₀ → 2 features
    - Heading change: heading(t₀) - heading(t₋₂) → 1 feature
    - Speed variance: across all frames → 1 feature (hesitation signal)

    Total: 16 features
    """
    features = {}
    current = ped_history[-1] if ped_history else None

    if current is None:
        # No pedestrian at current frame → zero everything
        return {f"ped_temporal_{i}": 0.0 for i in range(16)}

    # Current state (9 features)
    features["ped_temporal_0"] = current["x"]
    features["ped_temporal_1"] = current["y"]
    features["ped_temporal_2"] = math.sin(current["heading"])
    features["ped_temporal_3"] = math.cos(current["heading"])
    features["ped_temporal_4"] = current["vx"]
    features["ped_temporal_5"] = current["vy"]
    features["ped_temporal_6"] = current["speed"]
    features["ped_temporal_7"] = current["length"]
    features["ped_temporal_8"] = current["width"]

    # Velocity change t₀ vs t₋₁ (2 features)
    prev = ped_history[-2] if len(ped_history) >= 2 else None
    if prev is not None:
        features["ped_temporal_9"] = current["vx"] - prev["vx"]
        features["ped_temporal_10"] = current["vy"] - prev["vy"]
    else:
        features["ped_temporal_9"] = 0.0
        features["ped_temporal_10"] = 0.0

    # Speed trend over 1.0s: t₀ vs t₋₂ (1 feature)
    prev2 = ped_history[-3] if len(ped_history) >= 3 else None
    if prev2 is not None:
        features["ped_temporal_11"] = current["speed"] - prev2["speed"]
    elif prev is not None:
        features["ped_temporal_11"] = current["speed"] - prev["speed"]
    else:
        features["ped_temporal_11"] = 0.0

    # Position displacement from t₋₃ to t₀ (2 features)
    oldest = None
    for h in ped_history:
        if h is not None:
            oldest = h
            break
    if oldest is not None and oldest is not current:
        features["ped_temporal_12"] = current["x"] - oldest["x"]
        features["ped_temporal_13"] = current["y"] - oldest["y"]
    else:
        features["ped_temporal_12"] = 0.0
        features["ped_temporal_13"] = 0.0

    # Heading change over 1.0s (1 feature)
    if prev2 is not None:
        dh = current["heading"] - prev2["heading"]
        # Normalize to [-pi, pi]
        dh = math.atan2(math.sin(dh), math.cos(dh))
        features["ped_temporal_14"] = dh
    else:
        features["ped_temporal_14"] = 0.0

    # Speed variance across all visible frames (1 feature — hesitation signal)
    speeds = [h["speed"] for h in ped_history if h is not None]
    if len(speeds) >= 2:
        features["ped_temporal_15"] = float(np.var(speeds))
    else:
        features["ped_temporal_15"] = 0.0

    return features


def _compute_temporal_pairwise_features(
    ped_history: List[Optional[Dict[str, float]]],
    ego_statuses: List[EgoStatus],
) -> Dict[str, float]:
    """
    Compute temporal derivatives of pairwise (ego-ped) features.

    From our discussion Stage 2 pairwise features:
    - Current pairwise: dx, dy, dist, closing_rate, bearing, lateral_offset,
      longitudinal_distance → 7 features
    - Temporal derivatives (t₀ vs t₋₁ or t₋₂):
      Δdist, Δlateral, Δlongitudinal, Δclosing_rate → 4 features
    - TTC estimate from closing rate and distance → 1 feature

    Total: 12 features
    """
    features = {}
    current_ped = ped_history[-1] if ped_history else None
    current_ego = ego_statuses[-1]

    if current_ped is None:
        return {f"pairwise_temporal_{i}": 0.0 for i in range(12)}

    # Current pairwise features
    dx = current_ped["x"]
    dy = current_ped["y"]
    dist = math.sqrt(dx**2 + dy**2)
    lateral = abs(dy)
    longitudinal = dx  # positive = ahead

    # Closing rate: rate of distance change using velocity
    ego_vx, ego_vy = current_ego.ego_velocity
    rel_vx = current_ped["vx"] - ego_vx  # not quite right (frame issues) but usable
    rel_vy = current_ped["vy"] - ego_vy

    if dist > 0.1:
        closing_rate = -(dx * rel_vx + dy * rel_vy) / dist  # positive = approaching
    else:
        closing_rate = 0.0

    bearing = math.atan2(dy, dx)

    features["pairwise_temporal_0"] = dx / 30.0  # normalized
    features["pairwise_temporal_1"] = dy / 10.0
    features["pairwise_temporal_2"] = dist / 30.0
    features["pairwise_temporal_3"] = closing_rate
    features["pairwise_temporal_4"] = math.sin(bearing)
    features["pairwise_temporal_5"] = math.cos(bearing)
    features["pairwise_temporal_6"] = lateral / 10.0

    # Temporal derivatives — compute pairwise at t₋₂ and diff
    prev_ped = None
    # Look back 2 frames (1.0s) for more robust derivative
    if len(ped_history) >= 3 and ped_history[-3] is not None:
        prev_ped = ped_history[-3]
        dt = 1.0  # 2 frames at 0.5s
    elif len(ped_history) >= 2 and ped_history[-2] is not None:
        prev_ped = ped_history[-2]
        dt = 0.5

    if prev_ped is not None:
        prev_dx = prev_ped["x"]
        prev_dy = prev_ped["y"]
        prev_dist = math.sqrt(prev_dx**2 + prev_dy**2)
        prev_lateral = abs(prev_dy)
        prev_longitudinal = prev_dx

        features["pairwise_temporal_7"] = (dist - prev_dist) / dt  # Δdist/Δt
        features["pairwise_temporal_8"] = (lateral - prev_lateral) / dt  # Δlateral/Δt
        features["pairwise_temporal_9"] = (longitudinal - prev_longitudinal) / dt  # Δlongitudinal/Δt

        # Δclosing_rate (rough: just use dist derivative)
        features["pairwise_temporal_10"] = features["pairwise_temporal_7"]  # same as dist rate
    else:
        features["pairwise_temporal_7"] = 0.0
        features["pairwise_temporal_8"] = 0.0
        features["pairwise_temporal_9"] = 0.0
        features["pairwise_temporal_10"] = 0.0

    # TTC estimate
    if closing_rate > 0.5:  # approaching at > 0.5 m/s
        ttc = dist / closing_rate
        features["pairwise_temporal_11"] = min(ttc, 10.0) / 10.0  # normalized, capped at 10s
    else:
        features["pairwise_temporal_11"] = 1.0  # no collision imminent

    return features


def _compute_ego_history_features(ego_statuses: List[EgoStatus]) -> Dict[str, float]:
    """
    Ego temporal features from history.

    - Current: vx, vy, ax, ay, speed → 5 features
    - Speed trend: speed(t₀) - speed(t₋₂) → 1 feature
    - Is decelerating: binary from ax/speed alignment → 1 feature
    - Driving command → 3 features
    Total: 10 features
    """
    features = {}
    current = ego_statuses[-1]
    vx, vy = current.ego_velocity
    ax, ay = current.ego_acceleration
    speed = math.sqrt(vx**2 + vy**2)

    features["ego_hist_0"] = vx
    features["ego_hist_1"] = vy
    features["ego_hist_2"] = ax
    features["ego_hist_3"] = ay
    features["ego_hist_4"] = speed

    # Speed trend over 1.0s
    if len(ego_statuses) >= 3:
        prev_ego = ego_statuses[-3]
        prev_speed = math.sqrt(prev_ego.ego_velocity[0]**2 + prev_ego.ego_velocity[1]**2)
        features["ego_hist_5"] = speed - prev_speed
    elif len(ego_statuses) >= 2:
        prev_ego = ego_statuses[-2]
        prev_speed = math.sqrt(prev_ego.ego_velocity[0]**2 + prev_ego.ego_velocity[1]**2)
        features["ego_hist_5"] = speed - prev_speed
    else:
        features["ego_hist_5"] = 0.0

    # Is decelerating (longitudinal decel)
    if speed > 0.5:
        # Project acceleration onto velocity direction
        lon_accel = (vx * ax + vy * ay) / speed
        features["ego_hist_6"] = float(lon_accel < -0.3)
    else:
        features["ego_hist_6"] = 0.0

    # Driving command
    cmd = current.driving_command
    features["ego_hist_7"] = cmd[0]
    features["ego_hist_8"] = cmd[1]
    features["ego_hist_9"] = cmd[2] if len(cmd) > 2 else 0.0

    return features



# Main Feature Builder


class TemporalInteractionFeatureBuilder(AbstractFeatureBuilder):
    """
    Stage 2 feature builder with full temporal encoding.

    Replaces Stage 1's PedestrianInteractionFeatureBuilder.

    Output features:
        "ped_temporal_features": Tensor [16]  — state-difference encoding
        "pairwise_temporal_features": Tensor [12]  — temporal pairwise derivatives
        "ego_history_features": Tensor [10]  — ego dynamics over history
        "ped_risk_field": Tensor [1, 32, 32]  — spatial risk field (reused from S1)
        "has_relevant_pedestrian": Tensor [1]
        "ped_track_quality": Tensor [1]  — fraction of frames ped was visible (0-1)
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
        return "temporal_interaction_feature"

    def _select_best_ped_token(
        self, annotations: Annotations, ego_status: EgoStatus
    ) -> Optional[str]:
        """Select the best pedestrian track_token at the current frame."""
        best_token = None
        best_score = float("inf")

        for idx, name in enumerate(annotations.names):
            if name != "pedestrian":
                continue
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])
            dist = math.sqrt(px**2 + py**2)
            if dist > self.max_distance:
                continue

            in_corridor = 0 < px < self.corridor_length and abs(py) < self.corridor_width
            score = dist + (0 if in_corridor else 100.0)

            if score < best_score:
                best_score = score
                best_token = annotations.track_tokens[idx]

        return best_token

    def _build_risk_field(self, annotations: Annotations) -> np.ndarray:
        """Build spatial risk field — same as Stage 1 but slightly improved."""
        grid = np.zeros((self.risk_field_size, self.risk_field_size), dtype=np.float32)
        cell_size = (2 * self.risk_field_range) / self.risk_field_size
        half_grid = self.risk_field_size // 2

        for idx, name in enumerate(annotations.names):
            if name != "pedestrian":
                continue
            box = annotations.boxes[idx]
            px, py = float(box[0]), float(box[1])
            vx = float(annotations.velocity_3d[idx][0])
            vy = float(annotations.velocity_3d[idx][1])
            speed = math.sqrt(vx**2 + vy**2)
            dist = math.sqrt(px**2 + py**2)

            if dist > self.risk_field_range * 1.5:
                continue

            risk_mag = max(0.1, 1.0 / max(dist, 1.0)) * (1.0 + speed)
            sigma = max(2.0, speed * 1.5)

            # Current position blob
            gi = half_grid - int(round(py / cell_size))
            gj = half_grid + int(round(px / cell_size))
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    r, c = gi + di, gj + dj
                    if 0 <= r < self.risk_field_size and 0 <= c < self.risk_field_size:
                        d2 = (di * cell_size)**2 + (dj * cell_size)**2
                        grid[r, c] += risk_mag * math.exp(-d2 / (2 * sigma**2))

            # Future positions along velocity
            for t in [0.5, 1.0, 1.5, 2.0]:
                fp_x = px + vx * t
                fp_y = py + vy * t
                fi = half_grid - int(round(fp_y / cell_size))
                fj = half_grid + int(round(fp_x / cell_size))
                if 0 <= fi < self.risk_field_size and 0 <= fj < self.risk_field_size:
                    grid[fi, fj] += risk_mag * math.exp(-t * 0.5) * 0.5

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val
        return grid

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """Compute full temporal interaction features."""
        current_annot = agent_input.annotations[-1]
        current_ego = agent_input.ego_statuses[-1]

        # Select best pedestrian
        best_token = self._select_best_ped_token(current_annot, current_ego)
        has_ped = best_token is not None

        if has_ped:
            # Extract full history for this pedestrian
            ped_history = _extract_ped_history(
                agent_input.annotations,
                agent_input.ego_statuses,
                best_token,
            )

            # Track quality: fraction of frames visible
            visible_frames = sum(1 for h in ped_history if h is not None)
            track_quality = visible_frames / max(len(ped_history), 1)

            # State-difference encoding (16 features)
            temporal_feats = _compute_state_difference_features(ped_history)

            # Temporal pairwise features (12 features)
            pairwise_feats = _compute_temporal_pairwise_features(
                ped_history, agent_input.ego_statuses
            )
        else:
            track_quality = 0.0
            temporal_feats = {f"ped_temporal_{i}": 0.0 for i in range(16)}
            pairwise_feats = {f"pairwise_temporal_{i}": 0.0 for i in range(12)}

        # Ego history features (10 features) — always available
        ego_feats = _compute_ego_history_features(agent_input.ego_statuses)

        # Risk field (all pedestrians)
        risk_field = self._build_risk_field(current_annot)

        # Assemble tensors
        ped_temporal_tensor = torch.tensor(
            [temporal_feats[f"ped_temporal_{i}"] for i in range(16)],
            dtype=torch.float32,
        )
        pairwise_temporal_tensor = torch.tensor(
            [pairwise_feats[f"pairwise_temporal_{i}"] for i in range(12)],
            dtype=torch.float32,
        )
        ego_history_tensor = torch.tensor(
            [ego_feats[f"ego_hist_{i}"] for i in range(10)],
            dtype=torch.float32,
        )

        return {
            "ped_temporal_features": ped_temporal_tensor,          # [16]
            "pairwise_temporal_features": pairwise_temporal_tensor, # [12]
            "ego_history_features": ego_history_tensor,             # [10]
            "ped_risk_field": torch.tensor(risk_field, dtype=torch.float32).unsqueeze(0),  # [1, 32, 32]
            "has_relevant_pedestrian": torch.tensor([float(has_ped)], dtype=torch.float32),
            "ped_track_quality": torch.tensor([track_quality], dtype=torch.float32),
        }