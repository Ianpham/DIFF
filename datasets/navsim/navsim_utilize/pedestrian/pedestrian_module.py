"""
Pedestrian Scene Miner for NavSim
==================================
Data Mining & Scene Filtering

This script scans the entire NavSim training set and produces a JSON index
telling you exactly which scenes contain meaningful pedestrian interactions.

It answers the critical go/no-go question:
    "How many pedestrian interaction scenes does NavSim actually have?"

If < 500 interactive scenes  → need to relax criteria or augment
If 2000+                     → proceed with full architecture

Also validates:
    - track_token persistence across history frames
    - velocity_3d plausibility for pedestrians
    - crosswalk geometry availability from map API

Usage:
    python pedestrian_scene_miner.py \
        --data_root /home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download \
        --map_root  /home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps \
        --output_path ./ped_interaction_index.json
"""
import argparse
import json
import logging
import os
import time
import warnings

from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import asdict, dataclass, field

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

# navsim imports
from navsim.common.dataclasses import (
    Annotations,
    EgoStatus,
    Scene,
    SceneFilter,
    SensorConfig,
)

from navsim.common.dataloader import SceneLoader

# nuPlan map imports (for crosswalk queries)
try:
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False
    warnings.warn("nuPlan map API not available. Crosswalk features will be disabled.")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default paths — mirrors NavsimDataset conventions, no env vars needed
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path(
    "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download"
)
DEFAULT_MAP_ROOT = DEFAULT_DATA_ROOT / "maps"
DEFAULT_NAVSIM_LOGS = DEFAULT_DATA_ROOT / "mini_navsim_logs" / "mini"
DEFAULT_OUTPUT = Path("./ped_interaction_index.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PedestrianInfo:
    """Info about a single pedestrian in a scene."""
    track_token: str
    distance_to_ego: float
    rel_x: float            # in ego frame: positive = ahead
    rel_y: float            # in ego frame: positive = left
    speed: float            # magnitude of velocity
    vx: float
    vy: float
    bearing_angle: float    # angle from ego forward axis to ped
    is_in_forward_corridor: bool  # within ±5m lateral, 0-30m ahead
    lateral_offset: float   # perpendicular distance from ego's forward axis
    heading_alignment: float  # dot product of ped velocity and ego-to-ped vector

    # Track persistence (filled during temporal analysis)
    frames_visible: int = 0
    track_persistent: bool = False  # same track_token across history frames

    # Map context (filled if map available)
    on_crosswalk: bool = False
    near_crosswalk: bool = False    # within crosswalk_radius of a crosswalk
    crosswalk_distance: float = float("inf")


@dataclass
class SceneInteractionInfo:
    """Complete interaction analysis for one scene."""

    token: str
    log_name: str
    map_name: str

    # Pedestrian counts
    total_pedestrians: int = 0
    pedestrians_in_corridor: int = 0    # in ego's forward corridor (±5m, 0-30m)
    pedestrians_near_crosswalk: int = 0

    # Interaction classification
    has_pedestrian_interaction: bool = False
    interaction_strength: str = "none"  # none, weak, moderate, strong

    # Best candidate pedestrian (closest to forward path near crosswalk)
    best_ped_distance: float = float("inf")
    best_ped_lateral_offset: float = float("inf")
    best_ped_speed: float = 0.0
    best_ped_track_token: str = ""
    best_ped_on_crosswalk: bool = False

    # Track quality metrics
    ped_track_persistence_rate: float = 0.0
    avg_ped_velocity_magnitude: float = 0.0

    # Ego state at current frame
    ego_speed: float = 0.0
    ego_acceleration_magnitude: float = 0.0

    # Map context
    has_crosswalk_ahead: bool = False
    crosswalk_distance_ahead: float = float("inf")
    has_intersection: bool = False

    # Velocity quality check
    ped_velocity_plausible: bool = True
    num_implausible_velocities: int = 0

    # Per-pedestrian details (for debugging, optional)
    pedestrian_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetSummary:
    """Summary statistics for the full dataset."""

    total_scenes: int = 0
    scenes_with_pedestrians: int = 0
    scenes_with_corridor_peds: int = 0
    scenes_with_crosswalk_peds: int = 0
    scenes_with_interactions: int = 0

    interaction_strength_counts: Dict[str, int] = field(default_factory=lambda: Counter())

    avg_peds_per_scene: float = 0.0
    max_peds_in_scene: int = 0

    track_persistence_rate: float = 0.0
    velocity_plausibility_rate: float = 0.0

    # Per-map breakdown
    per_map_counts: Dict[str, int] = field(default_factory=lambda: Counter())
    per_map_interactions: Dict[str, int] = field(default_factory=lambda: Counter())


# ---------------------------------------------------------------------------
# Core mining class
# ---------------------------------------------------------------------------

class PedestrianSceneMiner:
    """Mines NavSim scenes for pedestrian interactions."""

    def __init__(
            self,
            data_path: Path,
            map_root: Optional[Path] = None,
            corridor_width: float = 5.0,
            corridor_length: float = 30.0,
            crosswalk_radius: float = 50.0,
            min_interaction_distance: float = 20.0,
            max_plausible_ped_speed: float = 6.0,
    ):
        self.data_path = Path(data_path)
        self.map_root = Path(map_root) if map_root is not None else None
        self.corridor_width = corridor_width
        self.corridor_length = corridor_length
        self.crosswalk_radius = crosswalk_radius
        self.min_interaction_distance = min_interaction_distance
        self.max_plausible_ped_speed = max_plausible_ped_speed

        # map api cache (avoid reloading for same map)
        self._map_cache: Dict[str, Any] = {}

        logger.info(f"PedestrianSceneMiner init")
        logger.info(f"  data_path : {self.data_path}")
        logger.info(f"  map_root  : {self.map_root}")

    # ------------------------------------------------------------------
    # Map helpers
    # ------------------------------------------------------------------

    def _get_map_api(self, map_name: str):
        """Get or create map API for a given map name, with caching."""
        if not MAP_AVAILABLE or self.map_root is None:
            return None

        if map_name not in self._map_cache:
            actual_name = map_name if map_name != "las_vegas" else "us-nv-las-strip"
            try:
                map_api = get_maps_api(
                    str(self.map_root),
                    "nuplan-maps-v1.0",
                    actual_name,
                )
                self._map_cache[map_name] = map_api
                logger.debug(f"Loaded map: {actual_name}")
            except Exception as e:
                logger.warning(f"Failed to load map {actual_name}: {e}")
                self._map_cache[map_name] = None

        return self._map_cache[map_name]

    def _query_crosswalks_near_point(
            self,
            map_api,
            x: float,
            y: float,
            radius: float,
    ) -> List[Dict[str, Any]]:
        """Query crosswalk polygons near a global point."""
        if map_api is None:
            return []

        try:
            point = Point2D(x, y)
            nearby = map_api.get_proximal_map_objects(
                point, radius, [SemanticMapLayer.CROSSWALK]
            )
            crosswalks = nearby.get(SemanticMapLayer.CROSSWALK, [])
            results = []
            for cw in crosswalks:
                polygon = cw.polygon
                centroid = polygon.centroid
                dist = np.sqrt((centroid.x - x) ** 2 + (centroid.y - y) ** 2)
                results.append({
                    "id": cw.id,
                    "centroid_x": centroid.x,
                    "centroid_y": centroid.y,
                    "distance": dist,
                    "polygon": polygon,
                })
            return results
        except Exception as e:
            logger.debug(f"crosswalk query failed: {e}")
            return []

    def _point_in_crosswalk(self, map_api, x: float, y: float) -> Tuple[bool, float]:
        """Check if a point is inside any crosswalk, return (is_inside, min_distance)."""
        crosswalks = self._query_crosswalks_near_point(map_api, x, y, self.crosswalk_radius)
        if not crosswalks:
            return False, float("inf")

        from shapely.geometry import Point as ShapelyPoint
        point = ShapelyPoint(x, y)
        min_dist = float("inf")
        is_inside = False

        for cw in crosswalks:
            dist = cw["polygon"].distance(point)
            if dist < min_dist:
                min_dist = dist
            if cw["polygon"].contains(point):
                is_inside = True

        return is_inside, min_dist

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_pedestrian_features(
            self,
            annotations: Annotations,
            ego_status: EgoStatus,
            map_api=None,
    ) -> List[PedestrianInfo]:
        """Extract features for all pedestrians in a single frame."""
        pedestrians = []

        ego_x, ego_y, ego_heading = ego_status.ego_pose
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)

        for idx, name in enumerate(annotations.names):
            if name != "pedestrian":
                continue

            # bounding box [x, y, z, length, width, height, heading] in ego frame
            box = annotations.boxes[idx]
            ped_x_ego, ped_y_ego = box[0], box[1]

            distance = np.sqrt(ped_x_ego ** 2 + ped_y_ego ** 2)

            vx, vy = annotations.velocity_3d[idx][0], annotations.velocity_3d[idx][1]
            speed = np.sqrt(vx ** 2 + vy ** 2)

            bearing = np.arctan2(ped_x_ego, ped_y_ego)

            # in ego frame: x is forward, y is left
            in_corridor = (
                0 < ped_x_ego < self.corridor_length
                and abs(ped_y_ego) < self.corridor_width
            )

            lateral_offset = abs(ped_y_ego)

            # heading alignment: does ped move toward ego path
            to_ego_x = -ped_x_ego
            to_ego_y = -ped_y_ego
            norm = max(np.sqrt(to_ego_x ** 2 + to_ego_y ** 2), 1e-6)
            heading_alignment = (vx * to_ego_x + vy * to_ego_y) / (norm * max(speed, 1e-6))

            # crosswalk context (needs global coordinates)
            on_crosswalk = False
            near_crosswalk = False
            crosswalk_dist = float("inf")

            if map_api is not None:
                ped_x_global = ego_x + cos_h * ped_x_ego - sin_h * ped_y_ego
                ped_y_global = ego_y + sin_h * ped_x_ego + cos_h * ped_y_ego
                on_crosswalk, crosswalk_dist = self._point_in_crosswalk(
                    map_api, ped_x_global, ped_y_global
                )
                near_crosswalk = crosswalk_dist < self.crosswalk_radius

            pedestrians.append(PedestrianInfo(
                track_token=annotations.track_tokens[idx],
                distance_to_ego=float(distance),
                rel_x=float(ped_x_ego),
                rel_y=float(ped_y_ego),
                speed=float(speed),
                vx=float(vx),
                vy=float(vy),
                bearing_angle=float(bearing),
                is_in_forward_corridor=bool(in_corridor),
                lateral_offset=float(lateral_offset),
                heading_alignment=float(heading_alignment),
                on_crosswalk=bool(on_crosswalk),
                near_crosswalk=bool(near_crosswalk),
                crosswalk_distance=float(crosswalk_dist),
            ))

        return pedestrians

    def _analyze_track_persistence(self, scene: Scene) -> Dict[str, int]:
        """Count how many history frames each pedestrian track_token appears in."""
        track_counts: Dict[str, int] = defaultdict(int)
        num_history = scene.scene_metadata.num_history_frames

        for frame_idx in range(num_history):
            frame = scene.frames[frame_idx]
            for idx, name in enumerate(frame.annotations.names):
                if name == "pedestrian":
                    token = frame.annotations.track_tokens[idx]
                    track_counts[token] += 1

        return dict(track_counts)

    def _classify_interaction_strength(self, info: SceneInteractionInfo) -> str:
        """Classify interaction strength based on features."""
        if info.total_pedestrians == 0:
            return "none"

        if info.pedestrians_in_corridor > 0 and (
            info.best_ped_on_crosswalk or info.best_ped_distance < 10.0
        ):
            return "strong"

        if info.pedestrians_in_corridor > 0 or (
            info.pedestrians_near_crosswalk > 0
            and info.best_ped_distance < self.min_interaction_distance
        ):
            return "moderate"

        if info.total_pedestrians > 0 and info.best_ped_distance < self.min_interaction_distance:
            return "weak"

        return "none"

    # ------------------------------------------------------------------
    # Scene & dataset analysis
    # ------------------------------------------------------------------

    def analyze_scene(self, scene: Scene, use_map: bool = True) -> SceneInteractionInfo:
        """Analyze a single scene for pedestrian interaction potential."""
        metadata = scene.scene_metadata
        current_frame_idx = metadata.num_history_frames - 1
        current_frame = scene.frames[current_frame_idx]
        ego_status = current_frame.ego_status

        map_api = self._get_map_api(metadata.map_name) if use_map else None

        pedestrians = self._extract_pedestrian_features(
            current_frame.annotations, ego_status, map_api
        )

        track_counts = self._analyze_track_persistence(scene)
        for ped in pedestrians:
            if ped.track_token in track_counts:
                ped.frames_visible = track_counts[ped.track_token]
                ped.track_persistent = track_counts[ped.track_token] >= 2

        ego_vx, ego_vy = ego_status.ego_velocity
        ego_ax, ego_ay = ego_status.ego_acceleration

        info = SceneInteractionInfo(
            token=metadata.initial_token,
            log_name=metadata.log_name,
            map_name=metadata.map_name,
            total_pedestrians=len(pedestrians),
            pedestrians_in_corridor=sum(1 for p in pedestrians if p.is_in_forward_corridor),
            pedestrians_near_crosswalk=sum(1 for p in pedestrians if p.near_crosswalk),
            ego_speed=float(np.sqrt(ego_vx ** 2 + ego_vy ** 2)),
            ego_acceleration_magnitude=float(np.sqrt(ego_ax ** 2 + ego_ay ** 2)),
        )

        implausible_count = sum(
            1 for p in pedestrians if p.speed > self.max_plausible_ped_speed
        )
        info.num_implausible_velocities = implausible_count
        info.ped_velocity_plausible = implausible_count == 0

        if pedestrians:
            info.avg_ped_velocity_magnitude = float(
                np.mean([p.speed for p in pedestrians])
            )

        # best candidate pedestrian
        candidates = sorted(
            pedestrians,
            key=lambda p: (
                -int(p.is_in_forward_corridor),
                -int(p.on_crosswalk),
                -int(p.near_crosswalk),
                p.distance_to_ego,
            ),
        )
        if candidates:
            best = candidates[0]
            info.best_ped_distance = best.distance_to_ego
            info.best_ped_lateral_offset = best.lateral_offset
            info.best_ped_speed = best.speed
            info.best_ped_track_token = best.track_token
            info.best_ped_on_crosswalk = best.on_crosswalk

        # track persistence rate
        if pedestrians:
            info.ped_track_persistence_rate = float(
                sum(1 for p in pedestrians if p.track_persistent) / len(pedestrians)
            )

        # crosswalk ahead check
        if map_api is not None:
            ego_x, ego_y, ego_heading = ego_status.ego_pose
            cos_h = np.cos(ego_heading)
            sin_h = np.sin(ego_heading)

            for lookahead in [10.0, 20.0, 30.0]:
                check_x = ego_x + cos_h * lookahead
                check_y = ego_y + sin_h * lookahead
                crosswalks = self._query_crosswalks_near_point(map_api, check_x, check_y, 10.0)
                if crosswalks:
                    info.has_crosswalk_ahead = True
                    info.crosswalk_distance_ahead = min(
                        info.crosswalk_distance_ahead,
                        min(cw["distance"] for cw in crosswalks) + lookahead - 10.0,
                    )
                    break

        info.interaction_strength = self._classify_interaction_strength(info)
        info.has_pedestrian_interaction = info.interaction_strength in ("moderate", "strong")

        info.pedestrian_details = [
            {
                "track_token": p.track_token,
                "distance": round(p.distance_to_ego, 2),
                "rel_x": round(p.rel_x, 2),
                "rel_y": round(p.rel_y, 2),
                "speed": round(p.speed, 2),
                "in_corridor": p.is_in_forward_corridor,
                "on_crosswalk": p.on_crosswalk,
                "near_crosswalk": p.near_crosswalk,
                "frames_visible": p.frames_visible,
                "heading_alignment": round(p.heading_alignment, 3),
            }
            for p in pedestrians
        ]

        return info

    def mine_dataset(
        self,
        scene_filter: Optional[SceneFilter] = None,
        max_scenes: Optional[int] = None,
        use_map: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[SceneInteractionInfo], DatasetSummary]:
        """Mine the entire dataset for pedestrian interactions."""

        if scene_filter is None:
            scene_filter = SceneFilter()

        scene_loader = SceneLoader(
            original_sensor_path=None,
            data_path=self.data_path,
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

        tokens = scene_loader.tokens
        if max_scenes is not None:
            tokens = tokens[:max_scenes]

        logger.info(f"Mining {len(tokens)} scenes for pedestrian interactions...")

        all_results: List[SceneInteractionInfo] = []
        summary = DatasetSummary()

        start_time = time.time()
        for idx, token in enumerate(tokens):
            try:
                scene = scene_loader.get_scene_from_token(token)
                info = self.analyze_scene(scene, use_map=use_map)
                all_results.append(info)

                summary.total_scenes += 1
                summary.per_map_counts[info.map_name] += 1

                if info.total_pedestrians > 0:
                    summary.scenes_with_pedestrians += 1
                if info.pedestrians_in_corridor > 0:
                    summary.scenes_with_corridor_peds += 1
                if info.pedestrians_near_crosswalk > 0:
                    summary.scenes_with_crosswalk_peds += 1
                if info.has_pedestrian_interaction:
                    summary.scenes_with_interactions += 1
                    summary.per_map_interactions[info.map_name] += 1

                summary.interaction_strength_counts[info.interaction_strength] += 1
                summary.max_peds_in_scene = max(
                    summary.max_peds_in_scene, info.total_pedestrians
                )

                if verbose and (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (len(tokens) - idx - 1) / rate
                    logger.info(
                        f"  [{idx + 1}/{len(tokens)}] "
                        f"interactions: {summary.scenes_with_interactions} "
                        f"({100 * summary.scenes_with_interactions / (idx + 1):.1f}%) "
                        f"| rate: {rate:.1f} scenes/s | ETA: {eta:.0f}s"
                    )

            except Exception as e:
                logger.warning(f"Failed to process token {token}: {e}")
                continue

        if summary.total_scenes > 0:
            summary.avg_peds_per_scene = (
                sum(r.total_pedestrians for r in all_results) / summary.total_scenes
            )
            scenes_with_peds = [r for r in all_results if r.total_pedestrians > 0]
            if scenes_with_peds:
                summary.track_persistence_rate = float(
                    np.mean([r.ped_track_persistence_rate for r in scenes_with_peds])
                )
                summary.velocity_plausibility_rate = float(
                    np.mean([r.ped_velocity_plausible for r in scenes_with_peds])
                )

        return all_results, summary


# ---------------------------------------------------------------------------
# Output & reporting
# ---------------------------------------------------------------------------

def save_results(
    results: List[SceneInteractionInfo],
    summary: DatasetSummary,
    output_path: str,
    include_details: bool = False,
):
    """Save mining results to JSON."""
    output = {
        "summary": {
            "total_scenes": summary.total_scenes,
            "scenes_with_pedestrians": summary.scenes_with_pedestrians,
            "scenes_with_corridor_peds": summary.scenes_with_corridor_peds,
            "scenes_with_crosswalk_peds": summary.scenes_with_crosswalk_peds,
            "scenes_with_interactions": summary.scenes_with_interactions,
            "interaction_rate": summary.scenes_with_interactions / max(summary.total_scenes, 1),
            "interaction_strength_counts": dict(summary.interaction_strength_counts),
            "avg_peds_per_scene": round(summary.avg_peds_per_scene, 2),
            "max_peds_in_scene": summary.max_peds_in_scene,
            "track_persistence_rate": round(summary.track_persistence_rate, 3),
            "velocity_plausibility_rate": round(summary.velocity_plausibility_rate, 3),
            "per_map_counts": dict(summary.per_map_counts),
            "per_map_interactions": dict(summary.per_map_interactions),
        },
        "scenes": [],
    }

    for r in results:
        scene_entry = {
            "token": r.token,
            "log_name": r.log_name,
            "map_name": r.map_name,
            "has_pedestrian_interaction": r.has_pedestrian_interaction,
            "interaction_strength": r.interaction_strength,
            "total_pedestrians": r.total_pedestrians,
            "pedestrians_in_corridor": r.pedestrians_in_corridor,
            "pedestrians_near_crosswalk": r.pedestrians_near_crosswalk,
            "best_ped_distance": round(r.best_ped_distance, 2)
                if r.best_ped_distance != float("inf") else None,
            "best_ped_lateral_offset": round(r.best_ped_lateral_offset, 2)
                if r.best_ped_lateral_offset != float("inf") else None,
            "best_ped_speed": round(r.best_ped_speed, 3),
            "best_ped_on_crosswalk": r.best_ped_on_crosswalk,
            "ego_speed": round(r.ego_speed, 2),
            "has_crosswalk_ahead": r.has_crosswalk_ahead,
            "ped_velocity_plausible": r.ped_velocity_plausible,
            "ped_track_persistence_rate": round(r.ped_track_persistence_rate, 3),
        }
        if include_details:
            scene_entry["pedestrian_details"] = r.pedestrian_details
        output["scenes"].append(scene_entry)

    output["interactive_tokens"] = [r.token for r in results if r.has_pedestrian_interaction]
    output["strong_interaction_tokens"] = [
        r.token for r in results if r.interaction_strength == "strong"
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def print_report(summary: DatasetSummary):
    """Print a human-readable report."""
    print("\n" + "=" * 70)
    print("PEDESTRIAN INTERACTION MINING REPORT")
    print("=" * 70)

    print(f"\nTotal scenes analyzed:        {summary.total_scenes}")
    print(f"Scenes with pedestrians:      {summary.scenes_with_pedestrians} "
          f"({100 * summary.scenes_with_pedestrians / max(summary.total_scenes, 1):.1f}%)")
    print(f"Scenes with corridor peds:    {summary.scenes_with_corridor_peds} "
          f"({100 * summary.scenes_with_corridor_peds / max(summary.total_scenes, 1):.1f}%)")
    print(f"Scenes with crosswalk peds:   {summary.scenes_with_crosswalk_peds} "
          f"({100 * summary.scenes_with_crosswalk_peds / max(summary.total_scenes, 1):.1f}%)")
    print(f"Scenes with interactions:     {summary.scenes_with_interactions} "
          f"({100 * summary.scenes_with_interactions / max(summary.total_scenes, 1):.1f}%)")

    print(f"\nInteraction strength distribution:")
    for strength in ["none", "weak", "moderate", "strong"]:
        count = summary.interaction_strength_counts.get(strength, 0)
        print(f"  {strength:>10}: {count:>6} ({100 * count / max(summary.total_scenes, 1):.1f}%)")

    print(f"\nData quality:")
    print(f"  Avg pedestrians per scene:  {summary.avg_peds_per_scene:.2f}")
    print(f"  Max pedestrians in scene:   {summary.max_peds_in_scene}")
    print(f"  Track persistence rate:     {summary.track_persistence_rate:.1%}")
    print(f"  Velocity plausibility rate: {summary.velocity_plausibility_rate:.1%}")

    print(f"\nPer-map breakdown:")
    for map_name in sorted(summary.per_map_counts.keys()):
        total = summary.per_map_counts[map_name]
        interactions = summary.per_map_interactions.get(map_name, 0)
        print(f"  {map_name:>30}: {total:>5} scenes, {interactions:>4} interactions "
              f"({100 * interactions / max(total, 1):.1f}%)")

    print("\n" + "-" * 70)
    if summary.scenes_with_interactions >= 2000:
        print("  GO: Sufficient pedestrian interactions (2000+). Proceed with full architecture.")
    elif summary.scenes_with_interactions >= 500:
        print("  CAUTIOUS GO: Moderate pedestrian interactions (500-2000).")
        print("  Consider relaxing interaction criteria or using data augmentation.")
    else:
        print("❌ CONCERN: Low pedestrian interaction count (<500).")
        print("  Options: relax criteria, add weak interactions, use synthetic augmentation.")

    if summary.track_persistence_rate < 0.5:
        print("  Track persistence is low. Temporal features may be unreliable.")
    else:
        print(f"  Track persistence is good ({summary.track_persistence_rate:.1%}).")

    if summary.velocity_plausibility_rate < 0.8:
        print("  Many implausible pedestrian velocities. Consider position-based speed instead.")
    else:
        print(f"  Velocity data is mostly plausible ({summary.velocity_plausibility_rate:.1%}).")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mine NavSim for pedestrian interactions")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"NavSim download root (default: {DEFAULT_DATA_ROOT}). "
             f"Logs are expected at <data_root>/mini_navsim_logs/mini/",
    )
    parser.add_argument(
        "--map_root",
        type=str,
        default=str(DEFAULT_MAP_ROOT),
        help=f"nuPlan maps directory (default: {DEFAULT_MAP_ROOT})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path (default: ./ped_interaction_index.json)",
    )
    parser.add_argument("--max_scenes", type=int, default=None, help="Limit scenes to process")
    parser.add_argument("--no_map", action="store_true", help="Skip map queries")
    parser.add_argument("--include_details", action="store_true", help="Include per-ped details in output")
    parser.add_argument("--corridor_width", type=float, default=5.0)
    parser.add_argument("--corridor_length", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # data_path = <data_root>/mini_navsim_logs/mini  (mirrors NavsimDataset)
    data_path = Path(args.data_root) / "mini_navsim_logs" / "mini"

    if not data_path.exists():
        raise FileNotFoundError(
            f"NavSim log directory not found: {data_path}\n"
            f"Please check --data_root points to the NavSim download folder."
        )

    map_root = Path(args.map_root) if args.map_root else None
    if map_root and not map_root.exists():
        logger.warning(f"Map root not found: {map_root}. Map features will be disabled.")
        map_root = None

    miner = PedestrianSceneMiner(
        data_path=data_path,
        map_root=map_root,
        corridor_width=args.corridor_width,
        corridor_length=args.corridor_length,
    )

    results, summary = miner.mine_dataset(
        max_scenes=args.max_scenes,
        use_map=not args.no_map,
        verbose=args.verbose,
    )

    save_results(results, summary, args.output_path, include_details=args.include_details)
    print_report(summary)


if __name__ == "__main__":
    main()