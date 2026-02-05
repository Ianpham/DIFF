"""
Enhanced NAVSIM Dataset with Full Integration
=============================================

Complete version with:
1. NavSimScenario & TrajectorySampling
2. BEVLabelExtractor (12 semantic labels)
3. Route & mission goal extraction
4. Curriculum learning support
5. Full collate function

Author: Complete enhanced version
"""

import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset, Subset
from typing import Dict, Tuple, Optional, List
import warnings
from dataclasses import dataclass
from enum import Enum
from skimage.draw import polygon as draw_polygon

# NAVSIM imports
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig, AgentInput
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint

# map including in this phase, navsim is for very light experience
from navsim_utilize.vectormapfeature import VectorMapExtractor, extract__vector_map_for_batch

# ===========================================================================
# Trajectory Sampling Configuration
# ===========================================================================

class TrajectoryConfig:
    """Configuration for trajectory sampling aligned with NAVSIM."""
    
    # Standard NAVSIM trajectory sampling
    PLANNING_TRAJECTORY_SAMPLING = TrajectorySampling(
        time_horizon=8.0,      # 8 seconds into future
        interval_length=0.5,   # 0.5s intervals (16 steps)
        num_poses=16
    )
    
    # Short-term prediction (for reactive planning)
    SHORT_TERM_SAMPLING = TrajectorySampling(
        time_horizon=3.0,
        interval_length=0.2,
        num_poses=15
    )
    
    # Long-term prediction (for strategic planning)
    LONG_TERM_SAMPLING = TrajectorySampling(
        time_horizon=12.0,
        interval_length=1.0,
        num_poses=12
    )


# ===========================================================================
# Difficulty Levels for Curriculum Learning
# ===========================================================================

class DifficultyLevel(Enum):
    """Scenario difficulty levels for curriculum learning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


# ===========================================================================
# BEV Label Extractor (12 Semantic Channels)
# ===========================================================================

class BEVLabelExtractor:
    """Extract all 12 BEV labels from NAVSIM HD maps + annotations."""
    
    def __init__(self, bev_size=(200, 200), bev_range=50.0, map_root=None, map_version="nuplan-maps-v1.0"):
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.resolution = (2 * bev_range) / bev_size[0]
        
        self.map_root = map_root or os.environ.get('NUPLAN_MAPS_ROOT')
        self.map_version = map_version
        
        if not self.map_root:
            raise ValueError(
                "Map root not provided! Set NUPLAN_MAPS_ROOT environment variable or pass map_root parameter."
            )
        
        # Cache map APIs to avoid reloading
        self._map_cache = {}
    
    def _get_map_api(self, scene):
        """Initialize map API for a scene with caching."""
        map_name = scene.scene_metadata.map_name
        
        if map_name == "las_vegas":
            map_name = "us-nv-las-vegas-strip"
        
        # Use cache
        if map_name in self._map_cache:
            return self._map_cache[map_name]
        
        if map_name not in MAP_LOCATIONS:
            available_maps = list(MAP_LOCATIONS.keys())
            raise ValueError(
                f"Map '{map_name}' not found! Available: {available_maps}"
            )
        
        try:
            map_api = get_maps_api(self.map_root, self.map_version, map_name)
            self._map_cache[map_name] = map_api
            return map_api
        except Exception as e:
            raise RuntimeError(f"Failed to load map '{map_name}': {e}")
    def _world_to_bev(self, world_coords, ego_pose):
        """Convert world coordinates to BEV pixel coordinates."""
        x, y, heading = ego_pose
        
        # Transform to ego frame
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx = world_coords[:, 0] - x
        dy = world_coords[:, 1] - y
        local_x = cos_h * dx + sin_h * dy
        local_y = -sin_h * dx + cos_h * dy
        
        # Convert to pixel coordinates (flip y-axis for image coordinates)
        H, W = self.bev_size
        pixel_x = ((local_x + self.bev_range) / (2 * self.bev_range) * W).astype(int)
        pixel_y = ((self.bev_range - local_y) / (2 * self.bev_range) * H).astype(int)
        
        # Filter valid pixels
        valid = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
        
        return pixel_y[valid], pixel_x[valid]
    # def extract_all_labels(self, scene, frame_idx: int) -> Dict[str, np.ndarray]:
    #     """Extract all 12 BEV labels."""
    #     frame = scene.frames[frame_idx]
    #     ego_pose = frame.ego_status.ego_pose
        
    #     try:
    #         map_api = self._get_map_api(scene)
    #     except Exception as e:
    #         warnings.warn(f"Map API failed: {e}")
    #         return self._get_empty_labels()
        
    #     labels = {}
        
    #     # Extract all labels with error handling
    #     label_extractors = [
    #         ('drivable_area', lambda: self._extract_drivable_area(map_api, ego_pose)),
    #         ('lane_boundaries', lambda: self._extract_lane_boundaries(map_api, ego_pose)),
    #         ('lane_dividers', lambda: self._extract_lane_dividers(map_api, ego_pose)),
    #         ('vehicle_occupancy', lambda: self._extract_vehicle_occupancy(frame)),
    #         ('pedestrian_occupancy', lambda: self._extract_pedestrian_occupancy(frame)),
    #         ('crosswalks', lambda: self._extract_crosswalks(map_api, ego_pose)),
    #         ('stop_lines', lambda: self._extract_stop_lines(map_api, ego_pose)),
    #     ]
        
    #     for name, extractor in label_extractors:
    #         try:
    #             labels[name] = extractor()
    #         except Exception as e:
    #             warnings.warn(f"{name} extraction failed: {e}")
    #             labels[name] = np.zeros(self.bev_size, dtype=np.float32)
        
    #     # Velocity fields
    #     velocity_x, velocity_y = self._extract_velocity_fields(frame)
    #     labels['velocity_x'] = velocity_x
    #     labels['velocity_y'] = velocity_y
        
    #     # Ego mask
    #     labels['ego_mask'] = self._extract_ego_mask(frame.ego_status)
        
    #     # Traffic lights
    #     labels['traffic_lights'] = self._extract_traffic_lights(frame, map_api, ego_pose)
        
    #     # Vehicle classes
    #     labels['vehicle_classes'] = self._extract_vehicle_classes(frame)
        
    #     return labels
    
    # def _world_to_bev(self, world_coords, ego_pose):
    #     """Convert world coordinates to BEV pixel coordinates."""
    #     x, y, heading = ego_pose
        
    #     # Transform to ego frame
    #     cos_h, sin_h = np.cos(heading), np.sin(heading)
    #     dx = world_coords[:, 0] - x
    #     dy = world_coords[:, 1] - y
    #     local_x = cos_h * dx + sin_h * dy
    #     local_y = -sin_h * dx + cos_h * dy
        
    #     # Convert to pixel coordinates (flip y-axis for image coordinates)
    #     H, W = self.bev_size
    #     pixel_x = ((local_x + self.bev_range) / (2 * self.bev_range) * W).astype(int)
    #     pixel_y = ((self.bev_range - local_y) / (2 * self.bev_range) * H).astype(int)
        
    #     # Filter valid pixels
    #     valid = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
        
    #     return pixel_y[valid], pixel_x[valid]


    def extract_all_labels(self, scene, frame_idx: int) -> Dict[str, np.ndarray]:
        """
        Extract all 12 BEV labels.
        
        Wraps map queries with warning suppression to avoid nuplan's
        'invalid value encountered in cast' RuntimeWarnings.
        """
        frame = scene.frames[frame_idx]
        ego_pose = frame.ego_status.ego_pose
        
        try:
            map_api = self._get_map_api(scene)
        except Exception as e:
            warnings.warn(f"Map API failed: {e}")
            return self._get_empty_labels()
        
        labels = {}
        
        # Suppress nuplan map warnings during extraction
        # These come from nuplan/common/maps/nuplan_map/utils.py:413
        # when map data contains NaN values that can't be cast to int
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning
            )
            
            # Extract all labels with error handling
            label_extractors = [
                ('drivable_area', lambda: self._extract_drivable_area(map_api, ego_pose)),
                ('lane_boundaries', lambda: self._extract_lane_boundaries(map_api, ego_pose)),
                ('lane_dividers', lambda: self._extract_lane_dividers(map_api, ego_pose)),
                ('vehicle_occupancy', lambda: self._extract_vehicle_occupancy(frame)),
                ('pedestrian_occupancy', lambda: self._extract_pedestrian_occupancy(frame)),
                ('crosswalks', lambda: self._extract_crosswalks(map_api, ego_pose)),
                ('stop_lines', lambda: self._extract_stop_lines(map_api, ego_pose)),
            ]
            
            for name, extractor in label_extractors:
                try:
                    labels[name] = extractor()
                except Exception as e:
                    warnings.warn(f"{name} extraction failed: {e}")
                    labels[name] = np.zeros(self.bev_size, dtype=np.float32)
            
            # Velocity fields
            velocity_x, velocity_y = self._extract_velocity_fields(frame)
            labels['velocity_x'] = velocity_x
            labels['velocity_y'] = velocity_y
            
            # Ego mask
            labels['ego_mask'] = self._extract_ego_mask(frame.ego_status)
            
            # Traffic lights (also queries map API)
            labels['traffic_lights'] = self._extract_traffic_lights(frame, map_api, ego_pose)
            
            # Vehicle classes
            labels['vehicle_classes'] = self._extract_vehicle_classes(frame)
        
        return labels
    
    def _extract_drivable_area(self, map_api, ego_pose):
        """Extract drivable area from HD map."""
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            roadblocks = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.ROADBLOCK]
            )
            
            for roadblock in roadblocks[SemanticMapLayer.ROADBLOCK]:
                polygon = roadblock.polygon
                coords = np.array(polygon.exterior.coords)
                
                if len(coords) > 2:
                    pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                    
                    if len(pixel_y) > 2:
                        rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                        mask[rr, cc] = 1.0
        except Exception as e:
            warnings.warn(f"Drivable area extraction error: {e}")
        
        return mask
    
    def _extract_lane_boundaries(self, map_api, ego_pose):
        """Extract lane boundaries (edges of lanes)."""
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            
            lanes = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE]
            )
            
            lane_objects = lanes.get(SemanticMapLayer.LANE, [])
            
            for lane in lane_objects:
                # Extract left boundary
                try:
                    left_boundary = lane.left_boundary
                    if left_boundary is not None:
                        linestring = left_boundary.linestring
                        coords = np.array(linestring.coords)
                        
                        if len(coords) > 1:
                            pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                            if len(pixel_y) > 0:
                                valid_mask = (pixel_y >= 0) & (pixel_y < self.bev_size[0]) & \
                                        (pixel_x >= 0) & (pixel_x < self.bev_size[1])
                                pixel_y = pixel_y[valid_mask]
                                pixel_x = pixel_x[valid_mask]
                                if len(pixel_y) > 0:
                                    mask[pixel_y, pixel_x] = 1.0
                except:
                    pass
                
                # Extract right boundary
                try:
                    right_boundary = lane.right_boundary
                    if right_boundary is not None:
                        linestring = right_boundary.linestring
                        coords = np.array(linestring.coords)
                        
                        if len(coords) > 1:
                            pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                            if len(pixel_y) > 0:
                                valid_mask = (pixel_y >= 0) & (pixel_y < self.bev_size[0]) & \
                                        (pixel_x >= 0) & (pixel_x < self.bev_size[1])
                                pixel_y = pixel_y[valid_mask]
                                pixel_x = pixel_x[valid_mask]
                                if len(pixel_y) > 0:
                                    mask[pixel_y, pixel_x] = 1.0
                except:
                    pass
                    
        except Exception as e:
            warnings.warn(f"Lane boundaries extraction error: {e}")
        
        return mask
    
    def _extract_lane_dividers(self, map_api, ego_pose):
        """Extract lane dividers (centerlines between lanes)."""
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            
            lanes = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE]
            )
            
            for lane in lanes.get(SemanticMapLayer.LANE, []):
                if hasattr(lane, 'baseline_path'):
                    try:
                        path = lane.baseline_path
                        if hasattr(path, 'discrete_path'):
                            coords = np.array([[state.x, state.y] for state in path.discrete_path])
                        elif hasattr(path, 'linestring'):
                            coords = np.array(path.linestring.coords)
                        else:
                            continue
                            
                        if len(coords) > 1:
                            pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                            if len(pixel_y) > 0:
                                mask[pixel_y, pixel_x] = 1.0
                    except:
                        pass
                        
        except Exception as e:
            warnings.warn(f"Lane dividers extraction error: {e}")
        
        return mask
    
    def _extract_vehicle_occupancy(self, frame):
        """Extract vehicle occupancy from annotations."""
        mask = np.zeros(self.bev_size, dtype=np.float32)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return mask
        
        boxes = frame.annotations.boxes
        names = frame.annotations.names
        
        for i, name in enumerate(names):
            if 'vehicle' in name.lower() or 'car' in name.lower() or 'truck' in name.lower():
                box = boxes[i]
                
                # Boxes are in ego frame
                x, y = box[0], box[1]
                yaw = box[6]
                w, l = box[3], box[4]
                
                # Check distance
                dist = np.sqrt(x**2 + y**2)
                if dist > self.bev_range:
                    continue
                
                corners = self._get_box_corners(x, y, yaw, w, l)
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = 1.0
        
        return mask
    
    def _ego_to_bev_pixels(self, ego_coords):
        """Convert ego-frame coordinates to BEV pixels."""
        H, W = self.bev_size
        
        pixel_x = ((ego_coords[:, 1] + self.bev_range) / (2 * self.bev_range) * W).astype(int)
        pixel_y = ((self.bev_range - ego_coords[:, 0]) / (2 * self.bev_range) * H).astype(int)
        
        valid = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
        
        return pixel_y[valid], pixel_x[valid]
    
    def _extract_pedestrian_occupancy(self, frame):
        """Extract pedestrian occupancy."""
        mask = np.zeros(self.bev_size, dtype=np.float32)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return mask
        
        boxes = frame.annotations.boxes
        names = frame.annotations.names
        
        for i, name in enumerate(names):
            if 'pedestrian' in name.lower() or 'person' in name.lower():
                box = boxes[i]
                x, y = box[0], box[1]
                yaw = box[6]
                w, l = box[3], box[4]
                
                dist = np.sqrt(x**2 + y**2)
                if dist > self.bev_range:
                    continue
                
                corners = self._get_box_corners(x, y, yaw, w, l)
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = 1.0
        
        return mask
    
    def _extract_velocity_fields(self, frame):
        """Extract velocity fields from annotations."""
        velocity_x = np.zeros(self.bev_size, dtype=np.float32)
        velocity_y = np.zeros(self.bev_size, dtype=np.float32)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return velocity_x, velocity_y
        
        boxes = frame.annotations.boxes
        
        if not hasattr(frame.annotations, 'velocity_3d') or frame.annotations.velocity_3d is None:
            return velocity_x, velocity_y
            
        velocities = frame.annotations.velocity_3d
        
        for i in range(len(boxes)):
            box = boxes[i]
            x, y = box[0], box[1]
            
            dist = np.sqrt(x**2 + y**2)
            if dist > self.bev_range:
                continue
            
            vx_ego, vy_ego = velocities[i, 0], velocities[i, 1]
            position = np.array([[x, y]])
            pixel_y, pixel_x = self._ego_to_bev_pixels(position)
            
            if len(pixel_y) > 0:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        py, px = pixel_y[0] + dy, pixel_x[0] + dx
                        if 0 <= py < self.bev_size[0] and 0 <= px < self.bev_size[1]:
                            velocity_x[py, px] = vx_ego
                            velocity_y[py, px] = vy_ego
        
        return velocity_x, velocity_y
    
    def _extract_ego_mask(self, ego_status):
        """Extract ego vehicle footprint."""
        mask = np.zeros(self.bev_size, dtype=np.float32)
        
        ego_width, ego_length = 2.0, 4.5
        
        x, y, heading = 0, 0, 0
        corners = self._get_box_corners(x, y, heading, ego_width, ego_length)
        
        H, W = self.bev_size
        center_x, center_y = W // 2, H // 2
        
        pixels_x = (corners[:, 0] / self.resolution + center_x).astype(int)
        pixels_y = (center_y - corners[:, 1] / self.resolution).astype(int)
        
        valid = (pixels_x >= 0) & (pixels_x < W) & (pixels_y >= 0) & (pixels_y < H)
        pixels_y, pixels_x = pixels_y[valid], pixels_x[valid]
        
        if len(pixels_y) > 2:
            rr, cc = draw_polygon(pixels_y, pixels_x, shape=self.bev_size)
            mask[rr, cc] = 1.0
        
        return mask
    
    def _extract_traffic_lights(self, frame, map_api, ego_pose):
        """Extract traffic light status."""
        mask = np.zeros(self.bev_size, dtype=np.uint8)
        
        if not hasattr(frame, 'traffic_lights') or frame.traffic_lights is None or len(frame.traffic_lights) == 0:
            return mask
            
        try:
            from shapely.geometry import Point as ShapelyPoint
            import re
            
            x, y, _ = ego_pose
            ego_point = ShapelyPoint(x, y)
            
            lane_connectors = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE_CONNECTOR]
            )
            
            traffic_light_dict = {}
            for light_id, is_green in frame.traffic_lights:
                traffic_light_dict[light_id] = is_green
            
            for connector in lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, []):
                try:
                    if hasattr(connector, 'id'):
                        connector_id_str = str(connector.id)
                        numbers = re.findall(r'\d+', connector_id_str)
                        if numbers:
                            connector_id = int(numbers[-1])
                        else:
                            continue
                    else:
                        continue
                    
                    if connector_id in traffic_light_dict:
                        is_green = traffic_light_dict[connector_id]
                        
                        if hasattr(connector, 'baseline_path'):
                            try:
                                path = connector.baseline_path
                                if hasattr(path, 'discrete_path'):
                                    coords = np.array([[state.x, state.y] for state in path.discrete_path])
                                elif hasattr(path, 'linestring'):
                                    coords = np.array(path.linestring.coords)
                                else:
                                    continue
                                    
                                if len(coords) > 0:
                                    pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                                    if len(pixel_y) > 0:
                                        value = 2 if is_green else 1
                                        mask[pixel_y, pixel_x] = value
                            except:
                                pass
                except:
                    continue
                
        except Exception as e:
            warnings.warn(f"Traffic lights extraction error: {e}")
        
        return mask
    
    def _extract_vehicle_classes(self, frame):
        """Extract vehicle classes."""
        mask = np.zeros(self.bev_size, dtype=np.uint8)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return mask
        
        boxes = frame.annotations.boxes
        names = frame.annotations.names
        
        class_map = {
            'car': 1, 'vehicle': 1,
            'truck': 2,
            'bus': 3,
            'bicycle': 4,
            'motorcycle': 5
        }
        
        for i, name in enumerate(names):
            name_lower = name.lower()
            class_id = 0
            for key, val in class_map.items():
                if key in name_lower:
                    class_id = val
                    break
            
            if class_id > 0:
                box = boxes[i]
                x, y = box[0], box[1]
                yaw = box[6]
                w, l = box[3], box[4]
                
                dist = np.sqrt(x**2 + y**2)
                if dist > self.bev_range:
                    continue
                
                corners = self._get_box_corners(x, y, yaw, w, l)
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = class_id
        
        return mask
    
    def _extract_crosswalks(self, map_api, ego_pose):
        """Extract crosswalks."""
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            walkways = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.WALKWAYS]
            )
            
            for walkway in walkways.get(SemanticMapLayer.WALKWAYS, []):
                polygon = walkway.polygon
                coords = np.array(polygon.exterior.coords)
                
                if len(coords) > 2:
                    pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                    
                    if len(pixel_y) > 2:
                        rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                        mask[rr, cc] = 1.0
        except Exception as e:
            warnings.warn(f"Crosswalks extraction error: {e}")
        
        return mask
    
    def _extract_stop_lines(self, map_api, ego_pose):
        """Extract stop lines."""
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            stops = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.STOP_LINE]
            )
            
            for stop in stops.get(SemanticMapLayer.STOP_LINE, []):
                coords = None
                
                if hasattr(stop, 'polygon') and stop.polygon is not None:
                    coords = np.array(stop.polygon.exterior.coords)
                elif hasattr(stop, 'linestring') and stop.linestring is not None:
                    coords = np.array(stop.linestring.coords)
                
                if coords is not None and len(coords) > 1:
                    pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                    if len(pixel_y) > 0:
                        mask[pixel_y, pixel_x] = 1.0
        except Exception as e:
            warnings.warn(f"Stop lines extraction error: {e}")
        
        return mask
    
    def _get_box_corners(self, x, y, yaw, width, length):
        """Get 4 corners of a bounding box."""
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        corners_local = np.array([
            [-length/2, -width/2],
            [-length/2, +width/2],
            [+length/2, +width/2],
            [+length/2, -width/2],
        ])
        
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners_world = corners_local @ rotation.T + np.array([x, y])
        
        return corners_world
    
    def _get_empty_labels(self) -> Dict[str, np.ndarray]:
        """Return empty label dict."""
        return {
            'drivable_area': np.zeros(self.bev_size, dtype=np.float32),
            'lane_boundaries': np.zeros(self.bev_size, dtype=np.float32),
            'lane_dividers': np.zeros(self.bev_size, dtype=np.float32),
            'vehicle_occupancy': np.zeros(self.bev_size, dtype=np.float32),
            'pedestrian_occupancy': np.zeros(self.bev_size, dtype=np.float32),
            'velocity_x': np.zeros(self.bev_size, dtype=np.float32),
            'velocity_y': np.zeros(self.bev_size, dtype=np.float32),
            'ego_mask': np.zeros(self.bev_size, dtype=np.float32),
            'traffic_lights': np.zeros(self.bev_size, dtype=np.uint8),
            'vehicle_classes': np.zeros(self.bev_size, dtype=np.uint8),
            'crosswalks': np.zeros(self.bev_size, dtype=np.float32),
            'stop_lines': np.zeros(self.bev_size, dtype=np.float32),
        }


# ===========================================================================
# Scenario Builder with NavSimScenario
# ===========================================================================

class NavsimScenarioBuilder:
    """Build NavSimScenario objects from NAVSIM scenes."""
    
    def __init__(self, map_root: str, map_version: str = "nuplan-maps-v1.0"):
        self.map_root = map_root
        self.map_version = map_version
        self._map_cache = {}
    
    def build_scenario(
        self, 
        scene: Scene, 
        trajectory_sampling: TrajectorySampling = None
    ) -> NavSimScenario:
        """Build NavSimScenario from a NAVSIM scene."""
        if trajectory_sampling is None:
            trajectory_sampling = TrajectoryConfig.PLANNING_TRAJECTORY_SAMPLING
        
        map_api = self._get_map_api(scene)
        
        scenario = NavSimScenario(
            scene=scene,
            map_root= self.map_root,
            map_version=self.map_version,
        )
        
        return scenario
    
    def _get_map_api(self, scene: Scene):
        """Get map API with caching."""
        map_name = scene.scene_metadata.map_name
        
        if map_name == "las_vegas":
            map_name = "us-nv-las-vegas-strip"
        
        if map_name in self._map_cache:
            return self._map_cache[map_name]
        
        if map_name not in MAP_LOCATIONS:
            available = list(MAP_LOCATIONS.keys())
            raise ValueError(f"Map '{map_name}' not in {available}")
        
        map_api = get_maps_api(self.map_root, self.map_version, map_name)
        self._map_cache[map_name] = map_api
        return map_api


# ===========================================================================
# Route and Mission Goal Extractor
# ===========================================================================

class RouteExtractor:
    """Extract route and mission goal information from scenarios."""
    
    @staticmethod
    def extract_route_info(scenario: NavSimScenario) -> Dict:
        """Extract route information from scenario."""
        route_info = {
            'mission_goal': None,
            'route_roadblocks': [],
            'route_length': 0.0,
            'distance_to_goal': 0.0,
            'has_valid_route': False
        }
        
        try:
            if hasattr(scenario, 'get_mission_goal'):
                mission_goal = scenario.get_mission_goal()
                if mission_goal is not None:
                    route_info['mission_goal'] = [
                        mission_goal.x,
                        mission_goal.y,
                        mission_goal.heading
                    ]
                    route_info['has_valid_route'] = True
            
            if hasattr(scenario, 'get_route_roadblock_ids'):
                roadblock_ids = scenario.get_route_roadblock_ids()
                route_info['route_roadblocks'] = roadblock_ids
            
            if route_info['mission_goal'] is not None:
                initial_ego = scenario.initial_ego_state
                goal_pos = np.array(route_info['mission_goal'][:2])
                ego_pos = np.array([initial_ego.rear_axle.x, initial_ego.rear_axle.y])
                route_info['distance_to_goal'] = np.linalg.norm(goal_pos - ego_pos)
            
            if len(route_info['route_roadblocks']) > 0:
                route_info['route_length'] = len(route_info['route_roadblocks']) * 50.0
                
        except Exception as e:
            warnings.warn(f"Route extraction failed: {e}")
        
        return route_info


# ===========================================================================
# Enhanced Difficulty Analyzer
# ===========================================================================

@dataclass
class ScenarioDifficulty:
    """Enhanced difficulty metrics."""
    num_agents: int
    ego_speed: float
    ego_acceleration: float
    min_distance_to_agents: float
    road_curvature: float
    traffic_density: float
    has_intersection: bool
    has_lane_change: bool
    distance_to_goal: float = 0.0
    num_roadblocks: int = 0
    
    @property
    def difficulty_score(self) -> float:
        """Compute overall difficulty score [0-1]."""
        score = 0.0
        score += min(self.num_agents / 20.0, 1.0) * 0.15
        score += min(self.ego_speed / 30.0, 1.0) * 0.12
        score += min(abs(self.ego_acceleration) / 5.0, 1.0) * 0.12
        score += (1.0 - min(self.min_distance_to_agents / 50.0, 1.0)) * 0.18
        score += min(self.road_curvature / 0.1, 1.0) * 0.08
        score += self.traffic_density * 0.08
        score += min(self.num_roadblocks / 20.0, 1.0) * 0.05
        
        if self.has_intersection:
            score += 0.08
        if self.has_lane_change:
            score += 0.09
        
        return min(score, 1.0)
    
    @property
    def difficulty_level(self) -> DifficultyLevel:
        """Categorize into difficulty levels."""
        score = self.difficulty_score
        if score < 0.3:
            return DifficultyLevel.EASY
        elif score < 0.5:
            return DifficultyLevel.MEDIUM
        elif score < 0.7:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT


class DifficultyAnalyzer:
    """Analyze scenario difficulty."""
    
    @staticmethod
    def compute_difficulty(scenario: NavSimScenario, route_info: Dict) -> ScenarioDifficulty:
        """Compute difficulty metrics from scenario."""
        ego_state = scenario.initial_ego_state
        ego_speed = ego_state.dynamic_car_state.speed
        ego_accel = ego_state.dynamic_car_state.acceleration
        
        observation = scenario.get_ego_state_at_iteration(0)
        detections = observation.observation.detections
        
        num_agents = len(detections) if detections is not None else 0
        min_distance = float('inf')
        
        if num_agents > 0:
            ego_pos = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
            for detection in detections:
                agent_pos = np.array([detection.center.x, detection.center.y])
                distance = np.linalg.norm(ego_pos - agent_pos)
                min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            min_distance = 50.0
        
        traffic_density = num_agents / (np.pi * 50**2) if num_agents > 0 else 0.0
        road_curvature = 0.01
        num_roadblocks = len(route_info.get('route_roadblocks', []))
        distance_to_goal = route_info.get('distance_to_goal', 0.0)
        has_intersection = num_roadblocks > 1
        has_lane_change = False
        
        return ScenarioDifficulty(
            num_agents=num_agents,
            ego_speed=ego_speed,
            ego_acceleration=ego_accel,
            min_distance_to_agents=min_distance,
            road_curvature=road_curvature,
            traffic_density=traffic_density,
            has_intersection=has_intersection,
            has_lane_change=has_lane_change,
            distance_to_goal=distance_to_goal,
            num_roadblocks=num_roadblocks,
        )


# ===========================================================================
# Complete Enhanced Dataset
# ===========================================================================

class EnhancedNavsimDataset(Dataset):
    """
    Complete dataset with:
    - NavSimScenario integration
    - TrajectorySampling
    - BEVLabelExtractor (12 labels)
    - Route extraction
    - Curriculum learning
    """
    
    def __init__(
        self,
        data_split: str = "mini",
        bev_size: Tuple[int, int] = (200, 200),
        bev_range: float = 50.0,
        trajectory_sampling: TrajectorySampling = None,
        difficulty_filter: Optional[DifficultyLevel] = None,
        extract_labels: bool = True,
        extract_route_info: bool = True,
        use_cache: bool = True,
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        use_uniad_bev: bool = True,  # Use UniAD BEV features
        uniad_cache_dir: str = None,  # Path to precomputed BEV cache
        interpolate_bev: bool = False,  # NEW: Whether to upsample UniAD features
        # new: vector map options\
        extract_vector_maps: bool = True, # toggle vector map extraction
        max_points_per_lane: int = 20,
        vector_map_feature_dim: int = 16,
        max_crosswalks: int = 10,
    ):
        """Initialize complete dataset."""
        data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        self.data_split = data_split
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.difficulty_filter = difficulty_filter
        self.extract_labels = extract_labels
        self.extract_route_info = extract_route_info
        self.use_cache = use_cache
        self.use_uniad_bev = use_uniad_bev
        self.interpolate_bev = interpolate_bev
        
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
        
        print("=" * 70)
        print(f"Complete Enhanced NAVSIM Dataset - {data_split.upper()}")
        print("=" * 70)
        print(f"BEV: {bev_size} @ {bev_range}m")
        print(f"Trajectory: {self.trajectory_sampling.num_poses} poses @ {self.trajectory_sampling.interval_length}s")
        print(f"Extract labels: {extract_labels}")
        print(f"Difficulty filter: {difficulty_filter}")
        print("=" * 70)
        
        # Verify paths
        if not Path(self.map_root).exists():
            raise FileNotFoundError(f"Map directory not found: {self.map_root}")
        
        # Sensor config
        sensor_config = SensorConfig(
            cam_f0=False, cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=True
        )
        
        # Scene filter
        num_history = int(self.trajectory_sampling.time_horizon / self.trajectory_sampling.interval_length)
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=max(4, num_history // 4),
            num_future_frames=self.trajectory_sampling.num_poses,
        )
        
        # Scene loader
        self.scene_loader = SceneLoader(
            data_path=data_root / f'mini_navsim_logs' / data_split,
            original_sensor_path=data_root / f'mini_sensor_blobs' / data_split,
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        # Builders
        self.scenario_builder = NavsimScenarioBuilder(
            map_root=self.map_root,
            map_version=self.map_version
        )
        self.route_extractor = RouteExtractor()
        
        # Label extractor
        if self.extract_labels:
            self.label_extractor = BEVLabelExtractor(
                bev_size, bev_range, map_root=self.map_root, map_version=self.map_version
            )
        else:
            self.label_extractor = None
        
        # Compute difficulties
        all_tokens = self.scene_loader.tokens
        print("\nComputing scenario difficulties...")
        self.difficulty_scores = self._compute_all_difficulties()
        
        # Filter
        if difficulty_filter is not None:
            filtered_tokens = [
                token for token in all_tokens
                if self.difficulty_scores[token].difficulty_level == difficulty_filter
            ]
            self.scene_tokens = filtered_tokens
            print(f"✓ Filtered to {len(filtered_tokens)} {difficulty_filter.value} scenes")
        else:
            self.scene_tokens = all_tokens
        
        # Cache
        cache_name = f'{data_split}_{difficulty_filter.value if difficulty_filter else "all"}'
        self.cache_dir = data_root / 'cache' / 'navsim_complete' / cache_name
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Loaded {len(self)} scenes")
        self._print_difficulty_stats()
        print("=" * 70)

        # Vector map extractor
        self.extract_vector_maps = extract_vector_maps
        
        if self.extract_vector_maps:
            from navsim_utilize.vectormapfeature import VectorMapExtractor

            self.vector_map_extractor = VectorMapExtractor(
                map_root= map_root or self.map_root,
                map_version= map_version,
                max_points_per_lane= max_points_per_lane,
                feature_dim=vector_map_feature_dim,
                max_crosswalks=max_crosswalks
            )

            #chace for map APIs
            self._map_api_cache = {}

            print(f"✓ Vector map extractor initialized")
            print(f"  - Max points per lane: {max_points_per_lane}")
            print(f"  - Feature dimension: {vector_map_feature_dim}")
            print(f"  - Max crosswalks: {max_crosswalks}")
        else:
            self.vector_map_extractor = None
            print("⚠️  Vector map extraction DISABLED")

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

    def _compute_all_difficulties(self) -> Dict[str, ScenarioDifficulty]:
        """Compute difficulty for all scenarios."""
        difficulties = {}
        
        for token in self.scene_loader.tokens:
            scene = self.scene_loader.get_scene_from_token(token)
            
            try:
                scenario = self.scenario_builder.build_scenario(scene, self.trajectory_sampling)
                
                if self.extract_route_info:
                    route_info = self.route_extractor.extract_route_info(scenario)
                else:
                    route_info = {}
                
                difficulty = DifficultyAnalyzer.compute_difficulty(scenario, route_info)
                difficulties[token] = difficulty
                
            except Exception as e:
                warnings.warn(f"Failed to compute difficulty for {token}: {e}")
                difficulties[token] = ScenarioDifficulty(
                    num_agents=0, ego_speed=0.0, ego_acceleration=0.0,
                    min_distance_to_agents=50.0, road_curvature=0.0,
                    traffic_density=0.0, has_intersection=False,
                    has_lane_change=False
                )
        
        return difficulties
    
    def _print_difficulty_stats(self):
        """Print difficulty distribution."""
        if not self.difficulty_scores:
            return
        
        levels = {level: 0 for level in DifficultyLevel}
        scores = []
        
        for diff in self.difficulty_scores.values():
            levels[diff.difficulty_level] += 1
            scores.append(diff.difficulty_score)
        
        print("\n📊 Difficulty Distribution:")
        for level, count in levels.items():
            pct = count / len(self.difficulty_scores) * 100
            print(f"  {level.value:8s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n  Avg score: {np.mean(scores):.3f}")
        print(f"  Std score: {np.std(scores):.3f}")
    
    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get complete sample."""
        token = self.scene_tokens[idx]
        
        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load and process
        scene = self.scene_loader.get_scene_from_token(token)
        sample = self._process_scene(scene, token)
        
        # Cache
        if self.use_cache:
            torch.save(sample, cache_file)
        
        return sample
    
    def _process_scene(self, scene: Scene, token: str) -> Dict:
        """Process scene completely."""
        
        # Build scenario
        scenario = self.scenario_builder.build_scenario(scene, self.trajectory_sampling)
        
        # Route info
        if self.extract_route_info:
            route_info = self.route_extractor.extract_route_info(scenario)
        else:
            route_info = {}
        
        # Current frame
        current_frame_idx = scenario.database_interval - 1
        current_frame = scene.frames[current_frame_idx]
        
        # Original LiDAR point cloud (keep full point cloud)
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T  # (N, 3) - x, y, z
            lidar_original = torch.from_numpy(lidar_pc).float()
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_original = torch.zeros(0, 3)  # Empty point cloud
            lidar_bev = torch.zeros(2, *self.bev_size)

        # Original camera images - extract all available cameras
        camera_images = self._extract_camera_images(current_frame)
        
        #image bev_ but we must acknowledge that this cam_bev is in collabroration between lidar and camera, in which this set as in feature extraction.
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

        # BEV Labels (12 channels)
        if self.extract_labels and self.label_extractor is not None:
            try:
                labels = self.label_extractor.extract_all_labels(scene, current_frame_idx)
                labels_tensor = {k: torch.from_numpy(v).float() for k, v in labels.items()}
            except Exception as e:
                warnings.warn(f"Label extraction failed for {token}: {e}")
                labels_tensor = self._get_empty_labels_tensor()
        else:
            labels_tensor = {}
        
        # GT Trajectory
        gt_trajectory = self._extract_gt_trajectory(scenario)
        
        # Agent states
        ego_state = scenario.initial_ego_state
        agent_states = torch.tensor([[
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            ego_state.rear_axle.heading,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ]], dtype=torch.float32)
        
        # Agent history
        agent_history = self._extract_agent_history(scene, scenario)
        
        # Nearby agents
        nearby_agents = self._extract_nearby_agents(scenario)
        
        # Metadata
        difficulty = self.difficulty_scores.get(token)
        
        # FIX: Vector feature map data (move extraction inside if block)
        vector_map_features = None  # Default: None
        
        if self.extract_vector_maps:
            ego_pose = current_frame.ego_status.ego_pose
            map_name = scene.scene_metadata.map_name

            # Get map API (with caching)
            if map_name not in self._map_api_cache:
                self._map_api_cache[map_name] = self.vector_map_extractor.get_map_api(map_name)
            map_api = self._map_api_cache[map_name]

            # Extract vector map features
            try:
                vector_map_features = self.vector_map_extractor.extract(
                    map_api=map_api,
                    ego_pose=ego_pose, 
                    radius=50.0
                )
            except Exception as e:
                print(f"Warning: Vector map extraction failed for {token}: {e}")
                vector_map_features = None  # Fallback to None
                
        metadata = {
            'token': token,
            'log_name': scene.scene_metadata.log_name,
            'map_name': scene.scene_metadata.map_name,
            'timestamp': current_frame.timestamp,
            'difficulty': {
                'score': difficulty.difficulty_score if difficulty else 0.0,
                'level': difficulty.difficulty_level.value if difficulty else 'unknown',
                'num_agents': difficulty.num_agents if difficulty else 0,
                'ego_speed': difficulty.ego_speed if difficulty else 0.0,
            },
            'route': route_info,
        }
        
        return {
            'camera_images': camera_images,
            'lidar_original': lidar_original,
            'camera_bev': camera_bev,
            'lidar_bev': lidar_bev,
            'labels': labels_tensor,
            'agent_states': agent_states,
            'agent_history': agent_history,
            'gt_trajectory': gt_trajectory,
            'nearby_agents': nearby_agents,
            'metadata': metadata,
            'vector_map_raw': vector_map_features,  # None if extract_vector_maps=False
        }
            
    def _extract_gt_trajectory(self, scenario: NavSimScenario) -> torch.Tensor:
        """Extract GT trajectory using scenario."""
        trajectory = scenario.get_expert_trajectory()
        
        waypoints = []
        for state in trajectory.trajectory:
            waypoints.append([
                state.rear_axle.x,
                state.rear_axle.y,
                state.dynamic_car_state.rear_axle_velocity_2d.x,
                state.dynamic_car_state.rear_axle_velocity_2d.y,
                state.rear_axle.heading,
            ])
        
        return torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
    def _extract_camera_images(self, frame: Frame) -> Dict[str, torch.Tensor]:
        """Extract original camera images from frame."""
        camera_images = {}
        
        # Camera mapping - adjust based on your sensor config
        camera_keys = [
            ('cam_f0', 'front'),
            ('cam_l0', 'front_left'),
            ('cam_l1', 'side_left'),
            ('cam_l2', 'back_left'),
            ('cam_r0', 'front_right'),
            ('cam_r1', 'side_right'),
            ('cam_r2', 'back_right'),
            ('cam_b0', 'back'),
        ]       

        for cam_attr, cam_name in camera_keys:
            if hasattr(frame, cam_attr):
                cam_data = getattr(frame, cam_attr)
                if cam_data is not None and hasattr(cam_data, 'image'):
                    img = cam_data.image
                    if img is not None:
                        # Convert to torch tensor (C, H, W) format
                        if isinstance(img, np.ndarray):
                            if len(img.shape) == 3:
                                # If (H, W, C), transpose to (C, H, W)
                                if img.shape[2] == 3:
                                    img = np.transpose(img, (2, 0, 1))
                                camera_images[cam_name] = torch.from_numpy(img).float() / 255.0
                            elif len(img.shape) == 2:
                                # Grayscale image
                                camera_images[cam_name] = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        
        return camera_images
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
                detection.center.x,
                detection.center.y,
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
            height[y_indices[i], x_indices[i]] = max(height[y_indices[i], x_indices[i]], z[i])
        
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()
    
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


# ===========================================================================
# Collate Function
# ===========================================================================

def enhanced_collate_fn(batch):
    """Collate function for DataLoader."""
    
    camera_bev = torch.stack([item['camera_bev'] for item in batch])
    lidar_bev = torch.stack([item['lidar_bev'] for item in batch])
    
    # Stack labels if present
    if batch[0]['labels']:
        labels = {}
        for key in batch[0]['labels'].keys():
            labels[key] = torch.stack([item['labels'][key] for item in batch])
    else:
        labels = {}
    
    # Collate camera images (dict of lists)
    camera_images = {}
    if batch[0]['camera_images']:
        for cam_name in batch[0]['camera_images'].keys():
            camera_images[cam_name] = torch.stack([item['camera_images'][cam_name] for item in batch])
    
    # Collate LiDAR point clouds (list, as they have different sizes)
    lidar_original = [item['lidar_original'] for item in batch]

    # Collate vector map features
    vector_map = None
    if batch[0].get('vector_map_raw') is not None:
        from navsim_utilize.vectormapfeature import (
            pad_and_stack,
            pad_and_stack_2d,
            pad_and_stack_2d_mixed
        )

        # Filter out None values
        valid_features = [item['vector_map_raw'] for item in batch if item['vector_map_raw'] is not None]
        
        if len(valid_features) > 0:
            # Find max dimensions
            max_lanes = max(f.num_lanes for f in valid_features)
            max_crosswalks = max(f.num_crosswalks for f in valid_features)

            # FIX: Pad samples with None to match batch size
            while len(valid_features) < len(batch):  # ✅ FIXED: < instead of >
                # Create empty feature placeholder (duplicate first, will be masked)
                valid_features.append(valid_features[0])

            # Pad and stack all features
            vector_map = {
                # Core features
                'lane_polylines': pad_and_stack([f.lane_polylines for f in valid_features], max_lanes),
                'lane_features': pad_and_stack([f.lane_features for f in valid_features], max_lanes),
                'lane_masks': pad_and_stack([f.lane_masks for f in valid_features], max_lanes),
                'connectivity': pad_and_stack_2d([f.connectivity_matrix for f in valid_features], max_lanes),
                
                # Intersection features
                'lane_in_intersection': pad_and_stack([f.lane_in_intersection for f in valid_features], max_lanes),
                'intersection_ids': pad_and_stack([f.intersection_ids for f in valid_features], max_lanes),
                'approach_vectors': pad_and_stack([f.approach_vectors for f in valid_features], max_lanes),
                'turn_intentions': pad_and_stack([f.turn_intentions for f in valid_features], max_lanes),
                
                # Traffic control
                'stop_line_positions': pad_and_stack([f.stop_line_positions for f in valid_features], max_lanes),
                'stop_line_distances': pad_and_stack([f.stop_line_distances for f in valid_features], max_lanes),
                'has_stop_line': pad_and_stack([f.has_stop_line for f in valid_features], max_lanes),
                
                # Crosswalks
                'crosswalk_positions': pad_and_stack([f.crosswalk_positions for f in valid_features], max_crosswalks),
                'crosswalk_mask': pad_and_stack([f.crosswalk_mask for f in valid_features], max_crosswalks),
                'lane_to_crosswalk': pad_and_stack_2d_mixed(
                    [f.lane_to_crosswalk for f in valid_features], max_lanes, max_crosswalks
                ),
                
                # Visibility
                'lane_visibility_scores': pad_and_stack([f.lane_visibility_scores for f in valid_features], max_lanes),
                'occluded_regions': pad_and_stack([f.occluded_regions for f in valid_features], max_lanes),
                
                # Geometric
                'lane_curvatures': pad_and_stack([f.lane_curvatures for f in valid_features], max_lanes),
                'lane_lengths': pad_and_stack([f.lane_lengths for f in valid_features], max_lanes),
                'lane_headings': pad_and_stack([f.lane_headings for f in valid_features], max_lanes),
            }    

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
        'metadata': [item['metadata'] for item in batch],
        'vector_maps': vector_map,
    }
# ===========================================================================
# Curriculum Learning Manager
# ===========================================================================

class CurriculumLearningManager:
    """Manage curriculum learning stages."""
    
    def __init__(self, data_root: str, map_root: str):
        self.data_root = data_root
        self.map_root = map_root
    
    def get_curriculum_datasets(
        self,
        stages: List[DifficultyLevel] = None,
        **dataset_kwargs
    ) -> List[EnhancedNavsimDataset]:
        """Create datasets for each curriculum stage."""
        if stages is None:
            stages = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        
        datasets = []
        for stage in stages:
            print(f"\n{'='*70}")
            print(f"Creating Stage: {stage.value.upper()}")
            print(f"{'='*70}")
            
            dataset = EnhancedNavsimDataset(
                difficulty_filter=stage,
                map_root=self.map_root,
                **dataset_kwargs
            )
            datasets.append(dataset)
        
        return datasets


# ===========================================================================
# Test Script
# ===========================================================================

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("Complete NAVSIM Dataset Test")
    print("=" * 70)
    
    map_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
    os.environ['NUPLAN_MAPS_ROOT'] = map_root
    
    # Test complete dataset
    dataset = EnhancedNavsimDataset(
        data_split="mini",
        difficulty_filter=DifficultyLevel.EASY,
        extract_labels=True,
        extract_route_info=True,
        use_cache=False,
        map_root=map_root
    )
    
    print(f"\n✓ Created dataset: {len(dataset)} scenes")
    
    # Get sample
    sample = dataset[0]
    
    print("\n📊 Sample Contents:")
    print(f"  Camera BEV:     {sample['camera_images'].shape}")
    print(f"  LiDAR BEV:      {sample['lidar_original'].shape}")
    print(f"  Camera BEV:     {sample['camera_bev'].shape}")
    print(f"  LiDAR BEV:      {sample['lidar_bev'].shape}")
    print(f"  GT trajectory:  {sample['gt_trajectory'].shape}")
    print(f"  Labels:         {len(sample['labels'])} channels")
    if sample['labels']:
        for k, v in sample['labels'].items():
            print(f"    - {k:20s}: {v.shape}")
    
    print("\n📍 Metadata:")
    print(f"  Map:       {sample['metadata']['map_name']}")
    print(f"  Difficulty: {sample['metadata']['difficulty']}")
    print(f"  Route:     {sample['metadata']['route']}")
    
    # Test DataLoader
    print("\n[Test DataLoader]")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=enhanced_collate_fn,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"  Batch camera BEV: {batch['camera_bev'].shape}")
    print(f"  Batch labels:     {len(batch['labels'])} channels")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("=" * 70)