"""
Complete NAVSIM Dataset for Trajectory Prediction - FIXED MAP LOADING
======================================================================

Key fixes:
1. Properly initialize map_api using get_maps_api()
2. Set NUPLAN_MAPS_ROOT environment variable
3. Handle map_name correctly (las_vegas -> us-nv-las-vegas-strip)

Usage:
    # Set environment variable BEFORE importing
    import os
    os.environ['NUPLAN_MAPS_ROOT'] = '/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps'
    
    dataset = CompleteNavsimDataset(
        use_uniad_bev=True,
        extract_labels=True,
        precompute_bev=False
    )
"""

import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
import warnings
from skimage.draw import polygon as draw_polygon

# import visualizer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
# NAVSIM imports
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.common.actor_state.state_representation import Point2D



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
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        # MAP_LOCATIONS is already imported at the top of the file (line 12)
        
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
    
    def extract_all_labels(self, scene, frame_idx: int) -> Dict[str, np.ndarray]:
        """Extract all 12 BEV labels."""
        frame = scene.frames[frame_idx]
        ego_pose = frame.ego_status.ego_pose
        
        try:
            map_api = self._get_map_api(scene)
        except Exception as e:
            warnings.warn(f"Map API failed: {e}")
            return self._get_empty_labels()
        
        labels = {}
        
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
        
        # Traffic lights
        labels['traffic_lights'] = self._extract_traffic_lights(frame, map_api, ego_pose)
        
        # Vehicle classes
        labels['vehicle_classes'] = self._extract_vehicle_classes(frame)
        
        return labels
    
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
    
    def _extract_drivable_area(self, map_api, ego_pose):
        """Extract drivable area from HD map."""
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from nuplan.common.maps.abstract_map_objects import PolygonMapObject
        from shapely.geometry import Point as Point2D
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            # Query roadblocks (drivable areas)
            ego_point = Point2D(x, y)
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
    
    # def _extract_lane_boundaries(self, map_api, ego_pose):
    #     """Extract lane boundaries (edges of lanes)."""
    #     from nuplan.common.maps.abstract_map import SemanticMapLayer
    #     from shapely.geometry import Point as Point2D
        
    #     mask = np.zeros(self.bev_size, dtype=np.float32)
    #     x, y, _ = ego_pose
        
    #     try:
    #         ego_point = Point2D(x, y)
            
    #         # Get lane connectors and lanes
    #         lanes = map_api.get_proximal_map_objects(
    #             ego_point, self.bev_range, [SemanticMapLayer.LANE]
    #         )
            
    #         for lane in lanes.get(SemanticMapLayer.LANE, []):
    #             # Extract left and right boundaries
    #             if hasattr(lane, 'left_boundary') and lane.left_boundary is not None:
    #                 try:
    #                     coords = np.array(lane.left_boundary.linestring.coords)
    #                     if len(coords) > 1:
    #                         pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
    #                         if len(pixel_y) > 0:
    #                             mask[pixel_y, pixel_x] = 1.0
    #                 except:
    #                     pass
                
    #             if hasattr(lane, 'right_boundary') and lane.right_boundary is not None:
    #                 try:
    #                     coords = np.array(lane.right_boundary.linestring.coords)
    #                     if len(coords) > 1:
    #                         pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
    #                         if len(pixel_y) > 0:
    #                             mask[pixel_y, pixel_x] = 1.0
    #                 except:
    #                     pass
                        
    #     except Exception as e:
    #         warnings.warn(f"Lane boundaries extraction error: {e}")
        
    #     return mask
    
    def _extract_lane_boundaries(self, map_api, ego_pose):
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point as ShapelyPoint
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = ShapelyPoint(x, y)
            
            # Get lanes near ego position
            lanes = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE]
            )
            
            lane_objects = lanes.get(SemanticMapLayer.LANE, [])
            
            for lane in lane_objects:
                # Extract left boundary
                try:
                    left_boundary = lane.left_boundary
                    if left_boundary is not None:
                        # Get the linestring from the boundary object
                        linestring = left_boundary.linestring
                        coords = np.array(linestring.coords)
                        
                        if len(coords) > 1:
                            pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                            if len(pixel_y) > 0:
                                # Filter valid pixels
                                valid_mask = (pixel_y >= 0) & (pixel_y < self.bev_size[0]) & \
                                        (pixel_x >= 0) & (pixel_x < self.bev_size[1])
                                pixel_y = pixel_y[valid_mask]
                                pixel_x = pixel_x[valid_mask]
                                if len(pixel_y) > 0:
                                    mask[pixel_y, pixel_x] = 1.0
                except Exception as e:
                    # Silently skip problematic boundaries
                    pass
                
                # Extract right boundary
                try:
                    right_boundary = lane.right_boundary
                    if right_boundary is not None:
                        # Get the linestring from the boundary object
                        linestring = right_boundary.linestring
                        coords = np.array(linestring.coords)
                        
                        if len(coords) > 1:
                            pixel_y, pixel_x = self._world_to_bev(coords, ego_pose)
                            if len(pixel_y) > 0:
                                # Filter valid pixels
                                valid_mask = (pixel_y >= 0) & (pixel_y < self.bev_size[0]) & \
                                        (pixel_x >= 0) & (pixel_x < self.bev_size[1])
                                pixel_y = pixel_y[valid_mask]
                                pixel_x = pixel_x[valid_mask]
                                if len(pixel_y) > 0:
                                    mask[pixel_y, pixel_x] = 1.0
                except Exception as e:
                    # Silently skip problematic boundaries
                    pass
                    
        except Exception as e:
            warnings.warn(f"Lane boundaries extraction error: {e}")
        
        return mask
    
    def _extract_lane_dividers(self, map_api, ego_pose):
        """Extract lane dividers (centerlines between lanes)."""
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point as Point2D
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = Point2D(x, y)
            
            # Get baseline paths (centerlines)
            lanes = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE]
            )
            
            for lane in lanes.get(SemanticMapLayer.LANE, []):
                if hasattr(lane, 'baseline_path'):
                    try:
                        # Get discrete path points
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
        
        vehicle_count = 0
        added_count = 0
        
        for i, name in enumerate(names):
            if 'vehicle' in name.lower() or 'car' in name.lower() or 'truck' in name.lower():
                vehicle_count += 1
                box = boxes[i]
                
                # Boxes are already in ego frame!
                x, y = box[0], box[1]  # ego-relative coordinates
                yaw = box[6]
                w, l = box[3], box[4]
                
                # Check distance in ego frame
                dist = np.sqrt(x**2 + y**2)
                
                if dist > self.bev_range:
                    if vehicle_count <= 5:
                        print(f"  DEBUG: Vehicle {i} at {dist:.1f}m - OUTSIDE range")
                    continue
                
                # Get corners in ego frame
                corners = self._get_box_corners(x, y, yaw, w, l)
                
                # Convert directly to BEV pixels (no ego pose transformation needed!)
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = 1.0
                    added_count += 1
                    if added_count <= 3:
                        print(f"  DEBUG: Vehicle {i} at {dist:.1f}m - ADDED {len(rr)} pixels")
        
        print(f"  DEBUG: Found {vehicle_count} vehicles, added {added_count} to BEV")
        return mask

    def _ego_to_bev_pixels(self, ego_coords):
        """Convert ego-frame coordinates directly to BEV pixels.
        
        Ego frame: x=forward, y=left
        BEV image: row 0 = far ahead, row 199 = behind
                col 0 = left, col 199 = right
        """
        H, W = self.bev_size
        
        # Convert ego coordinates to pixel indices
        pixel_x = ((ego_coords[:, 1] + self.bev_range) / (2 * self.bev_range) * W).astype(int)
        pixel_y = ((self.bev_range - ego_coords[:, 0]) / (2 * self.bev_range) * H).astype(int)
        
        # Filter valid pixels
        valid = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
        
        return pixel_y[valid], pixel_x[valid]
    
    def _extract_pedestrian_occupancy(self, frame):
        """Extract pedestrian occupancy."""
        mask = np.zeros(self.bev_size, dtype=np.float32)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return mask
        
        boxes = frame.annotations.boxes
        names = frame.annotations.names
        # ego_pose = frame.ego_status.ego_pose
        
        ped_count = 0
        added_count = 0
        
        for i, name in enumerate(names):
            if 'pedestrian' in name.lower() or 'person' in name.lower():
                ped_count += 1
                box = boxes[i]
                x, y = box[0], box[1]
                yaw = box[6]
                w, l = box[3], box[4]
                # Check distance in ego frame
                dist = np.sqrt(x**2 + y**2)
                
                if dist > self.bev_range:
                    if ped_count <= 5:
                        print(f"  DEBUG: Vehicle {i} at {dist:.1f}m - OUTSIDE range")
                    continue
                # get corner in ego frame
                corners = self._get_box_corners(x, y, yaw, w, l)
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = 1.0
                    added_count += 1
                    if added_count <= 3:
                        print(f"  DEBUG: Vehicle {i} at {dist:.1f}m - ADDED {len(rr)} pixels")
    
        
        return mask
    
    def _extract_velocity_fields(self, frame):
        """Extract velocity fields from annotations."""
        velocity_x = np.zeros(self.bev_size, dtype=np.float32)
        velocity_y = np.zeros(self.bev_size, dtype=np.float32)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return velocity_x, velocity_y
        
        boxes = frame.annotations.boxes
        
        # Check if velocity_3d exists
        if not hasattr(frame.annotations, 'velocity_3d') or frame.annotations.velocity_3d is None:
            return velocity_x, velocity_y
            
        velocities = frame.annotations.velocity_3d
        # ego_pose = frame.ego_status.ego_pose
        # ego_heading = ego_pose[2]
        # vel_count = 0
        # added_count = 0
        for i in range(len(boxes)):

            box = boxes[i]
            
            #boxes are in ego frame, velocities shoud be to
            x,  y = box[0], box[1]
            # check if within range
            dist = np.sqrt(x**2 + y**2)
            if dist > self.bev_range:
                continue
            #get velocity in ego frame
            vx_ego, vy_ego = velocities[i, 0], velocities[i, 1]

            # covert pos
            position = np.array([[x, y]])

            pixel_y, pixel_x = self._ego_to_bev_pixels(position)


            if len(pixel_y) > 0:
                # Spread velocity to nearby pixels
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
        
        # Ego vehicle dimensions (typical sedan)
        ego_width, ego_length = 2.0, 4.5
        
        # Ego is at origin in local frame
        x, y, heading = 0, 0, 0  # Local frame
        corners = self._get_box_corners(x, y, heading, ego_width, ego_length)
        
        # Convert to pixels (ego at center)
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
        
        # Check if traffic lights exist in frame
        if not hasattr(frame, 'traffic_lights') or frame.traffic_lights is None or len(frame.traffic_lights) == 0:
            return mask
            
        try:
            from nuplan.common.maps.abstract_map import SemanticMapLayer
            from shapely.geometry import Point as Point2D
            
            x, y, _ = ego_pose
            ego_point = Point2D(x, y)
            
            # Get traffic light lane connectors
            lane_connectors = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.LANE_CONNECTOR]
            )
            
            # Create dict of traffic light states - FIXED: handle integer IDs
            traffic_light_dict = {}
            for light_id, is_green in frame.traffic_lights:
                traffic_light_dict[light_id] = is_green
            
            connector_count = 0
            matched_count = 0
            
            for connector in lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, []):
                connector_count += 1
                
                # FIXED: Convert connector.id to integer for matching
                try:
                    # Extract numeric ID from connector
                    if hasattr(connector, 'id'):
                        connector_id_str = str(connector.id)
                        # Try to extract integer from ID (may have prefix)
                        import re
                        numbers = re.findall(r'\d+', connector_id_str)
                        if numbers:
                            connector_id = int(numbers[-1])  # Get last number
                        else:
                            continue
                    else:
                        continue
                    
                    # Check if this connector has a traffic light in our data
                    if connector_id in traffic_light_dict:
                        matched_count += 1
                        is_green = traffic_light_dict[connector_id]
                        
                        # Get connector path
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
                                        # 1=red, 2=green
                                        value = 2 if is_green else 1
                                        mask[pixel_y, pixel_x] = value
                            except Exception as e:
                                pass
                except:
                    continue
            
            print(f"  DEBUG: Traffic lights - Found {connector_count} lane connectors, matched {matched_count} with traffic lights, non-zero pixels: {(mask > 0).sum()}")
                
        except Exception as e:
            warnings.warn(f"Traffic lights extraction error: {e}")
        
        return mask
    
    def _extract_vehicle_classes(self, frame):
        """Extract vehicle classes (0=background, 1=car, 2=truck, 3=bus, etc)."""
        mask = np.zeros(self.bev_size, dtype=np.uint8)
        
        if frame.annotations is None or len(frame.annotations.boxes) == 0:
            return mask
        
        boxes = frame.annotations.boxes
        names = frame.annotations.names
        # ego_pose = frame.ego_status.ego_pose
        
        # Map class names to IDs
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
                # FIXED: Correct box indexing [x, y, z, width, length, height, heading]
                x, y = box[0], box[1]
                yaw = box[6]  # heading at index 6
                w, l = box[3], box[4]
                
                # Check distance
                dist = np.sqrt(x**2 + y**2)
                if dist > self.bev_range:
                    continue
                
                # Get corners in ego frame
                corners = self._get_box_corners(x, y, yaw, w, l)
                
                # Convert to BEV pixels
                pixel_y, pixel_x = self._ego_to_bev_pixels(corners)
                
                if len(pixel_y) > 2:
                    rr, cc = draw_polygon(pixel_y, pixel_x, shape=self.bev_size)
                    mask[rr, cc] = class_id
        
        return mask
    
    def _extract_crosswalks(self, map_api, ego_pose):
        """Extract crosswalks."""
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point as Point2D
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = Point2D(x, y)
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
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point as Point2D
        
        mask = np.zeros(self.bev_size, dtype=np.float32)
        x, y, _ = ego_pose
        
        try:
            ego_point = Point2D(x, y)
            stops = map_api.get_proximal_map_objects(
                ego_point, self.bev_range, [SemanticMapLayer.STOP_LINE]
            )
            
            for stop in stops.get(SemanticMapLayer.STOP_LINE, []):
                # Stop lines can be polygons or linestrings
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
        """Get 4 corners of a bounding box.
        
        Note: In NAVSIM, box format is [x, y, z, width, length, height, heading]
              where heading is at index 6.
        """
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Corners in local frame (centered at box center)
        corners_local = np.array([
            [-length/2, -width/2],
            [-length/2, +width/2],
            [+length/2, +width/2],
            [+length/2, -width/2],
        ])
        
        # Rotate and translate to world frame
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



class NavsimDataset(Dataset):
    """
    Complete NAVSIM dataset with proper map loading.
    """
    
    def __init__(
        self,
        bev_size: Tuple[int, int] = (64, 64),
        bev_range: float = 50.0,
        history_length: int = 4,
        future_horizon: int = 8,
        extract_labels: bool = True,
        use_cache: bool = True,
        map_root: str = None,
        map_version: str = "nuplan-maps-v1.0",
        use_uniad_bev: bool = True,  # Changed default to False
        uniad_cache_dir: str = None, # Path to precompted BEV cache
        interpolate_bev: bool = False, # to upsample UniAD features, to equal BEV features extract from HD map
    ):
        """
        Initialize complete NAVSIM dataset.
        
        Args:
            bev_size: (H, W) for BEV grids
            bev_range: Range in meters
            history_length: Number of history frames
            future_horizon: Number of future waypoints         
            extract_labels: Whether to extract 12 BEV labels
            use_cache: Whether to cache samples
            map_root: Path to maps directory (if None, uses NUPLAN_MAPS_ROOT env var)
            map_version: Map version string
            use_uniad_bev: Whether to use UniAD BEV encoder
            uniad_cache_dir: Directory containing cached UniAD features
            interpolate_bev: Whether to upsample UniAD to match bev_size
                           - True: Upsample UniAD (64,64)->(200,200), slower but aligned
                           - False: Keep native UniAD (64,64), faster, mixed resolution
        """
        data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
        
        # Set up map paths
        if map_root is None:
            map_root = os.environ.get('NUPLAN_MAPS_ROOT')
            if map_root is None:
                raise ValueError(
                    "Map root not specified! Either:\n"
                    "1. Pass map_root parameter, or\n"
                    "2. Set NUPLAN_MAPS_ROOT environment variable\n"
                    "Example: export NUPLAN_MAPS_ROOT=/path/to/maps"
                )
        
        self.map_root = map_root
        self.map_version = map_version
        
        self.bev_size = bev_size
        self.bev_range = bev_range
        self.history_length = history_length
        self.future_horizon = future_horizon
        self.use_uniad_bev = use_uniad_bev
        self.extract_labels = extract_labels
        self.use_cache = use_cache
        self.use_uniad_bev = use_uniad_bev
        self.interpolate_bev = interpolate_bev
        self.uniad_cache_dir = uniad_cache_dir

        self.cache_dir = data_root / 'cache' / 'complete_navsim'
        
        print("=" * 70)
        print("Initializing Complete NAVSIM Dataset")
        print("=" * 70)
        print(f"BEV size: {bev_size}, range: {bev_range}m")
        print(f"Map root: {self.map_root}")
        print(f"Map version: {self.map_version}")
        print(f"UniAD BEV: {use_uniad_bev}")
        print(f"Extract labels: {extract_labels}")
        print("=" * 70)
        
        # Verify map directory exists
        if not Path(self.map_root).exists():
            raise FileNotFoundError(
                f"Map directory not found: {self.map_root}\n"
                f"Please ensure maps are downloaded and the path is correct."
            )
        
        # Sensor config
        sensor_config = SensorConfig(
            cam_f0=False,  # We'll load this manually if needed
            cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False,
            cam_b0=False,
            lidar_pc=True
        )
        
        # Scene filter
        scene_filter = SceneFilter(
            log_names=None,
            num_history_frames=history_length,
            num_future_frames=future_horizon,
        )
        
        # Scene loader
        self.scene_loader = SceneLoader(
            data_path=data_root / 'mini_navsim_logs' / 'mini',
            original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        
        self.scene_tokens = self.scene_loader.tokens
        
        # Label extractor with map support
        if self.extract_labels:
            self.label_extractor = BEVLabelExtractor(
                bev_size, 
                bev_range,
                map_root=self.map_root,
                map_version=self.map_version
            )
            print(f"  Label extractor ready (12 channels)")
        else:
            self.label_extractor = None
        
        # Cache
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
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
                print(f"  Using precomputed UniAD BEV from: {self.uniad_cache_dir}")
                # Detect BEV feature dimensions from first available file
                self._detect_bev_dimensions()
        else:
            self.bev_channels = 32  # Fallback placeholder channels
            self.uniad_bev_size = bev_size
              
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

        print(f"\n  Loaded {len(self)} scenes")
        print("=" * 70)
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

    def __len__(self):
        return len(self.scene_tokens)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get complete sample with all modalities."""
        token = self.scene_tokens[idx]
        
        # Check cache
        if self.use_cache:
            cache_file = self.cache_dir / f'{token}.pt'
            if cache_file.exists():
                return torch.load(cache_file)
        
        # Load scene
        scene = self.scene_loader.get_scene_from_token(token)
        
        # Process
        sample = self._process_scene(scene, token)
        
        # Cache
        if self.use_cache:
            torch.save(sample, self.cache_dir / f'{token}.pt')
        
        return sample
     
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
            print(f"⚠ Error loading UniAD BEV for {token}: {e}")
            return None
        
    def _process_scene(self, scene: Scene, token: str) -> Dict:
        """Process scene to extract all modalities."""
        current_frame = scene.frames[self.history_length - 1]
        
        # ===================================================================
        # 1. LiDAR BEV (always compute)
        # ===================================================================
        if current_frame.lidar.lidar_pc is not None:
            lidar_pc = current_frame.lidar.lidar_pc[:3, :].T
            lidar_bev = self._rasterize_lidar(lidar_pc)
        else:
            lidar_bev = torch.zeros(2, *self.bev_size)
        
        # ===================================================================
        # 2. Camera BEV (placeholder for now)
        # ===================================================================
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
        
        # ===================================================================
        # 3. BEV Labels (12 channels from HD map)
        # ===================================================================
        if self.extract_labels and self.label_extractor is not None:
            try:
                labels = self.label_extractor.extract_all_labels(
                    scene, self.history_length - 1
                )
                # Convert to tensors
                labels_tensor = {
                    k: torch.from_numpy(v).float() 
                    for k, v in labels.items()
                }
            except Exception as e:
                print(f"⚠ Label extraction failed for {token}: {e}")
                labels_tensor = self._get_empty_labels()
        else:
            labels_tensor = {}
        
        # ===================================================================
        # 4. Agent States
        # ===================================================================
        ego = current_frame.ego_status
        agent_states = torch.tensor(
            [[ego.ego_pose[0], ego.ego_pose[1], ego.ego_velocity[0],
              ego.ego_velocity[1], ego.ego_pose[2]]],
            dtype=torch.float32
        )
        
        # ===================================================================
        # 5. Agent History
        # ===================================================================
        history_states = []
        for i in range(self.history_length):
            ego = scene.frames[i].ego_status
            state = torch.tensor(
                [[ego.ego_pose[0], ego.ego_pose[1], ego.ego_velocity[0],
                  ego.ego_velocity[1], ego.ego_pose[2]]],
                dtype=torch.float32
            )
            history_states.append(state)
        agent_history = torch.cat(history_states, dim=0).unsqueeze(0)
        
        # ===================================================================
        # 6. Ground Truth Trajectory
        # ===================================================================
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=self.future_horizon
        )
        poses = future_traj.poses
        
        waypoints = []
        current_ego = current_frame.ego_status
        
        for i in range(len(poses)):
            x, y, heading = poses[i]
            
            if i > 0:
                prev_x, prev_y, _ = poses[i-1]
                dt = 0.5
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
            else:
                vx = current_ego.ego_velocity[0]
                vy = current_ego.ego_velocity[1]
            
            waypoints.append([x, y, vx, vy, heading])
        
        gt_trajectory = torch.tensor(waypoints, dtype=torch.float32).unsqueeze(0)
        
        # ===================================================================
        # 7. Assemble Complete Sample
        # ===================================================================
        return {
            # BEV features
            'camera_bev': camera_bev,  # [32, 200, 200]
            'lidar_bev': lidar_bev,    # [2, 200, 200]
            
            # HD Map labels (12 channels)
            'labels': labels_tensor,    # Dict with 12 keys
            
            # Agent info
            'agent_states': agent_states,      # [1, 5]
            'agent_history': agent_history,    # [1, 4, 5]
            'gt_trajectory': gt_trajectory,    # [1, 8, 5]
            
            # Metadata
            'token': token,
        }
    
    def _rasterize_lidar(self, point_cloud: np.ndarray) -> torch.Tensor:
        """Rasterize LiDAR to BEV grid."""
        H, W = self.bev_size
        
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2] if point_cloud.shape[1] >= 3 else np.zeros_like(x)
        
        # Grid bounds
        x_min, x_max = -self.bev_range, self.bev_range
        y_min, y_max = -self.bev_range, self.bev_range
        
        # Convert to indices
        x_indices = ((x - x_min) / (x_max - x_min) * W).astype(int)
        y_indices = ((y - y_min) / (y_max - y_min) * H).astype(int)
        
        # Filter valid
        valid = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        z = z[valid]
        
        # Create grids
        density = np.zeros((H, W), dtype=np.float32)
        height = np.zeros((H, W), dtype=np.float32)
        
        for i in range(len(x_indices)):
            density[y_indices[i], x_indices[i]] += 1
            height[y_indices[i], x_indices[i]] = max(
                height[y_indices[i], x_indices[i]], z[i]
            )
        
        # Normalize
        density = np.clip(density / 10.0, 0, 1)
        height = np.clip((height + 2) / 5.0, 0, 1)
        
        return torch.from_numpy(np.stack([density, height])).float()
    
    def _get_empty_labels(self) -> Dict[str, torch.Tensor]:
        """Return empty label dict."""
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


def collate_fn(batch):
    """Collate function for DataLoader."""
    
    # Stack BEV features
    camera_bev = torch.stack([item['camera_bev'] for item in batch])
    lidar_bev = torch.stack([item['lidar_bev'] for item in batch])
    
    # Stack labels (if present)
    if batch[0]['labels']:
        labels = {}
        for key in batch[0]['labels'].keys():
            labels[key] = torch.stack([item['labels'][key] for item in batch])
    else:
        labels = {}
    
    return {
        'camera_bev': camera_bev,
        'lidar_bev': lidar_bev,
        'labels': labels,
        'agent_states': torch.cat([item['agent_states'] for item in batch], dim=0),
        'agent_history': torch.cat([item['agent_history'] for item in batch], dim=0),
        'gt_trajectory': torch.cat([item['gt_trajectory'] for item in batch], dim=0),
        'tokens': [item['token'] for item in batch],
    }

# for testing
def visualize_bev_labels(sample, save_path=None, show=True):
    """
    Visualize all 12 BEV label channels in a single figure.
    
    Args:
        sample: Dataset sample containing 'labels' dict
        save_path: Optional path to save the visualization
        show: Whether to display the plot
    """
    labels = sample['labels']
    
    # Create figure with 4x3 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('NAVSIM BEV Labels Visualization', fontsize=16, fontweight='bold')
    
    # Define colormap for each channel
    channel_configs = [
        # Row 1: Map features
        {
            'name': 'drivable_area',
            'title': 'Drivable Area',
            'cmap': 'Greys',
            'vmin': 0, 'vmax': 1
        },
        {
            'name': 'lane_boundaries',
            'title': 'Lane Boundaries',
            'cmap': 'Blues',
            'vmin': 0, 'vmax': 1
        },
        {
            'name': 'lane_dividers',
            'title': 'Lane Dividers',
            'cmap': 'Purples',
            'vmin': 0, 'vmax': 1
        },
        {
            'name': 'crosswalks',
            'title': 'Crosswalks',
            'cmap': 'Oranges',
            'vmin': 0, 'vmax': 1
        },
        
        # Row 2: Dynamic objects
        {
            'name': 'vehicle_occupancy',
            'title': 'Vehicle Occupancy',
            'cmap': 'Reds',
            'vmin': 0, 'vmax': 1
        },
        {
            'name': 'pedestrian_occupancy',
            'title': 'Pedestrian Occupancy',
            'cmap': 'Greens',
            'vmin': 0, 'vmax': 1
        },
        {
            'name': 'vehicle_classes',
            'title': 'Vehicle Classes\n(1=car, 2=truck, 3=bus)',
            'cmap': 'tab10',
            'vmin': 0, 'vmax': 5
        },
        {
            'name': 'ego_mask',
            'title': 'Ego Vehicle',
            'cmap': 'Greys',
            'vmin': 0, 'vmax': 1
        },
        
        # Row 3: Velocity and traffic
        {
            'name': 'velocity_x',
            'title': 'Velocity X (forward)',
            'cmap': 'RdBu_r',
            'vmin': -10, 'vmax': 10
        },
        {
            'name': 'velocity_y',
            'title': 'Velocity Y (lateral)',
            'cmap': 'RdBu_r',
            'vmin': -10, 'vmax': 10
        },
        {
            'name': 'traffic_lights',
            'title': 'Traffic Lights\n(1=red, 2=green)',
            'cmap': ListedColormap(['black', 'red', 'green']),
            'vmin': 0, 'vmax': 2
        },
        {
            'name': 'stop_lines',
            'title': 'Stop Lines',
            'cmap': 'YlOrRd',
            'vmin': 0, 'vmax': 1
        },
    ]
    
    # Plot each channel
    for idx, config in enumerate(channel_configs):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        name = config['name']
        if name in labels:
            data = labels[name].numpy() if isinstance(labels[name], torch.Tensor) else labels[name]
            
            # Plot
            im = ax.imshow(
                data,
                cmap=config['cmap'],
                vmin=config['vmin'],
                vmax=config['vmax'],
                origin='upper',
                interpolation='nearest'
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Count non-zero pixels
            non_zero = (data != 0).sum()
            total = data.size
            percentage = (non_zero / total) * 100
            
            # Set title with statistics
            ax.set_title(
                f"{config['title']}\n{non_zero}/{total} ({percentage:.1f}%)",
                fontsize=10,
                fontweight='bold'
            )
        else:
            ax.text(0.5, 0.5, f'{name}\nNot Available', 
                ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config['title'], fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('X (pixels)', fontsize=8)
        ax.set_ylabel('Y (pixels)', fontsize=8)
        
        # Add range labels
        ax.text(0.02, 0.98, '50m ahead', transform=ax.transAxes,
            fontsize=7, va='top', ha='left', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.text(0.98, 0.98, '50m right', transform=ax.transAxes,
            fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_composite_bev(sample, save_path=None, show=True):
    """
    Create a composite overlay visualization of multiple channels.
    """
    labels = sample['labels']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('NAVSIM BEV Composite Views', fontsize=14, fontweight='bold')
    
    # --- Panel 1: HD Map layers ---
    ax = axes[0]
    composite = np.zeros((*labels['drivable_area'].shape, 3))
    
    # Drivable area (gray)
    drivable = labels['drivable_area'].numpy()
    composite[:, :, 0] += drivable * 0.3
    composite[:, :, 1] += drivable * 0.3
    composite[:, :, 2] += drivable * 0.3
    
    # Lane boundaries (blue)
    boundaries = labels['lane_boundaries'].numpy()
    composite[:, :, 2] += boundaries * 0.8
    
    # Lane dividers (cyan)
    dividers = labels['lane_dividers'].numpy()
    composite[:, :, 1] += dividers * 0.5
    composite[:, :, 2] += dividers * 0.5
    
    # Crosswalks (yellow)
    crosswalks = labels['crosswalks'].numpy()
    composite[:, :, 0] += crosswalks * 0.6
    composite[:, :, 1] += crosswalks * 0.6
    
    # Stop lines (red)
    stops = labels['stop_lines'].numpy()
    composite[:, :, 0] += stops * 1.0
    
    ax.imshow(np.clip(composite, 0, 1), origin='upper')
    ax.set_title('HD Map Layers\n(drivable + lanes + crosswalks + stops)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: Dynamic objects ---
    ax = axes[1]
    composite = np.zeros((*labels['vehicle_occupancy'].shape, 3))
    
    # Drivable area (faint gray background)
    composite[:, :, 0] += drivable * 0.2
    composite[:, :, 1] += drivable * 0.2
    composite[:, :, 2] += drivable * 0.2
    
    # Vehicles (red)
    vehicles = labels['vehicle_occupancy'].numpy()
    composite[:, :, 0] += vehicles * 1.0
    
    # Pedestrians (green)
    pedestrians = labels['pedestrian_occupancy'].numpy()
    composite[:, :, 1] += pedestrians * 1.0
    
    # Ego vehicle (blue)
    ego = labels['ego_mask'].numpy()
    composite[:, :, 2] += ego * 1.0
    
    ax.imshow(np.clip(composite, 0, 1), origin='upper')
    ax.set_title('Dynamic Objects\n(vehicles=red, pedestrians=green, ego=blue)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Traffic control ---
    ax = axes[2]
    composite = np.zeros((*labels['drivable_area'].shape, 3))
    
    # Drivable area (background)
    composite[:, :, 0] += drivable * 0.2
    composite[:, :, 1] += drivable * 0.2
    composite[:, :, 2] += drivable * 0.2
    
    # Traffic lights
    traffic = labels['traffic_lights'].numpy()
    red_lights = (traffic == 1).astype(float)
    green_lights = (traffic == 2).astype(float)
    
    composite[:, :, 0] += red_lights * 1.0  # Red
    composite[:, :, 1] += green_lights * 1.0  # Green
    
    # Stop lines (yellow)
    composite[:, :, 0] += stops * 0.8
    composite[:, :, 1] += stops * 0.8
    
    ax.imshow(np.clip(composite, 0, 1), origin='upper')
    ax.set_title('Traffic Control\n(red lights, green lights, stop lines)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        composite_path = str(save_path).replace('.png', '_composite.png')
        plt.savefig(composite_path, dpi=150, bbox_inches='tight')
        print(f"  Saved composite to: {composite_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def print_label_statistics(sample):
    """Print detailed statistics for all labels."""
    labels = sample['labels']
    
    print("\n" + "=" * 70)
    print("  DETAILED BEV LABEL STATISTICS")
    print("=" * 70)
    
    for name, data in labels.items():
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        non_zero = (data != 0).sum()
        total = data.size
        percentage = (non_zero / total) * 100
        
        if data.dtype in [np.float32, np.float64]:
            min_val = data.min()
            max_val = data.max()
            mean_val = data.mean()
            print(f"\n{name:25s}:")
            print(f"  Non-zero: {non_zero:6d} / {total:6d} ({percentage:5.2f}%)")
            print(f"  Range:    [{min_val:8.3f}, {max_val:8.3f}]")
            print(f"  Mean:     {mean_val:8.3f}")
        else:
            unique_vals = np.unique(data)
            print(f"\n{name:25s}:")
            print(f"  Non-zero: {non_zero:6d} / {total:6d} ({percentage:5.2f}%)")
            print(f"  Unique values: {unique_vals}")
    
    print("\n" + "=" * 70)
# ============================================================
# Test Script
# ============================================================

if __name__ == "__main__":
    import os
    
    print("=" * 70)
    print("Complete NAVSIM Dataset Test - FIXED MAP LOADING")
    print("=" * 70)
    
    # IMPORTANT: Set map root
    map_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
    os.environ['NUPLAN_MAPS_ROOT'] = map_root
    
    print(f"\n  Set NUPLAN_MAPS_ROOT to: {map_root}")
    print(f"  Verifying map directory exists...")
    
    if not Path(map_root).exists():
        print(f"  ERROR: Map directory not found at {map_root}")
        print("Please check the path and try again.")
        exit(1)
    
    print(f"  Map directory found!")
    
    # List available maps
    print(f"\n📁 Maps in directory:")
    for item in Path(map_root).iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
    
    # Create dataset
    from torch.utils.data import DataLoader
    
    dataset = NavsimDataset(
        bev_size=(200, 200),
        bev_range=50.0,
        use_uniad_bev=False,
        extract_labels=True,
        use_cache=False,
        map_root=map_root
    )
    
    print(f"\n  Dataset created: {len(dataset)} scenes")
    
    # Test single sample
    print("\nTesting single sample...")
    sample = dataset[0]
    
    print("\n  Sample Contents:")
    print(f"  Camera BEV:        {sample['camera_bev'].shape}")
    print(f"  LiDAR BEV:         {sample['lidar_bev'].shape}")
    print(f"  Agent states:      {sample['agent_states'].shape}")
    print(f"  Agent history:     {sample['agent_history'].shape}")
    print(f"  GT trajectory:     {sample['gt_trajectory'].shape}")
    
    if sample['labels']:
        print(f"\n📋 BEV Labels (12 channels):")
        for key, val in sample['labels'].items():
            non_zero = (val > 0).sum().item()
            print(f"    {key:25s}: {val.shape} | non-zero: {non_zero}")
    print("\n" + "=" * 70)
    print("🔍 COORDINATE SYSTEM INVESTIGATION")
    print("=" * 70)

    scene = dataset.scene_loader.get_scene_from_token(dataset.scene_tokens[0])
    frame = scene.frames[dataset.history_length - 1]

    print("\n  Ego Pose:")
    ego_pose = frame.ego_status.ego_pose
    print(f"  x={ego_pose[0]:.2f}, y={ego_pose[1]:.2f}, heading={ego_pose[2]:.4f} rad")

    print("\n📦 First 3 Vehicle Boxes:")
    if frame.annotations and len(frame.annotations.boxes) > 0:
        for i in range(min(3, len(frame.annotations.boxes))):
            if 'vehicle' in frame.annotations.names[i].lower():
                box = frame.annotations.boxes[i]
                print(f"  Vehicle {i}: x={box[0]:.2f}, y={box[1]:.2f}, z={box[2]:.2f}")
                print(f"             width={box[3]:.2f}, length={box[4]:.2f}, height={box[5]:.2f}, heading={box[6]:.4f}")

    print("\n🎯 LiDAR Point Cloud Stats:")
    if frame.lidar.lidar_pc is not None:
        pc = frame.lidar.lidar_pc
        print(f"  Shape: {pc.shape}")
        print(f"  X range: [{pc[0].min():.2f}, {pc[0].max():.2f}]")
        print(f"  Y range: [{pc[1].min():.2f}, {pc[1].max():.2f}]")
        print(f"  Z range: [{pc[2].min():.2f}, {pc[2].max():.2f}]")

    print("\n" + "=" * 70)
    print("🚦 TRAFFIC LIGHT INVESTIGATION")
    print("=" * 70)

    scene = dataset.scene_loader.get_scene_from_token(dataset.scene_tokens[0])
    frame = scene.frames[dataset.history_length - 1]

    print(f"\nFrame has traffic_lights attribute: {hasattr(frame, 'traffic_lights')}")
    if hasattr(frame, 'traffic_lights'):
        print(f"Traffic lights value: {frame.traffic_lights}")
        if frame.traffic_lights:
            print(f"Traffic lights type: {type(frame.traffic_lights)}")
            print(f"Number of traffic lights: {len(frame.traffic_lights)}")
            if len(frame.traffic_lights) > 0:
                print(f"First 3 traffic lights: {frame.traffic_lights[:3]}")

    # Check map for traffic light lane connectors
    ego_pose = frame.ego_status.ego_pose
    try:
        map_api = dataset.label_extractor._get_map_api(scene)
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from shapely.geometry import Point as Point2D
        
        x, y, _ = ego_pose
        ego_point = Point2D(x, y)
        
        lane_connectors = map_api.get_proximal_map_objects(
            ego_point, dataset.bev_range, [SemanticMapLayer.LANE_CONNECTOR]
        )
        
        print(f"\nLane connectors found: {len(lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, []))}")
        
        tl_count = 0
        for connector in lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])[:5]:
            print(f"\nConnector ID: {connector.id}")
            print(f"  Has traffic_light_status: {hasattr(connector, 'traffic_light_status')}")
            if hasattr(connector, 'traffic_light_status'):
                print(f"  Traffic light status: {connector.traffic_light_status}")
                tl_count += 1
        
        print(f"\nTotal connectors with traffic lights: {tl_count}")
    except Exception as e:
        print(f"Error checking traffic lights: {e}")
    print("\n🔍 Checking Connector ID Format:")
    for connector in lane_connectors.get(SemanticMapLayer.LANE_CONNECTOR, [])[:3]:
        print(f"  Connector ID: {connector.id} (type: {type(connector.id)})")
        print(f"  ID as string: '{str(connector.id)}'")
        print("\n" + "=" * 70)
    print("🎨 VISUALIZATION")
    print("=" * 70)
    
    # Create output directory
    vis_dir = Path('./visualizations')
    vis_dir.mkdir(exist_ok=True)
    
    # Test multiple samples
    for idx in [0, 10, 20]:
        print(f"\n--- Visualizing Sample {idx} ---")
        sample = dataset[idx]
        
        # Print statistics
        print_label_statistics(sample)
        
        # Create visualizations
        visualize_bev_labels(
            sample,
            save_path=vis_dir / f'sample_{idx}_all_channels.png',
            show=False  # Set to True if you want to see interactively
        )
        
        visualize_composite_bev(
            sample,
            save_path=vis_dir / f'sample_{idx}_composite.png',
            show=False
        )
    
    print(f"\n  Visualizations saved to: {vis_dir.absolute()}")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("  Test completed!")
    print("=" * 70)

# result of taking 
#  BEV Labels (12 channels):
#     drivable_area            : torch.Size([200, 200]) | non-zero: 4464
#     lane_boundaries          : torch.Size([200, 200]) | non-zero: 161
#     lane_dividers            : torch.Size([200, 200]) | non-zero: 671
#     vehicle_occupancy        : torch.Size([200, 200]) | non-zero: 1554
#     pedestrian_occupancy     : torch.Size([200, 200]) | non-zero: 561
#     crosswalks               : torch.Size([200, 200]) | non-zero: 3421
#     stop_lines               : torch.Size([200, 200]) | non-zero: 50
#     velocity_x               : torch.Size([200, 200]) | non-zero: 804
#     velocity_y               : torch.Size([200, 200]) | non-zero: 839
#     ego_mask                 : torch.Size([200, 200]) | non-zero: 50
#     traffic_lights           : torch.Size([200, 200]) | non-zero: 2844
#     vehicle_classes          : torch.Size([200, 200]) | non-zero: 1554


# 1. Sparse Occupancy (This is realistic!)
# Most of the BEV grid is empty space:
#     • Drivable area (4,464 pixels): ~11% of the grid is drivable road 
#     • Vehicle occupancy (1,554 pixels): ~4% - only a few vehicles nearby 
#     • Pedestrians (561 pixels): ~1.4% - even fewer pedestrians 
# This makes sense! In a 100m×100m area (50m radius), most space is NOT occupied by vehicles.
# 2. Sparse Line Features
#     • Lane boundaries (161 pixels): These are thin lines, not filled areas 
#     • Lane dividers (671 pixels): Centerlines are also thin 
#     • Stop lines (50 pixels): Very small features 
# 3. Expected Ranges by Channel Type
# Channel
# Typical %
# Why
# Drivable area # 5-20%  -> # Only roads, not buildings/grass
# Crosswalks  # 5-15% -> # Large painted areas
# Vehicle occupancy # 1-5% -> # Few cars at any moment
# Pedestrians # 0.5-2% -> # Even fewer people
# Lane markers # 0.5-3% ->  # Thin lines
# Velocity fields # 1-5% ->  # Only where agents exist
# Ego mask  # 0.1-0.5%  -> # Just your car (~4.5m × 2m)

# 4. Specific Numbers Analysis
# drivable_area:      4,464 / 40,000 = 11.2%    Normal (city driving)
# vehicle_occupancy:  1,554 / 40,000 = 3.9%     Good (30 vehicles nearby)
# crosswalks:         3,421 / 40,000 = 8.6%     Urban area with crossings
# traffic_lights:     2,844 / 40,000 = 7.1%     Many controlled intersections
# lane_boundaries:      161 / 40,000 = 0.4%     Thin lines
# pedestrians:          561 / 40,000 = 1.4%     Urban scene
# ego_mask:              50 / 40,000 = 0.1%     Single car footprint