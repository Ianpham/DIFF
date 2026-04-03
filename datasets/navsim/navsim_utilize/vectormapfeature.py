"""
 Vector Map Extractor for NAVSIM
Adds: Intersection detection, traffic lights, crosswalks, stop lines, occlusion reasoning
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
import geopandas as gpd


@dataclass
class VectorMapFeatures:
    """
     container for vector map features.
    Includes: lanes, intersections, traffic lights, crosswalks, stop lines, occlusion
    """
    # GeoDataFrame format (for debugging, visualization)
    lanes_gdf: gpd.GeoDataFrame
    lane_connectors_gdf: gpd.GeoDataFrame
    intersections_gdf: gpd.GeoDataFrame
    stop_lines_gdf: Optional[gpd.GeoDataFrame]
    crosswalks_gdf: Optional[gpd.GeoDataFrame]
    
    # === CORE LANE FEATURES ===
    lane_polylines: torch.Tensor          # [num_lanes, max_points, 2] - (x, y)
    lane_features: torch.Tensor           # [num_lanes, D] - speed, type, etc.
    lane_masks: torch.Tensor              # [num_lanes] - valid lane mask
    connectivity_matrix: torch.Tensor     # [num_lanes, num_lanes] - adjacency
    
    # === INTERSECTION FEATURES (Level 3) ===
    lane_in_intersection: torch.Tensor    # [num_lanes] - binary flag
    intersection_ids: torch.Tensor        # [num_lanes] - which intersection (-1 if none)
    approach_vectors: torch.Tensor        # [num_lanes, 2] - direction entering intersection
    turn_intentions: torch.Tensor         # [num_lanes, 3] - [left, straight, right] probs
    
    # === TRAFFIC CONTROL FEATURES ===
    stop_line_positions: torch.Tensor     # [num_lanes, 2] - stop line position or (-1, -1)
    stop_line_distances: torch.Tensor     # [num_lanes] - distance to stop line
    has_stop_line: torch.Tensor           # [num_lanes] - binary flag
    
    # === CROSSWALK FEATURES (Pedestrian Safety) ===
    crosswalk_positions: torch.Tensor     # [num_crosswalks, 4, 2] - 4 corners of crosswalk
    crosswalk_mask: torch.Tensor          # [num_crosswalks] - valid mask
    lane_to_crosswalk: torch.Tensor       # [num_lanes, num_crosswalks] - proximity matrix
    
    # === OCCLUSION & VISIBILITY ===
    lane_visibility_scores: torch.Tensor  # [num_lanes] - how visible is this lane
    occluded_regions: torch.Tensor        # [num_lanes, 4] - bounding box of occluded areas
    
    # === GEOMETRIC CONTEXT ===
    lane_curvatures: torch.Tensor         # [num_lanes] - max curvature
    lane_lengths: torch.Tensor            # [num_lanes] - total length
    lane_headings: torch.Tensor           # [num_lanes] - average heading
    
    # Metadata
    num_lanes: int
    num_intersections: int
    num_crosswalks: int
    ego_position: Tuple[float, float]


class VectorMapExtractor:
    """
     vector map extractor with intersection, traffic, and occlusion reasoning.
    """
    
    def __init__(
        self,
        map_root: str,
        map_version: str = "nuplan-maps-v1.0",
        max_points_per_lane: int = 20,
        feature_dim: int = 16,  # Increased from 8
        max_crosswalks: int = 10,
    ):
        self.map_root = map_root
        self.map_version = map_version
        self.max_points_per_lane = max_points_per_lane
        self.feature_dim = feature_dim
        self.max_crosswalks = max_crosswalks
        
        # Map cache
        self._map_cache = {}
    
    def get_map_api(self, map_name: str):
        """Get map API with caching."""
        if map_name in self._map_cache:
            return self._map_cache[map_name]
        
        # Handle name conversion
        converted_name = map_name
        if map_name == "las_vegas":
            converted_name = "us-nv-las-vegas-strip"
        
        map_api = get_maps_api(self.map_root, self.map_version, converted_name)
        self._map_cache[map_name] = map_api
        return map_api
    
    def extract_objects_from_result(self, result) -> List:
        """Extract objects from map API result (handles defaultdict)."""
        if result is None:
            return []
        
        if isinstance(result, defaultdict) or isinstance(result, dict):
            all_objects = []
            for layer_objects in result.values():
                if isinstance(layer_objects, list):
                    all_objects.extend(layer_objects)
                else:
                    all_objects.append(layer_objects)
            return all_objects
        
        if isinstance(result, list):
            return result
        
        return [result] if result else []
    
    def extract(
        self,
        map_api,
        ego_pose: np.ndarray,  # [x, y, z, qw, qx, qy, qz]
        radius: float = 50.0,
    ) -> VectorMapFeatures:
        """
        Extract  vector map features around ego vehicle.
        """
        ego_x, ego_y = ego_pose[0], ego_pose[1]
        ego_point = Point2D(ego_x, ego_y)
        
        # Extract GeoDataFrames
        lanes_gdf = self._extract_lanes(map_api, ego_point, radius)
        lane_connectors_gdf = self._extract_lane_connectors(map_api, ego_point, radius)
        intersections_gdf = self._extract_intersections(map_api, ego_point, radius)
        stop_lines_gdf = self._extract_stop_lines(map_api, ego_point, radius)
        crosswalks_gdf = self._extract_crosswalks(map_api, ego_point, radius)
        
        # === CORE FEATURES ===
        lane_polylines, lane_features, lane_masks = self._gdf_to_tensors(lanes_gdf)
        connectivity = self._build_connectivity_matrix(lanes_gdf)
        
        # === INTERSECTION FEATURES ===
        (lane_in_intersection, intersection_ids, 
         approach_vectors, turn_intentions) = self._extract_intersection_features(
            lanes_gdf, intersections_gdf, ego_point
        )
        
        # === TRAFFIC CONTROL FEATURES ===
        (stop_line_positions, stop_line_distances, 
         has_stop_line) = self._extract_stop_line_features(
            lanes_gdf, stop_lines_gdf, ego_point
        )
        
        # === CROSSWALK FEATURES ===
        (crosswalk_positions, crosswalk_mask, 
         lane_to_crosswalk) = self._extract_crosswalk_features(
            lanes_gdf, crosswalks_gdf, ego_point
        )
        
        # === OCCLUSION & VISIBILITY ===
        (lane_visibility_scores, 
         occluded_regions) = self._compute_visibility_features(
            lanes_gdf, ego_point
        )
        
        # === GEOMETRIC FEATURES ===
        (lane_curvatures, lane_lengths, 
         lane_headings) = self._compute_geometric_features(lanes_gdf)
        
        return VectorMapFeatures(
            # GeoDataFrame format
            lanes_gdf=lanes_gdf,
            lane_connectors_gdf=lane_connectors_gdf,
            intersections_gdf=intersections_gdf,
            stop_lines_gdf=stop_lines_gdf,
            crosswalks_gdf=crosswalks_gdf,
            # Core features
            lane_polylines=lane_polylines,
            lane_features=lane_features,
            lane_masks=lane_masks,
            connectivity_matrix=connectivity,
            # Intersection features
            lane_in_intersection=lane_in_intersection,
            intersection_ids=intersection_ids,
            approach_vectors=approach_vectors,
            turn_intentions=turn_intentions,
            # Traffic control
            stop_line_positions=stop_line_positions,
            stop_line_distances=stop_line_distances,
            has_stop_line=has_stop_line,
            # Crosswalks
            crosswalk_positions=crosswalk_positions,
            crosswalk_mask=crosswalk_mask,
            lane_to_crosswalk=lane_to_crosswalk,
            # Occlusion
            lane_visibility_scores=lane_visibility_scores,
            occluded_regions=occluded_regions,
            # Geometric
            lane_curvatures=lane_curvatures,
            lane_lengths=lane_lengths,
            lane_headings=lane_headings,
            # Metadata
            num_lanes=len(lanes_gdf),
            num_intersections=len(intersections_gdf),
            num_crosswalks=len(crosswalks_gdf) if crosswalks_gdf is not None else 0,
            ego_position=(ego_x, ego_y),
        )
    
    # === BASIC EXTRACTION (same as before) ===
    
    def _extract_lanes(self, map_api, ego_point: Point2D, radius: float) -> gpd.GeoDataFrame:
        """Extract lane GeoDataFrame."""
        result = map_api.get_proximal_map_objects(
            ego_point, radius, [SemanticMapLayer.LANE]
        )
        lanes = self.extract_objects_from_result(result)
        
        if not lanes:
            return gpd.GeoDataFrame({
                'lane_id': [], 'speed_limit': [], 'incoming_edges': [],
                'outgoing_edges': [], 'geometry': [], 'baseline_path': [],
            })
        
        lane_data = []
        for lane in lanes:
            lane_data.append({
                'lane_id': lane.id,
                'speed_limit': lane.speed_limit_mps if lane.speed_limit_mps else 13.9,
                'incoming_edges': lane.incoming_edges if hasattr(lane, 'incoming_edges') else [],
                'outgoing_edges': lane.outgoing_edges if hasattr(lane, 'outgoing_edges') else [],
                'geometry': lane.polygon if hasattr(lane, 'polygon') else None,
                'baseline_path': lane.baseline_path if hasattr(lane, 'baseline_path') else None,
            })
        
        return gpd.GeoDataFrame(lane_data)
    
    def _extract_lane_connectors(self, map_api, ego_point: Point2D, radius: float) -> gpd.GeoDataFrame:
        """Extract lane connector GeoDataFrame."""
        result = map_api.get_proximal_map_objects(
            ego_point, radius, [SemanticMapLayer.LANE_CONNECTOR]
        )
        connectors = self.extract_objects_from_result(result)
        
        if not connectors:
            return gpd.GeoDataFrame({'connector_id': [], 'geometry': []})
        
        connector_data = []
        for conn in connectors:
            connector_data.append({
                'connector_id': conn.id,
                'geometry': conn.polygon if hasattr(conn, 'polygon') else None,
            })
        
        return gpd.GeoDataFrame(connector_data)
    
    def _extract_intersections(self, map_api, ego_point: Point2D, radius: float) -> gpd.GeoDataFrame:
        """Extract intersection GeoDataFrame."""
        result = map_api.get_proximal_map_objects(
            ego_point, radius, [SemanticMapLayer.INTERSECTION]
        )
        intersections = self.extract_objects_from_result(result)
        
        if not intersections:
            return gpd.GeoDataFrame({'intersection_id': [], 'geometry': []})
        
        intersection_data = []
        for inter in intersections:
            intersection_data.append({
                'intersection_id': inter.id,
                'geometry': inter.polygon if hasattr(inter, 'polygon') else None,
            })
        
        return gpd.GeoDataFrame(intersection_data)
    
    def _extract_stop_lines(self, map_api, ego_point: Point2D, radius: float) -> Optional[gpd.GeoDataFrame]:
        """Extract stop line GeoDataFrame."""
        try:
            result = map_api.get_proximal_map_objects(
                ego_point, radius, [SemanticMapLayer.STOP_LINE]
            )
            stop_lines = self.extract_objects_from_result(result)
            
            if not stop_lines:
                return None
            
            stop_data = []
            for stop in stop_lines:
                stop_data.append({
                    'stop_id': stop.id,
                    'geometry': stop.polygon if hasattr(stop, 'polygon') else None,
                })
            
            return gpd.GeoDataFrame(stop_data)
        except:
            return None
    
    def _extract_crosswalks(self, map_api, ego_point: Point2D, radius: float) -> Optional[gpd.GeoDataFrame]:
        """Extract crosswalk GeoDataFrame."""
        try:
            result = map_api.get_proximal_map_objects(
                ego_point, radius, [SemanticMapLayer.CROSSWALK]
            )
            crosswalks = self.extract_objects_from_result(result)
            
            if not crosswalks:
                return None
            
            crosswalk_data = []
            for cw in crosswalks:
                crosswalk_data.append({
                    'crosswalk_id': cw.id,
                    'geometry': cw.polygon if hasattr(cw, 'polygon') else None,
                })
            
            return gpd.GeoDataFrame(crosswalk_data)
        except:
            return None
    
    # === TENSOR CONVERSION ===
    
    def _gdf_to_tensors(
        self, 
        lanes_gdf: gpd.GeoDataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert lane GeoDataFrame to tensors."""
        if len(lanes_gdf) == 0:
            return (
                torch.zeros(1, self.max_points_per_lane, 2),
                torch.zeros(1, self.feature_dim),
                torch.zeros(1),
            )
        
        num_lanes = len(lanes_gdf)
        
        # Extract polylines
        polylines = torch.zeros(num_lanes, self.max_points_per_lane, 2)
        masks = torch.zeros(num_lanes)
        
        for i, row in lanes_gdf.iterrows():
            if row['baseline_path'] is not None:
                path = row['baseline_path'].discrete_path
                
                # Handle StateSE2 objects
                if len(path) > 0 and hasattr(path[0], 'x'):
                    points = np.array([[p.x, p.y] for p in path])
                else:
                    points = np.array(path)[:, :2]
                
                # Resample to max_points_per_lane
                num_points = len(points)
                if num_points > self.max_points_per_lane:
                    indices = np.linspace(0, num_points-1, self.max_points_per_lane, dtype=int)
                    points = points[indices]
                elif num_points < self.max_points_per_lane:
                    padding = np.repeat(points[-1:], self.max_points_per_lane - num_points, axis=0)
                    points = np.vstack([points, padding])
                
                polylines[i] = torch.from_numpy(points).float()
                masks[i] = 1.0
        
        # Extract  lane features
        features = torch.zeros(num_lanes, self.feature_dim)
        
        for i, row in lanes_gdf.iterrows():
            features[i, 0] = row['speed_limit'] / 30.0  # Normalize
            features[i, 1] = len(row['incoming_edges']) / 5.0
            features[i, 2] = len(row['outgoing_edges']) / 5.0
            # Rest will be filled by other feature extractors
        
        return polylines, features, masks
    
    def _build_connectivity_matrix(self, lanes_gdf: gpd.GeoDataFrame) -> torch.Tensor:
        """Build lane connectivity matrix."""
        if len(lanes_gdf) == 0:
            return torch.zeros(1, 1)
        
        num_lanes = len(lanes_gdf)
        connectivity = torch.zeros(num_lanes, num_lanes)
        
        lane_id_to_idx = {row['lane_id']: i for i, row in lanes_gdf.iterrows()}
        
        for i, row in lanes_gdf.iterrows():
            for successor_id in row['outgoing_edges']:
                if successor_id in lane_id_to_idx:
                    j = lane_id_to_idx[successor_id]
                    connectivity[i, j] = 1.0
        
        return connectivity
    
    # === NEW: INTERSECTION FEATURES ===
    
    def _extract_intersection_features(
        self,
        lanes_gdf: gpd.GeoDataFrame,
        intersections_gdf: gpd.GeoDataFrame,
        ego_point: Point2D,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract intersection-related features.
        
        Returns:
            lane_in_intersection: [num_lanes] - binary flag
            intersection_ids: [num_lanes] - which intersection
            approach_vectors: [num_lanes, 2] - direction vector
            turn_intentions: [num_lanes, 3] - [left, straight, right]
        """
        num_lanes = len(lanes_gdf) if len(lanes_gdf) > 0 else 1
        
        lane_in_intersection = torch.zeros(num_lanes)
        intersection_ids = torch.full((num_lanes,), -1, dtype=torch.long)
        approach_vectors = torch.zeros(num_lanes, 2)
        turn_intentions = torch.zeros(num_lanes, 3)  # [left, straight, right]
        
        if len(lanes_gdf) == 0 or len(intersections_gdf) == 0:
            return lane_in_intersection, intersection_ids, approach_vectors, turn_intentions
        
        # Map lanes to intersections via spatial overlap
        for i, lane_row in lanes_gdf.iterrows():
            if lane_row['geometry'] is None:
                continue
            
            for j, inter_row in intersections_gdf.iterrows():
                if inter_row['geometry'] is None:
                    continue
                
                if lane_row['geometry'].intersects(inter_row['geometry']):
                    lane_in_intersection[i] = 1.0
                    intersection_ids[i] = j
                    
                    # Compute approach vector (direction entering intersection)
                    if lane_row['baseline_path'] is not None:
                        path = lane_row['baseline_path'].discrete_path
                        if len(path) >= 2:
                            if hasattr(path[0], 'x'):
                                start = np.array([path[0].x, path[0].y])
                                end = np.array([path[-1].x, path[-1].y])
                            else:
                                start = np.array(path[0][:2])
                                end = np.array(path[-1][:2])
                            
                            direction = end - start
                            direction = direction / (np.linalg.norm(direction) + 1e-6)
                            approach_vectors[i] = torch.from_numpy(direction).float()
                    
                    # Estimate turn intention based on successors
                    num_successors = len(lane_row['outgoing_edges'])
                    if num_successors > 0:
                        # Simple heuristic: equal probability
                        turn_intentions[i] = torch.tensor([0.33, 0.34, 0.33])
                    else:
                        turn_intentions[i] = torch.tensor([0.0, 1.0, 0.0])  # Default: straight
                    
                    break
        
        return lane_in_intersection, intersection_ids, approach_vectors, turn_intentions
    
    # === NEW: STOP LINE FEATURES ===
    
    def _extract_stop_line_features(
        self,
        lanes_gdf: gpd.GeoDataFrame,
        stop_lines_gdf: Optional[gpd.GeoDataFrame],
        ego_point: Point2D,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract stop line features for each lane.
        
        Returns:
            stop_line_positions: [num_lanes, 2] - position or (-1, -1)
            stop_line_distances: [num_lanes] - distance from ego
            has_stop_line: [num_lanes] - binary flag
        """
        num_lanes = len(lanes_gdf) if len(lanes_gdf) > 0 else 1
        
        stop_line_positions = torch.full((num_lanes, 2), -1.0)
        stop_line_distances = torch.full((num_lanes,), float('inf'))
        has_stop_line = torch.zeros(num_lanes)
        
        if len(lanes_gdf) == 0 or stop_lines_gdf is None or len(stop_lines_gdf) == 0:
            return stop_line_positions, stop_line_distances, has_stop_line
        
        ego_pos = np.array([ego_point.x, ego_point.y])
        
        # Map stop lines to lanes
        for i, lane_row in lanes_gdf.iterrows():
            if lane_row['geometry'] is None:
                continue
            
            closest_dist = float('inf')
            closest_pos = None
            
            for _, stop_row in stop_lines_gdf.iterrows():
                if stop_row['geometry'] is None:
                    continue
                
                # Check if stop line intersects or is near lane
                if lane_row['geometry'].intersects(stop_row['geometry']) or \
                   lane_row['geometry'].distance(stop_row['geometry']) < 5.0:
                    
                    # Get stop line centroid
                    centroid = stop_row['geometry'].centroid
                    stop_pos = np.array([centroid.x, centroid.y])
                    dist = np.linalg.norm(stop_pos - ego_pos)
                    
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_pos = stop_pos
            
            if closest_pos is not None:
                stop_line_positions[i] = torch.from_numpy(closest_pos).float()
                stop_line_distances[i] = closest_dist
                has_stop_line[i] = 1.0
        
        return stop_line_positions, stop_line_distances, has_stop_line
    
    # === NEW: CROSSWALK FEATURES ===
    
    def _extract_crosswalk_features(
        self,
        lanes_gdf: gpd.GeoDataFrame,
        crosswalks_gdf: Optional[gpd.GeoDataFrame],
        ego_point: Point2D,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract crosswalk features.
        
        Returns:
            crosswalk_positions: [num_crosswalks, 4, 2] - 4 corners
            crosswalk_mask: [num_crosswalks] - valid mask
            lane_to_crosswalk: [num_lanes, num_crosswalks] - proximity
        """
        num_lanes = len(lanes_gdf) if len(lanes_gdf) > 0 else 1
        
        if crosswalks_gdf is None or len(crosswalks_gdf) == 0:
            return (
                torch.zeros(1, 4, 2),
                torch.zeros(1),
                torch.zeros(num_lanes, 1),
            )
        
        num_crosswalks = min(len(crosswalks_gdf), self.max_crosswalks)
        
        crosswalk_positions = torch.zeros(num_crosswalks, 4, 2)
        crosswalk_mask = torch.zeros(num_crosswalks)
        lane_to_crosswalk = torch.zeros(num_lanes, num_crosswalks)
        
        for j, cw_row in crosswalks_gdf.iterrows():
            if j >= num_crosswalks:
                break
            
            if cw_row['geometry'] is not None:
                # Get crosswalk bounds
                bounds = cw_row['geometry'].bounds  # (minx, miny, maxx, maxy)
                
                # Extract 4 corners
                corners = np.array([
                    [bounds[0], bounds[1]],  # bottom-left
                    [bounds[2], bounds[1]],  # bottom-right
                    [bounds[2], bounds[3]],  # top-right
                    [bounds[0], bounds[3]],  # top-left
                ])
                
                crosswalk_positions[j] = torch.from_numpy(corners).float()
                crosswalk_mask[j] = 1.0
                
                # Compute proximity to lanes
                for i, lane_row in lanes_gdf.iterrows():
                    if lane_row['geometry'] is not None:
                        dist = lane_row['geometry'].distance(cw_row['geometry'])
                        # Proximity score: 1 if < 10m, decays to 0 at 30m
                        proximity = max(0.0, 1.0 - dist / 30.0)
                        lane_to_crosswalk[i, j] = proximity
        
        return crosswalk_positions, crosswalk_mask, lane_to_crosswalk
    
    # === NEW: OCCLUSION & VISIBILITY ===
    
    def _compute_visibility_features(
        self,
        lanes_gdf: gpd.GeoDataFrame,
        ego_point: Point2D,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visibility scores for lanes.
        
        Returns:
            lane_visibility_scores: [num_lanes] - 0 (occluded) to 1 (visible)
            occluded_regions: [num_lanes, 4] - bounding box of occluded area
        """
        num_lanes = len(lanes_gdf) if len(lanes_gdf) > 0 else 1
        
        lane_visibility_scores = torch.ones(num_lanes)  # Default: fully visible
        occluded_regions = torch.zeros(num_lanes, 4)    # [x1, y1, x2, y2]
        
        if len(lanes_gdf) == 0:
            return lane_visibility_scores, occluded_regions
        
        ego_pos = np.array([ego_point.x, ego_point.y])
        
        for i, lane_row in lanes_gdf.iterrows():
            if lane_row['geometry'] is None:
                lane_visibility_scores[i] = 0.0
                continue
            
            # Simple heuristic: visibility decreases with distance
            centroid = lane_row['geometry'].centroid
            lane_pos = np.array([centroid.x, centroid.y])
            dist = np.linalg.norm(lane_pos - ego_pos)
            
            # Visibility: 1.0 at 0m, 0.5 at 50m, 0.0 at 100m
            visibility = max(0.0, 1.0 - dist / 100.0)
            lane_visibility_scores[i] = visibility
            
            # Occluded region (simplified: use lane bounds)
            bounds = lane_row['geometry'].bounds
            occluded_regions[i] = torch.tensor([bounds[0], bounds[1], bounds[2], bounds[3]])
        
        return lane_visibility_scores, occluded_regions
    
    # === NEW: GEOMETRIC FEATURES ===
    
    def _compute_geometric_features(
        self,
        lanes_gdf: gpd.GeoDataFrame,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute geometric features for lanes.
        
        Returns:
            lane_curvatures: [num_lanes] - max curvature
            lane_lengths: [num_lanes] - total length
            lane_headings: [num_lanes] - average heading (radians)
        """
        num_lanes = len(lanes_gdf) if len(lanes_gdf) > 0 else 1
        
        lane_curvatures = torch.zeros(num_lanes)
        lane_lengths = torch.zeros(num_lanes)
        lane_headings = torch.zeros(num_lanes)
        
        if len(lanes_gdf) == 0:
            return lane_curvatures, lane_lengths, lane_headings
        
        for i, lane_row in lanes_gdf.iterrows():
            if lane_row['baseline_path'] is None:
                continue
            
            path = lane_row['baseline_path'].discrete_path
            
            if len(path) < 2:
                continue
            
            # Extract points
            if hasattr(path[0], 'x'):
                points = np.array([[p.x, p.y] for p in path])
            else:
                points = np.array(path)[:, :2]
            
            # Compute length
            segments = np.diff(points, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            total_length = np.sum(lengths)
            lane_lengths[i] = total_length
            
            # Compute heading (average direction)
            if len(segments) > 0:
                avg_direction = np.mean(segments, axis=0)
                heading = np.arctan2(avg_direction[1], avg_direction[0])
                lane_headings[i] = heading
            
            # Compute curvature (simple approximation)
            if len(points) >= 3:
                # Curvature = max change in heading
                headings = np.arctan2(segments[:, 1], segments[:, 0])
                curvature = np.max(np.abs(np.diff(headings)))
                lane_curvatures[i] = curvature
        
        return lane_curvatures, lane_lengths, lane_headings


 
#  Batch Extraction
 

def extract__vector_map_for_batch(
    extractor: VectorMapExtractor,
    batch_scenes: List,
    batch_frames: List,
    map_apis: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Extract  vector map features for a batch.
    
    Returns dict with ALL features ready for neural network.
    """
    # Collect features from all samples
    all_features = []
    
    for scene, frame_idx in zip(batch_scenes, batch_frames):
        frame = scene.frames[frame_idx]
        ego_pose = frame.ego_status.ego_pose
        map_name = scene.scene_metadata.map_name
        
        # Get map API
        if map_name not in map_apis:
            map_apis[map_name] = extractor.get_map_api(map_name)
        map_api = map_apis[map_name]
        
        # Extract features
        features = extractor.extract(map_api, ego_pose, radius=50.0)
        all_features.append(features)
    
    # Find max dimensions for padding
    max_lanes = max(f.num_lanes for f in all_features)
    max_crosswalks = max(f.num_crosswalks for f in all_features)
    
    # Pad and stack all features
    batch_dict = {}
    
    # Core features
    batch_dict['lane_polylines'] = pad_and_stack([f.lane_polylines for f in all_features], max_lanes)
    batch_dict['lane_features'] = pad_and_stack([f.lane_features for f in all_features], max_lanes)
    batch_dict['lane_masks'] = pad_and_stack([f.lane_masks for f in all_features], max_lanes)
    batch_dict['connectivity'] = pad_and_stack_2d([f.connectivity_matrix for f in all_features], max_lanes)
    
    # Intersection features
    batch_dict['lane_in_intersection'] = pad_and_stack([f.lane_in_intersection for f in all_features], max_lanes)
    batch_dict['intersection_ids'] = pad_and_stack([f.intersection_ids for f in all_features], max_lanes)
    batch_dict['approach_vectors'] = pad_and_stack([f.approach_vectors for f in all_features], max_lanes)
    batch_dict['turn_intentions'] = pad_and_stack([f.turn_intentions for f in all_features], max_lanes)
    
    # Traffic control
    batch_dict['stop_line_positions'] = pad_and_stack([f.stop_line_positions for f in all_features], max_lanes)
    batch_dict['stop_line_distances'] = pad_and_stack([f.stop_line_distances for f in all_features], max_lanes)
    batch_dict['has_stop_line'] = pad_and_stack([f.has_stop_line for f in all_features], max_lanes)
    
    # Crosswalks
    batch_dict['crosswalk_positions'] = pad_and_stack([f.crosswalk_positions for f in all_features], max_crosswalks)
    batch_dict['crosswalk_mask'] = pad_and_stack([f.crosswalk_mask for f in all_features], max_crosswalks)
    batch_dict['lane_to_crosswalk'] = pad_and_stack_2d_mixed(
        [f.lane_to_crosswalk for f in all_features], max_lanes, max_crosswalks
    )
    
    # Visibility
    batch_dict['lane_visibility_scores'] = pad_and_stack([f.lane_visibility_scores for f in all_features], max_lanes)
    batch_dict['occluded_regions'] = pad_and_stack([f.occluded_regions for f in all_features], max_lanes)
    
    # Geometric
    batch_dict['lane_curvatures'] = pad_and_stack([f.lane_curvatures for f in all_features], max_lanes)
    batch_dict['lane_lengths'] = pad_and_stack([f.lane_lengths for f in all_features], max_lanes)
    batch_dict['lane_headings'] = pad_and_stack([f.lane_headings for f in all_features], max_lanes)
    
    return batch_dict


# Helper padding functions
def pad_and_stack(tensors: List[torch.Tensor], max_size: int) -> torch.Tensor:
    """Pad tensors to max_size and stack."""
    padded = []
    for t in tensors:
        if t.dim() == 1:
            # [N] -> [max_size]
            if t.shape[0] < max_size:
                pad = torch.zeros(max_size - t.shape[0], dtype=t.dtype)
                t = torch.cat([t, pad])
        elif t.dim() == 2:
            # [N, D] -> [max_size, D]
            if t.shape[0] < max_size:
                pad = torch.zeros(max_size - t.shape[0], t.shape[1], dtype=t.dtype)
                t = torch.cat([t, pad])
        elif t.dim() == 3:
            # [N, P, D] -> [max_size, P, D]
            if t.shape[0] < max_size:
                pad = torch.zeros(max_size - t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype)
                t = torch.cat([t, pad])
        padded.append(t)
    return torch.stack(padded)

def pad_and_stack_2d(tensors: List[torch.Tensor], max_size: int) -> torch.Tensor:
    """Pad 2D connectivity matrices."""
    padded = []
    for t in tensors:
        if t.shape[0] < max_size:
            # Pad rows
            row_pad = torch.zeros(max_size - t.shape[0], t.shape[1], dtype=t.dtype)
            t = torch.cat([t, row_pad], dim=0)
        if t.shape[1] < max_size:
            # Pad columns
            col_pad = torch.zeros(t.shape[0], max_size - t.shape[1], dtype=t.dtype)
            t = torch.cat([t, col_pad], dim=1)
        padded.append(t)
    return torch.stack(padded)

def pad_and_stack_2d_mixed(tensors: List[torch.Tensor], max_lanes: int, max_crosswalks: int) -> torch.Tensor:
    """Pad lane-to-crosswalk matrices."""
    padded = []
    for t in tensors:
        if t.shape[0] < max_lanes:
            row_pad = torch.zeros(max_lanes - t.shape[0], t.shape[1], dtype=t.dtype)
            t = torch.cat([t, row_pad], dim=0)
        if t.shape[1] < max_crosswalks:
            col_pad = torch.zeros(t.shape[0], max_crosswalks - t.shape[1], dtype=t.dtype)
            t = torch.cat([t, col_pad], dim=1)
        padded.append(t)
    return torch.stack(padded)


# just for checking the map
if __name__ == "__main__":
    import os
    
    print("=" * 70)
    print(" Vector Map Extraction Test")
    print("=" * 70)
    
    map_root = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
    os.environ['NUPLAN_MAPS_ROOT'] = map_root
    
    # Initialize  extractor
    map_extractor = VectorMapExtractor(
        map_root=map_root,
        max_points_per_lane=20,
        feature_dim=16,  # Increased!
        max_crosswalks=10,
    )
    print("   VectorMapExtractor initialized")
    
    # Load dataset
    from navsim_utilize.navsimdataset import NavsimDataset
    
    dataset = NavsimDataset(
        bev_size=(200, 200),
        bev_range=50.0,
        use_uniad_bev=False,
        extract_labels=True,
        use_cache=False,
        map_root=map_root
    )
    print(f"  Dataset loaded: {len(dataset)} samples")
    
    # Get a scene
    scene = dataset.scene_loader.get_scene_from_token(dataset.scene_tokens[0])
    frame_idx = dataset.history_length - 1
    
    print(f"\n{'='*70}")
    print("Testing  Vector Map Extraction")
    print(f"{'='*70}")
    
    map_apis = {}
    vector_maps = extract__vector_map_for_batch(
        extractor=map_extractor,
        batch_scenes=[scene],
        batch_frames=[frame_idx],
        map_apis=map_apis,
    )
    
    print(f"\n   vector map extracted!")
    print(f"\n{'='*70}")
    print("ALL FEATURES:")
    print(f"{'='*70}")
    
    for key, tensor in vector_maps.items():
        print(f"  {key:30s}: {tensor.shape}")
    
    # Show sample data
    print(f"\n{'='*70}")
    print("SAMPLE DATA:")
    print(f"{'='*70}")
    
    b = 0  # Batch index
    
    print(f"\nIntersection Features:")
    print(f"  Lanes in intersection: {vector_maps['lane_in_intersection'][b].sum().item():.0f}")
    print(f"  Sample approach vector: {vector_maps['approach_vectors'][b][0]}")
    print(f"  Sample turn intention: {vector_maps['turn_intentions'][b][0]}")
    
    print(f"\nTraffic Control:")
    print(f"  Lanes with stop lines: {vector_maps['has_stop_line'][b].sum().item():.0f}")
    if vector_maps['has_stop_line'][b].sum() > 0:
        idx = torch.where(vector_maps['has_stop_line'][b] > 0)[0][0]
        print(f"  Sample stop line position: {vector_maps['stop_line_positions'][b][idx]}")
        print(f"  Distance: {vector_maps['stop_line_distances'][b][idx]:.2f}m")
    
    print(f"\nCrosswalks:")
    print(f"  Number of crosswalks: {vector_maps['crosswalk_mask'][b].sum().item():.0f}")
    if vector_maps['crosswalk_mask'][b].sum() > 0:
        print(f"  First crosswalk corners:\n{vector_maps['crosswalk_positions'][b][0]}")
    
    print(f"\nVisibility:")
    print(f"  Average lane visibility: {vector_maps['lane_visibility_scores'][b].mean().item():.3f}")
    print(f"  Min visibility: {vector_maps['lane_visibility_scores'][b].min().item():.3f}")
    
    print(f"\nGeometric:")
    print(f"  Average lane length: {vector_maps['lane_lengths'][b].mean().item():.2f}m")
    print(f"  Max curvature: {vector_maps['lane_curvatures'][b].max().item():.3f} rad")
    
    print(f"\n{'='*70}")
    print("   Vector Map Extraction Test PASSED!")
    print(f"{'='*70}")