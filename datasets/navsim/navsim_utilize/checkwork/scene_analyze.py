#!/usr/bin/env python3
"""
Run SceneAnalyzer on NAVSIM Mini Dataset
Analyzes all scenes and saves metadata to JSON
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

# # Set environment variables (matching your setup)
# os.environ['OPENSCENE_DATA_ROOT'] = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download"
# os.environ['NUPLAN_MAPS_ROOT'] = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"
# os.environ['NAVSIM_DEVKIT_ROOT'] = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim"

# Import NAVSIM
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig


class SceneAnalyzer:
    """
    Analyzes NAVSIM scenes to extract complexity metrics,
    interaction patterns, and modality requirements.
    """
    
    def __init__(self, history_length=4, future_length=8):
        self.history_length = history_length
        self.future_length = future_length
        
    def analyze_scene(self, scene: Scene) -> Dict:
        """Extract comprehensive scene metadata"""
        
        metadata = {
            'scene_id': scene.scene_metadata.scene_token,
            'log_name': scene.scene_metadata.log_name,
            'map_name': scene.scene_metadata.map_name,
            'initial_token': scene.scene_metadata.initial_token,
            'complexity': self._compute_complexity(scene),
            'environment': self._classify_environment(scene),
            'modality_requirements': self._determine_modality_needs(scene),
            'interaction_patterns': self._extract_interactions(scene),
            'pedestrian_details': self._analyze_pedestrians(scene),
            'failure_modes': self._predict_failure_modes(scene),
            'encoding_level_requirements': self._determine_encoding_needs(scene),
            'safety_critical': self._is_safety_critical(scene),
            'difficulty_rating': self._rate_difficulty(scene),
            'tags': self._generate_tags(scene)
        }
        
        return metadata
    
    def _get_agents_at_frame(self, scene: Scene, frame_idx: int) -> List:
        """Extract agent information from annotations at specific frame"""
        frame = scene.frames[frame_idx]
        
        if frame.annotations is None:
            return []
        
        agents = []
        for i in range(len(frame.annotations.boxes)):
            agent = {
                'box': frame.annotations.boxes[i],
                'center': frame.annotations.boxes[i][:3],  # x, y, z
                'agent_type': frame.annotations.names[i],
                'velocity': frame.annotations.velocity_3d[i] if frame.annotations.velocity_3d is not None else np.zeros(3),
                'instance_token': frame.annotations.instance_tokens[i],
                'track_token': frame.annotations.track_tokens[i]
            }
            agents.append(agent)
        
        return agents
    
    def _compute_complexity(self, scene: Scene) -> Dict:
        """Compute multi-dimensional complexity score"""
        
        # Get current frame (last history frame)
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        agents = self._get_agents_at_frame(scene, current_frame_idx)
        
        # Agent density
        agent_count = len(agents)
        pedestrian_count = sum(1 for a in agents if 'pedestrian' in a['agent_type'].lower())
        vehicle_count = sum(1 for a in agents if 'vehicle' in a['agent_type'].lower() or 'car' in a['agent_type'].lower())
        
        # Get ego position
        ego = scene.frames[current_frame_idx].ego_status
        ego_pos = ego.ego_pose[:2]
        
        # Interaction density (agents within 20m of ego)
        close_agents = []
        for agent in agents:
            agent_pos = agent['center'][:2]
            dist = np.linalg.norm(agent_pos - ego_pos)
            if dist < 20.0:
                close_agents.append(agent)
        
        interaction_count = len(close_agents)
        
        # Occlusion level (based on agent positions)
        occlusion_level = self._estimate_occlusion(scene, agents, ego_pos)
        
        # Overall complexity score
        complexity_score = (
            0.3 * min(agent_count / 20.0, 1.0) +
            0.3 * min(interaction_count / 10.0, 1.0) +
            0.2 * min(pedestrian_count / 5.0, 1.0) +
            0.2 * occlusion_level
        )
        
        return {
            'overall_score': float(complexity_score),
            'agent_density': int(agent_count),
            'vehicle_count': int(vehicle_count),
            'pedestrian_count': int(pedestrian_count),
            'interaction_count': int(interaction_count),
            'occlusion_level': float(occlusion_level)
        }
    
    def _estimate_occlusion(self, scene: Scene, agents: List, ego_pos: np.ndarray) -> float:
        """Estimate occlusion level in scene"""
        
        if len(agents) < 2:
            return 0.0
        
        occlusion_count = 0
        total_pairs = 0
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i >= j:
                    continue
                    
                total_pairs += 1
                
                pos1 = agent1['center'][:2]
                pos2 = agent2['center'][:2]
                
                dist1 = np.linalg.norm(pos1 - ego_pos)
                dist2 = np.linalg.norm(pos2 - ego_pos)
                
                if abs(dist1 - dist2) > 2.0:
                    angle1 = np.arctan2(pos1[1] - ego_pos[1], pos1[0] - ego_pos[0])
                    angle2 = np.arctan2(pos2[1] - ego_pos[1], pos2[0] - ego_pos[0])
                    angle_diff = abs(self._angle_difference(angle1, angle2))
                    
                    if angle_diff < 0.2:
                        occlusion_count += 1
        
        if total_pairs == 0:
            return 0.0
            
        return min(occlusion_count / max(total_pairs * 0.3, 1), 1.0)
    
    def _angle_difference(self, angle1, angle2):
        """Compute smallest difference between two angles"""
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def _classify_environment(self, scene: Scene) -> Dict:
        """Classify the driving environment"""
        
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[current_frame_idx]
        ego_pos = frame.ego_status.ego_pose[:2]
        
        # Check for traffic lights
        has_traffic_light = len(frame.traffic_lights) > 0 if frame.traffic_lights else False
        
        # Check roadblock IDs to infer intersection
        has_intersection = False
        if frame.roadblock_ids:
            has_intersection = len(frame.roadblock_ids) > 2
        
        # Use map API if available
        has_crosswalk = False
        road_type = "unknown"
        
        if scene.map_api is not None:
            try:
                has_crosswalk = self._check_crosswalk(scene.map_api, ego_pos)
                road_type = self._classify_road_type(scene.map_api, ego_pos)
                
                if not has_intersection:
                    has_intersection = self._check_intersection(scene.map_api, ego_pos)
            except Exception as e:
                print(f"Map API error: {e}")
        
        return {
            'has_intersection': bool(has_intersection),
            'has_crosswalk': bool(has_crosswalk),
            'has_traffic_light': bool(has_traffic_light),
            'road_type': str(road_type),
            'map_name': str(scene.scene_metadata.map_name),
            'complexity_factors': {
                'intersection': 0.8 if has_intersection else 0.0,
                'crosswalk': 0.7 if has_crosswalk else 0.0,
                'traffic_control': 0.5 if has_traffic_light else 0.0
            }
        }
    
    def _check_intersection(self, map_api, ego_pos, radius=30.0) -> bool:
        """Check if ego is near an intersection"""
        try:
            if hasattr(map_api, 'get_proximal_map_objects'):
                nearby_objects = map_api.get_proximal_map_objects(ego_pos, radius, ['lane', 'lane_connector'])
                return len(nearby_objects.get('lane_connector', [])) > 0
        except:
            pass
        return False
    
    def _check_crosswalk(self, map_api, ego_pos, radius=25.0) -> bool:
        """Check if ego is near a crosswalk"""
        try:
            if hasattr(map_api, 'get_proximal_map_objects'):
                nearby_objects = map_api.get_proximal_map_objects(ego_pos, radius, ['crosswalk'])
                return len(nearby_objects.get('crosswalk', [])) > 0
        except:
            pass
        return False
    
    def _classify_road_type(self, map_api, ego_pos) -> str:
        """Classify the type of road"""
        try:
            if hasattr(map_api, 'get_proximal_map_objects'):
                nearby_objects = map_api.get_proximal_map_objects(ego_pos, 10.0, ['lane'])
                lanes = nearby_objects.get('lane', [])
                
                if len(lanes) > 0:
                    lane = lanes[0]
                    if hasattr(lane, 'speed_limit_mps'):
                        speed_limit = lane.speed_limit_mps
                        if speed_limit > 25:
                            return "highway"
                        elif speed_limit > 15:
                            return "arterial"
                        else:
                            return "residential"
            return "urban"
        except:
            return "unknown"
    
    def _determine_modality_needs(self, scene: Scene) -> Dict:
        """Determine which modalities are critical for this scene"""
        
        complexity = self._compute_complexity(scene)
        environment = self._classify_environment(scene)
        
        # Map modality importance
        map_importance = 0.5
        if environment['has_intersection']:
            map_importance += 0.3
        if environment['has_crosswalk']:
            map_importance += 0.2
        map_importance = min(map_importance, 1.0)
        
        # Agent modality importance
        agent_importance = min(complexity['interaction_count'] / 10.0, 1.0)
        
        # Pedestrian modality importance
        pedestrian_importance = 0.0
        if complexity['pedestrian_count'] > 0:
            pedestrian_importance = 0.5 + 0.5 * min(complexity['pedestrian_count'] / 5.0, 1.0)
        
        # Temporal modality
        temporal_importance = 0.8
        
        # Camera importance
        camera_importance = 0.7
        if complexity['pedestrian_count'] > 0 or complexity['occlusion_level'] > 0.5:
            camera_importance = 0.9
        
        return {
            'map': float(map_importance),
            'agents': float(agent_importance),
            'pedestrians': float(pedestrian_importance),
            'temporal': float(temporal_importance),
            'camera': float(camera_importance),
            'critical_modalities': self._identify_critical_modalities(
                map_importance, agent_importance, pedestrian_importance, camera_importance
            )
        }
    
    def _identify_critical_modalities(self, map_imp, agent_imp, ped_imp, camera_imp) -> List[str]:
        """Identify which modalities are critical (>0.6 importance)"""
        
        critical = []
        if map_imp > 0.6:
            critical.append('map')
        if agent_imp > 0.6:
            critical.append('agents')
        if ped_imp > 0.6:
            critical.append('pedestrians')
        if camera_imp > 0.6:
            critical.append('camera')
        critical.append('temporal')
        
        return critical
    
    def _extract_interactions(self, scene: Scene) -> List[Dict]:
        """Extract interaction patterns between agents"""
        
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        agents = self._get_agents_at_frame(scene, current_frame_idx)
        ego = scene.frames[current_frame_idx].ego_status
        ego_pos = ego.ego_pose[:2]
        ego_vel = ego.ego_velocity[:2]
        
        interactions = []
        
        for i, agent in enumerate(agents):
            agent_pos = agent['center'][:2]
            dist = np.linalg.norm(agent_pos - ego_pos)
            
            if dist < 15.0:
                interaction_type = self._classify_interaction(ego, agent, ego_pos, ego_vel)
                
                if interaction_type:
                    interactions.append({
                        'type': str(interaction_type),
                        'agents': [0, int(i + 1)],
                        'distance': float(dist),
                        'agent_type': str(agent['agent_type']),
                        'difficulty': str(self._rate_interaction_difficulty(interaction_type, dist, agent)),
                        'level_required': str(self._get_required_encoding_level(interaction_type))
                    })
        
        return interactions
    
    def _classify_interaction(self, ego, agent, ego_pos, ego_vel) -> Optional[str]:
        """Classify the type of interaction"""
        
        agent_pos = agent['center'][:2]
        agent_vel = agent['velocity'][:2]
        
        relative_vel = agent_vel - ego_vel
        relative_pos = agent_pos - ego_pos
        
        rel_vel_norm = np.linalg.norm(relative_vel)
        if rel_vel_norm > 0.5:
            ttc = -np.dot(relative_pos, relative_vel) / (rel_vel_norm ** 2)
        else:
            ttc = float('inf')
        
        agent_type = agent['agent_type'].lower()
        
        if 'pedestrian' in agent_type:
            if ttc < 5.0 and ttc > 0:
                return 'pedestrian_crossing'
            else:
                return 'pedestrian_nearby'
        
        elif 'vehicle' in agent_type or 'car' in agent_type:
            angle = np.arctan2(relative_pos[1], relative_pos[0])
            
            if abs(angle) < np.pi / 4:
                if np.linalg.norm(relative_vel) < -1.0:
                    return 'vehicle_yielding'
                else:
                    return 'vehicle_following'
            elif np.pi / 4 < abs(angle) < 3 * np.pi / 4:
                return 'vehicle_merging'
        
        return 'general_interaction'
    
    def _rate_interaction_difficulty(self, interaction_type: str, distance: float, agent) -> str:
        """Rate the difficulty of an interaction"""
        
        distance_factor = 1.0 - min(distance / 15.0, 1.0)
        agent_type = agent['agent_type'].lower()
        type_factor = 0.8 if 'pedestrian' in agent_type else 0.5
        difficulty_score = 0.5 * distance_factor + 0.5 * type_factor
        
        if difficulty_score > 0.7:
            return 'hard'
        elif difficulty_score > 0.4:
            return 'medium'
        else:
            return 'easy'
    
    def _get_required_encoding_level(self, interaction_type: str) -> str:
        """Determine which encoding level is needed"""
        
        level_mapping = {
            'pedestrian_crossing': 'scene',
            'pedestrian_nearby': 'interaction',
            'vehicle_yielding': 'interaction',
            'vehicle_merging': 'scene',
            'vehicle_following': 'temporal',
            'general_interaction': 'interaction'
        }
        
        return level_mapping.get(interaction_type, 'interaction')
    
    def _analyze_pedestrians(self, scene: Scene) -> List[Dict]:
        """Detailed analysis of pedestrian behavior"""
        
        current_frame_idx = scene.scene_metadata.num_history_frames - 1
        agents = self._get_agents_at_frame(scene, current_frame_idx)
        ego = scene.frames[current_frame_idx].ego_status
        ego_pos = ego.ego_pose[:2]
        
        pedestrian_details = []
        
        for i, agent in enumerate(agents):
            agent_type = agent['agent_type'].lower()
            if 'pedestrian' not in agent_type:
                continue
            
            ped_pos = agent['center'][:2]
            ped_vel = agent['velocity'][:2]
            
            crossing_intention = self._estimate_crossing_intention(agent, ped_pos, ped_vel, ego_pos)
            
            dist_to_crosswalk = None
            if scene.map_api is not None:
                try:
                    dist_to_crosswalk = float(self._distance_to_nearest_crosswalk(scene.map_api, ped_pos))
                except:
                    pass
            
            occlusion_status = self._estimate_agent_occlusion(agent, agents, ego_pos)
            behavior = self._classify_pedestrian_behavior(ped_vel, crossing_intention, dist_to_crosswalk or float('inf'))
            risk_level = self._assess_pedestrian_risk(ped_pos, ego_pos, crossing_intention, occlusion_status)
            
            pedestrian_details.append({
                'agent_id': int(i + 1),
                'crossing_intention': float(crossing_intention),
                'distance_to_crosswalk': dist_to_crosswalk,
                'occlusion_status': str(occlusion_status),
                'behavior_label': str(behavior),
                'risk_level': float(risk_level),
                'distance_to_ego': float(np.linalg.norm(ped_pos - ego_pos)),
                'velocity_magnitude': float(np.linalg.norm(ped_vel))
            })
        
        return pedestrian_details
    
    def _estimate_crossing_intention(self, pedestrian, ped_pos, ped_vel, ego_pos) -> float:
        """Estimate pedestrian's intention to cross"""
        
        vel_magnitude = np.linalg.norm(ped_vel)
        
        if vel_magnitude > 0.3:
            to_ego = ego_pos - ped_pos
            to_ego_norm = to_ego / (np.linalg.norm(to_ego) + 1e-6)
            vel_norm = ped_vel / (vel_magnitude + 1e-6)
            vel_alignment = np.dot(vel_norm, to_ego_norm)
            
            if vel_alignment > 0.5:
                return 0.8
            elif vel_alignment < -0.5:
                return 0.1
        
        return 0.3
    
    def _distance_to_nearest_crosswalk(self, map_api, position) -> float:
        """Find distance to nearest crosswalk"""
        try:
            if hasattr(map_api, 'get_proximal_map_objects'):
                nearby_objects = map_api.get_proximal_map_objects(position, 20.0, ['crosswalk'])
                crosswalks = nearby_objects.get('crosswalk', [])
                
                if len(crosswalks) == 0:
                    return float('inf')
                
                min_dist = float('inf')
                for crosswalk in crosswalks:
                    if hasattr(crosswalk, 'polygon') and hasattr(crosswalk.polygon, 'centroid'):
                        centroid = np.array([crosswalk.polygon.centroid.x, crosswalk.polygon.centroid.y])
                        dist = np.linalg.norm(position - centroid)
                        min_dist = min(min_dist, dist)
                
                return min_dist
        except:
            pass
        
        return float('inf')
    
    def _estimate_agent_occlusion(self, agent, all_agents, ego_pos) -> str:
        """Estimate if agent is occluded"""
        
        agent_pos = agent['center'][:2]
        agent_dist = np.linalg.norm(agent_pos - ego_pos)
        
        for other in all_agents:
            if other['instance_token'] == agent['instance_token']:
                continue
            
            other_pos = other['center'][:2]
            other_dist = np.linalg.norm(other_pos - ego_pos)
            
            if other_dist < agent_dist - 1.0:
                angle_agent = np.arctan2(agent_pos[1] - ego_pos[1], agent_pos[0] - ego_pos[0])
                angle_other = np.arctan2(other_pos[1] - ego_pos[1], other_pos[0] - ego_pos[0])
                angle_diff = abs(self._angle_difference(angle_agent, angle_other))
                
                if angle_diff < 0.15:
                    return 'occluded'
        
        if agent_dist > 25.0:
            return 'partially_visible'
        
        return 'visible'
    
    def _classify_pedestrian_behavior(self, ped_vel, crossing_intention, dist_to_crosswalk) -> str:
        """Classify pedestrian behavior"""
        
        vel_magnitude = np.linalg.norm(ped_vel)
        
        if crossing_intention > 0.7:
            return 'crossing'
        elif crossing_intention > 0.4 and dist_to_crosswalk < 3.0:
            return 'waiting_to_cross'
        elif vel_magnitude < 0.2:
            return 'standing'
        else:
            return 'walking'
    
    def _assess_pedestrian_risk(self, ped_pos, ego_pos, crossing_intention, occlusion_status) -> float:
        """Assess risk level posed by pedestrian"""
        
        distance = np.linalg.norm(ped_pos - ego_pos)
        distance_risk = max(0, 1.0 - distance / 20.0)
        intention_risk = crossing_intention
        
        occlusion_risk = {
            'visible': 0.0,
            'partially_visible': 0.3,
            'occluded': 0.7
        }.get(occlusion_status, 0.0)
        
        total_risk = 0.4 * distance_risk + 0.4 * intention_risk + 0.2 * occlusion_risk
        return min(total_risk, 1.0)
    
    def _predict_failure_modes(self, scene: Scene) -> Dict:
        """Predict potential failure modes"""
        
        complexity = self._compute_complexity(scene)
        interactions = self._extract_interactions(scene)
        pedestrians = self._analyze_pedestrians(scene)
        
        has_failures = False
        prediction_failures = []
        detection_failures = []
        planning_failures = []
        
        if complexity['occlusion_level'] > 0.5:
            has_failures = True
            detection_failures.append({
                'failure_type': 'occlusion_miss',
                'reason': f"High occlusion level: {complexity['occlusion_level']:.2f}"
            })
        
        for ped in pedestrians:
            if ped['risk_level'] > 0.6 and ped['occlusion_status'] != 'visible':
                has_failures = True
                prediction_failures.append({
                    'agent_id': ped['agent_id'],
                    'failure_type': 'pedestrian_surprise',
                    'reason': f"High-risk occluded pedestrian"
                })
        
        hard_interactions = [i for i in interactions if i['difficulty'] == 'hard']
        if len(hard_interactions) > 2:
            has_failures = True
            planning_failures.append({
                'failure_type': 'interaction_overload',
                'reason': f"Too many hard interactions: {len(hard_interactions)}"
            })
        
        return {
            'has_failures': bool(has_failures),
            'prediction_failures': prediction_failures,
            'detection_failures': detection_failures,
            'planning_failures': planning_failures
        }
    
    def _determine_encoding_needs(self, scene: Scene) -> Dict:
        """Determine which encoding levels this scene requires"""
        
        interactions = self._extract_interactions(scene)
        environment = self._classify_environment(scene)
        complexity = self._compute_complexity(scene)
        
        needs_temporal = True
        needs_interaction = len(interactions) > 0
        needs_scene = (
            environment['has_intersection'] or 
            environment['has_crosswalk'] or
            complexity['occlusion_level'] > 0.4
        )
        
        if needs_scene:
            critical_level = 'scene'
        elif needs_interaction:
            critical_level = 'interaction'
        else:
            critical_level = 'temporal'
        
        auxiliary_modules = []
        if environment['has_intersection']:
            auxiliary_modules.append('intersection')
        if complexity['pedestrian_count'] > 0:
            auxiliary_modules.append('pedestrian')
        if complexity['occlusion_level'] > 0.5:
            auxiliary_modules.append('occlusion')
        
        return {
            'needs_temporal': bool(needs_temporal),
            'needs_interaction': bool(needs_interaction),
            'needs_scene': bool(needs_scene),
            'critical_level': str(critical_level),
            'auxiliary_modules': auxiliary_modules
        }
    
    def _is_safety_critical(self, scene: Scene) -> bool:
        """Determine if scene is safety-critical"""
        
        pedestrians = self._analyze_pedestrians(scene)
        interactions = self._extract_interactions(scene)
        
        high_risk_peds = [p for p in pedestrians if p['risk_level'] > 0.7]
        hard_interactions = [i for i in interactions if i['difficulty'] == 'hard']
        failure_modes = self._predict_failure_modes(scene)
        
        return (
            len(high_risk_peds) > 0 or
            len(hard_interactions) > 0 or
            failure_modes['has_failures']
        )
    
    def _rate_difficulty(self, scene: Scene) -> str:
        """Rate overall scene difficulty"""
        
        complexity = self._compute_complexity(scene)
        
        if complexity['overall_score'] > 0.7:
            return 'hard'
        elif complexity['overall_score'] > 0.4:
            return 'medium'
        else:
            return 'easy'
    
    def _generate_tags(self, scene: Scene) -> List[str]:
        """Generate descriptive tags for the scene"""
        
        tags = []
        
        environment = self._classify_environment(scene)
        complexity = self._compute_complexity(scene)
        pedestrians = self._analyze_pedestrians(scene)
        interactions = self._extract_interactions(scene)
        failure_modes = self._predict_failure_modes(scene)
        
        if environment['has_intersection']:
            tags.append('intersection')
        if environment['has_crosswalk']:
            tags.append('crosswalk')
        if environment['has_traffic_light']:
            tags.append('traffic_light')
        
        if environment['road_type'] != 'unknown':
            tags.append(environment['road_type'])
        
        if complexity['pedestrian_count'] > 0:
            tags.append('pedestrian')
        if complexity['vehicle_count'] > 5:
            tags.append('multi_vehicle')
        if complexity['occlusion_level'] > 0.5:
            tags.append('occlusion')
        if complexity['agent_density'] > 10:
            tags.append('dense_traffic')
        
        interaction_types = set(i['type'] for i in interactions)
        for itype in interaction_types:
            tags.append(itype)
        
        if any(p['risk_level'] > 0.7 for p in pedestrians):
            tags.append('high_risk_pedestrian')
        
        if failure_modes['has_failures']:
            tags.append('failure_prone')
        
        return list(set(tags))


def main():
    """Main analysis function"""
    
    print("=" * 80)
    print("NAVSIM SCENE ANALYZER - Mini Dataset")
    print("=" * 80)
    
    # Setup paths
    data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    output_path = data_root / 'scene_analysis_mini.json'
    
    print(f"\nData root: {data_root}")
    print(f"Output path: {output_path}")
    
    # Create scene loader
    print("\nInitializing scene loader...")
    
    sensor_config = SensorConfig(
        cam_f0=True,
        cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False,
        cam_b0=False,
        lidar_pc=True
    )
    
    scene_filter = SceneFilter(
        log_names=None,
        num_history_frames=4,
        num_future_frames=8,
    )
    
    scene_loader = SceneLoader(
        data_path=data_root / 'mini_navsim_logs' / 'mini',
        original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',
        scene_filter=scene_filter,
        sensor_config=sensor_config
    )
    
    print(f"  Loaded {len(scene_loader.tokens)} scenes")
    
    # Initialize analyzer
    analyzer = SceneAnalyzer(history_length=4, future_length=8)
    
    # Analyze all scenes
    print("\nAnalyzing scenes...")
    all_metadata = []
    
    for token in tqdm(scene_loader.tokens, desc="Processing"):
        try:
            scene = scene_loader.get_scene_from_token(token)
            metadata = analyzer.analyze_scene(scene)
            all_metadata.append(metadata)
        except Exception as e:
            print(f"\nError analyzing scene {token}: {e}")
            continue
    
    # Save metadata
    print(f"\nSaving metadata to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"  Saved metadata for {len(all_metadata)} scenes")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal scenes analyzed: {len(all_metadata)}")
    print(f"Safety-critical scenes: {sum(m['safety_critical'] for m in all_metadata)}")
    
    print("\nDifficulty distribution:")
    for diff in ['easy', 'medium', 'hard']:
        count = sum(1 for m in all_metadata if m['difficulty_rating'] == diff)
        pct = 100 * count / len(all_metadata) if all_metadata else 0
        print(f"  {diff.capitalize()}: {count:3d} ({pct:5.1f}%)")
    
    print("\nEnvironment distribution:")
    env_counts = {
        'Intersections': sum(1 for m in all_metadata if m['environment']['has_intersection']),
        'Crosswalks': sum(1 for m in all_metadata if m['environment']['has_crosswalk']),
        'Traffic lights': sum(1 for m in all_metadata if m['environment']['has_traffic_light'])
    }
    for env_type, count in env_counts.items():
        pct = 100 * count / len(all_metadata) if all_metadata else 0
        print(f"  {env_type}: {count:3d} ({pct:5.1f}%)")
    
    print("\nAgent statistics:")
    if all_metadata:
        avg_pedestrians = sum(m['complexity']['pedestrian_count'] for m in all_metadata) / len(all_metadata)
        avg_vehicles = sum(m['complexity']['vehicle_count'] for m in all_metadata) / len(all_metadata)
        avg_occlusion = sum(m['complexity']['occlusion_level'] for m in all_metadata) / len(all_metadata)
        print(f"  Avg pedestrians per scene: {avg_pedestrians:.2f}")
        print(f"  Avg vehicles per scene: {avg_vehicles:.2f}")
        print(f"  Avg occlusion level: {avg_occlusion:.2f}")
    
    print("\nMost common tags:")
    from collections import Counter
    all_tags = []
    for m in all_metadata:
        all_tags.extend(m['tags'])
    tag_counts = Counter(all_tags)
    for tag, count in tag_counts.most_common(15):
        pct = 100 * count / len(all_metadata) if all_metadata else 0
        print(f"  {tag:25s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("  Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()