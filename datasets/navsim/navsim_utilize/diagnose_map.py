"""
Comprehensive map availability and feature inspection for NAVSIM dataset - FINAL FIX
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
import warnings
import numpy as np

# Set environments
os.environ['NUPLAN_MAPS_ROOT'] = '/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps'
os.environ['OPENSCENE_DATA_ROOT'] = '/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download'

# Import after setting env
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

def extract_objects_from_result(result):
    """
    Extract actual map objects from API result.
    NuPlan returns defaultdict(list) where keys are layer names.
    """
    if result is None:
        return []
    
    # If it's a defaultdict, extract values
    if isinstance(result, defaultdict) or isinstance(result, dict):
        all_objects = []
        for layer_objects in result.values():
            if isinstance(layer_objects, list):
                all_objects.extend(layer_objects)
            else:
                all_objects.append(layer_objects)
        return all_objects
    
    # If it's already a list
    if isinstance(result, list):
        return result
    
    # Single object
    return [result] if result else []

def check_map_features(map_api, map_name, ego_point):
    """
    Inspect what features are available in a map using REAL ego position.
    """
    features = {}
    radius = 50.0
    
    layers_to_check = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.INTERSECTION,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]
    
    print(f"\n  Checking available layers for '{map_name}' at ego position...")
    
    for layer in layers_to_check:
        try:
            result = map_api.get_proximal_map_objects(ego_point, radius, [layer])
            objects = extract_objects_from_result(result)
            count = len(objects)
            
            features[layer.name] = count
            status = " " if count > 0 else " (empty)"
            print(f"    {status} {layer.name}: {count} objects")
        except Exception as e:
            features[layer.name] = f"ERROR: {str(e)}"
            print(f"      {layer.name}: {str(e)}")
    
    return features

def inspect_lane_details(map_api, map_name, ego_point):
    """
    Inspect detailed lane information.
    """
    result = map_api.get_proximal_map_objects(
        ego_point, 50.0, [SemanticMapLayer.LANE]
    )
    
    lanes = extract_objects_from_result(result)
    
    if not lanes or len(lanes) == 0:
        print(f"\n    No lanes found at ego position for '{map_name}'")
        return None
    
    lane = lanes[0]
    
    print(f"\n   Sample Lane Attributes for '{map_name}':")
    print(f"    Lane ID: {lane.id}")
    
    lane_info = {}
    
    # Check available attributes
    attrs_to_check = [
        ('baseline_path', 'Centerline polyline'),
        ('speed_limit_mps', 'Speed limit'),
        ('type', 'Lane type'),
        ('incoming_edges', 'Predecessors'),
        ('outgoing_edges', 'Successors'),
    ]
    
    for attr, description in attrs_to_check:
        try:
            value = getattr(lane, attr, None)
            if value is not None:
                if attr == 'baseline_path':
                    path = value.discrete_path
                    lane_info[attr] = f"{len(path)} points"
                    print(f"      {description}: {len(path)} points")
                elif attr == 'incoming_edges':
                    lane_info[attr] = len(value)
                    print(f"      {description}: {len(value)} lanes")
                elif attr == 'outgoing_edges':
                    lane_info[attr] = len(value)
                    print(f"      {description}: {len(value)} lanes")
                else:
                    lane_info[attr] = value
                    print(f"      {description}: {value}")
            else:
                lane_info[attr] = None
                print(f"      {description}: None")
        except Exception as e:
            lane_info[attr] = f"ERROR: {str(e)}"
            print(f"      {description}: {str(e)}")
    
    # Check methods
    methods_to_check = [
        ('get_roadblock_id', 'Roadblock ID'),
        ('get_width', 'Lane width'),
    ]
    
    for method, description in methods_to_check:
        try:
            if hasattr(lane, method):
                value = getattr(lane, method)()
                lane_info[method] = value
                print(f"      {description}: {value}")
            else:
                print(f"      {description}: Method not found")
        except Exception as e:
            print(f"      {description}: {str(e)}")
    
    return lane_info

def test_actual_scene_extraction(scene_loader, map_api, map_name):
    """
    Test extraction using ACTUAL ego pose from a scene.
    """
    print(f"\n   Testing extraction with REAL ego pose from scene...")
    
    # Get a scene for this map
    for token in scene_loader.tokens[:20]:
        scene = scene_loader.get_scene_from_token(token)
        if scene.scene_metadata.map_name == map_name or \
           (map_name == "us-nv-las-vegas-strip" and scene.scene_metadata.map_name == "las_vegas"):
            
            frame = scene.frames[0]
            ego_status = frame.ego_status
            
            # Handle different ego_pose formats
            if hasattr(ego_status, 'ego_pose'):
                ego_pose = ego_status.ego_pose
                if isinstance(ego_pose, np.ndarray):
                    ego_x, ego_y = ego_pose[0], ego_pose[1]
                else:
                    ego_x, ego_y = ego_pose.x, ego_pose.y
            else:
                ego_x, ego_y = 0.0, 0.0
            
            ego_point = Point2D(ego_x, ego_y)
            
            print(f"    Ego position: ({ego_x:.2f}, {ego_y:.2f})")
            
            # Try to extract lanes around ego
            try:
                result = map_api.get_proximal_map_objects(
                    ego_point, 50.0, [SemanticMapLayer.LANE]
                )
                lanes = extract_objects_from_result(result)
                num_lanes = len(lanes)
                
                print(f"      Found {num_lanes} lanes within 50m of ego")
                
                if num_lanes > 0:
                    lane = lanes[0]
                    print(f"      Sample lane ID: {lane.id}")
                    
                    # Check connectivity
                    incoming = lane.incoming_edges if hasattr(lane, 'incoming_edges') else []
                    outgoing = lane.outgoing_edges if hasattr(lane, 'outgoing_edges') else []
                    print(f"      Connectivity: {len(incoming)} incoming, {len(outgoing)} outgoing")
                    
                    # Check baseline path
                    if hasattr(lane, 'baseline_path'):
                        path = lane.baseline_path.discrete_path
                        print(f"      Lane polyline: {len(path)} points")
                        
                        # Show first few points
                        if len(path) > 0:
                            print(f"      First point: ({path[0][0]:.2f}, {path[0][1]:.2f})")
                    
                    # Check speed limit
                    if hasattr(lane, 'speed_limit_mps') and lane.speed_limit_mps:
                        print(f"      Speed limit: {lane.speed_limit_mps:.1f} m/s")
                    
                    # Check lane type
                    if hasattr(lane, 'type'):
                        print(f"      Lane type: {lane.type}")
                    
                    return True
                else:
                    print(f"      No lanes found within 50m")
                    return False
                    
            except Exception as e:
                print(f"      Extraction failed: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            break
    
    print(f"      No scenes found for map '{map_name}'")
    return False

def check_mini_maps_comprehensive():
    """Comprehensive check of maps, features, and extractability."""
    
    data_root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    maps_root = Path(os.environ['NUPLAN_MAPS_ROOT'])
    
    # Verify paths exist
    if not data_root.exists():
        print(f" ERROR: OPENSCENE_DATA_ROOT not found: {data_root}")
        return None
    
    if not maps_root.exists():
        print(f" ERROR: NUPLAN_MAPS_ROOT not found: {maps_root}")
        return None
    
    # Minimal sensor config
    sensor_config = SensorConfig(
        cam_f0=False, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
        lidar_pc=False
    )
    
    scene_filter = SceneFilter(
        log_names=None,
        num_history_frames=4,
        num_future_frames=8,
    )
    
    print("=" * 80)
    print(" NAVSIM MAP AVAILABILITY & FEATURE INSPECTION")
    print("=" * 80)
    
    # Step 1: Load scenes
    print("\n[1/5] Loading NAVSIM scenes...")
    print("-" * 80)
    
    try:
        scene_loader = SceneLoader(
            data_path=data_root / 'mini_navsim_logs' / 'mini',
            original_sensor_path=data_root / 'mini_sensor_blobs' / 'mini',
            scene_filter=scene_filter,
            sensor_config=sensor_config
        )
        print(f"  Loaded {len(scene_loader.tokens)} scenes")
    except Exception as e:
        print(f" Failed to load scenes: {str(e)}")
        return None
    
    # Step 2: Check map usage in scenes
    print("\n[2/5] Analyzing map usage in scenes...")
    print("-" * 80)
    
    map_counter = Counter()
    map_examples = {}
    
    num_scenes_to_check = min(50, len(scene_loader.tokens))
    
    for i, token in enumerate(scene_loader.tokens[:num_scenes_to_check]):
        scene = scene_loader.get_scene_from_token(token)
        map_name = scene.scene_metadata.map_name
        
        map_counter[map_name] += 1
        
        if map_name not in map_examples:
            map_examples[map_name] = token
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{num_scenes_to_check} scenes...")
    
    print(f"\n  Analyzed {num_scenes_to_check} scenes")
    print(f"\nMap distribution:")
    for map_name, count in map_counter.most_common():
        print(f"  • {map_name}: {count} scenes ({count/num_scenes_to_check*100:.1f}%)")
    
    # Step 3: Check available maps in directory
    print("\n[3/5] Checking available maps in directory...")
    print("-" * 80)
    print(f"Map directory: {maps_root}")
    
    available_maps = {}
    for item in maps_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            has_structure = (item / 'map.json').exists() or list(item.glob('**/map.json'))
            available_maps[item.name] = has_structure
            status = " " if has_structure else " (may work)"
            print(f"  {status} {item.name}")
    
    # Step 4: Get ego position from first scene
    print("\n[4/5] Getting real ego position for map testing...")
    print("-" * 80)
    
    first_scene = scene_loader.get_scene_from_token(scene_loader.tokens[0])
    first_frame = first_scene.frames[0]
    ego_status = first_frame.ego_status
    
    if hasattr(ego_status, 'ego_pose'):
        ego_pose = ego_status.ego_pose
        if isinstance(ego_pose, np.ndarray):
            ego_x, ego_y = ego_pose[0], ego_pose[1]
        else:
            ego_x, ego_y = ego_pose.x, ego_pose.y
    else:
        ego_x, ego_y = 0.0, 0.0
    
    test_ego_point = Point2D(ego_x, ego_y)
    print(f"  Using ego position: ({ego_x:.2f}, {ego_y:.2f})")
    
    # Step 5: Check map API access and features
    print("\n[5/5] Testing map API access and extractable features...")
    print("-" * 80)
    
    used_maps = set(map_counter.keys())
    map_version = "nuplan-maps-v1.0"
    
    map_features_summary = {}
    map_api_cache = {}
    extraction_test_results = {}
    
    for map_name in used_maps:
        print(f"\n{'='*80}")
        print(f"Map: {map_name}")
        print(f"{'='*80}")
        
        # Check if map needs name conversion
        converted_name = map_name
        if map_name == "las_vegas":
            converted_name = "us-nv-las-vegas-strip"
            print(f"   Name conversion: '{map_name}' → '{converted_name}'")
        
        # Check if available
        if converted_name not in available_maps:
            print(f"    Map NOT FOUND in directory!")
            map_features_summary[map_name] = "NOT_AVAILABLE"
            extraction_test_results[map_name] = False
            continue
        
        # Try to load map API
        try:
            map_api = get_maps_api(str(maps_root), map_version, converted_name)
            print(f"    Map API loaded successfully")
            map_api_cache[map_name] = map_api
            
            # Check available features
            features = check_map_features(map_api, map_name, test_ego_point)
            map_features_summary[map_name] = features
            
            # Inspect lane details
            lane_info = inspect_lane_details(map_api, map_name, test_ego_point)
            
            # Test extraction with real scene
            success = test_actual_scene_extraction(scene_loader, map_api, map_name)
            extraction_test_results[map_name] = success
            
        except Exception as e:
            print(f"    Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            map_features_summary[map_name] = f"ERROR: {str(e)}"
            extraction_test_results[map_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    missing_maps = used_maps - set(available_maps.keys())
    if "las_vegas" in missing_maps and "us-nv-las-vegas-strip" in available_maps:
        missing_maps.remove("las_vegas")
    
    if missing_maps:
        print("\n  WARNING: Missing maps:")
        for m in missing_maps:
            print(f"  • {m}")
    else:
        print("\n  All required maps are available!")
    
    # Feature extractability
    print("\n Feature Extractability Summary:")
    
    extractable_features = defaultdict(list)
    for map_name, features in map_features_summary.items():
        if isinstance(features, dict):
            print(f"\n  Map: {map_name}")
            for layer, count in features.items():
                if isinstance(count, int) and count > 0:
                    extractable_features[layer].append(map_name)
                    print(f"      {layer}: {count} objects")
                elif isinstance(count, int):
                    print(f"      {layer}: empty")
                else:
                    print(f"      {layer}: {count}")
    
    print("\n" + "=" * 80)
    print(" VECTOR MAP EXTRACTION READINESS")
    print("=" * 80)
    
    critical_layers = ['LANE', 'LANE_CONNECTOR', 'INTERSECTION', 'STOP_LINE', 'CROSSWALK']
    
    print("\nCritical layers for VectorMapExtractor:")
    all_ready = True
    for layer in critical_layers:
        maps_with_layer = extractable_features.get(layer, [])
        if len(maps_with_layer) == len(used_maps):
            print(f"    {layer}: Available in all maps")
        elif maps_with_layer:
            print(f"    {layer}: Available in {len(maps_with_layer)}/{len(used_maps)} maps")
            all_ready = False
        else:
            print(f"    {layer}: Not available")
            all_ready = False
    
    print("\n Real Scene Extraction Test Results:")
    for map_name, success in extraction_test_results.items():
        status = " " if success else "✗"
        print(f"  {status} {map_name}")
    
    if all_ready and all(extraction_test_results.values()):
        print("\n" + " " * 20)
        print("  ALL SYSTEMS GO!")
        print("  You can proceed with VectorMapExtractor implementation.")
        print(" " * 20)
    else:
        print("\n  Issues detected - you may need fallback handling")
    
    print("\n" + "=" * 80)
    
    return {
        'map_counter': map_counter,
        'available_maps': available_maps,
        'map_features': map_features_summary,
        'extractable_features': extractable_features,
        'extraction_tests': extraction_test_results,
    }

if __name__ == "__main__":
    results = check_mini_maps_comprehensive()
    
    if results:
        print("\n Inspection complete!")
    else:
        print("\n Inspection failed.")