#!/usr/bin/env python3
"""
Dataset Inspector v2 — with correct paths.
Checks all 4 data sources, how they connect, and whether they duplicate.
"""

import os
import sys
import pickle
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# CORRECTED PATHS
# ============================================================
META_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene_metadata_mini/openscene-v1.1/meta_datas/mini"
OCC_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/mini"
SENSOR_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"
NAVSIM_LOG_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_navsim_logs/mini"
MAPS_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps"

CAMERA_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 1. META_DATAS (OpenScene v1.1)
# ============================================================
def inspect_metadata():
    sep("1. OPENSCENE META_DATAS (v1.1)")
    
    if not os.path.isdir(META_ROOT):
        print(f"  NOT FOUND: {META_ROOT}")
        return None
    
    pkl_files = sorted(glob.glob(os.path.join(META_ROOT, "*.pkl")))
    print(f"  Path: {META_ROOT}")
    print(f"  Pkl files: {len(pkl_files)}")
    
    if not pkl_files:
        print("  No pkl files found!")
        return None
    
    # Load first pkl
    with open(pkl_files[0], "rb") as f:
        data = pickle.load(f)
    
    frames = data if isinstance(data, list) else []
    print(f"  First pkl ({os.path.basename(pkl_files[0])}): {len(frames)} frames")
    
    if frames and isinstance(frames[0], dict):
        f0 = frames[0]
        print(f"  Frame keys: {sorted(f0.keys())}")
        print(f"  token: '{f0.get('token', '?')}'")
        print(f"  lidar_path: '{f0.get('lidar_path', '?')}'")
        print(f"  scene_name: '{f0.get('scene_name', '?')}'")
        print(f"  log_name: '{f0.get('log_name', '?')}'")
        print(f"  occ_gt_final_path: '{f0.get('occ_gt_final_path', '?')}'")
        print(f"  flow_gt_final_path: '{f0.get('flow_gt_final_path', '?')}'")
        
        # Camera path format
        cams = f0.get("cams", {})
        if "CAM_F0" in cams:
            print(f"  CAM_F0 data_path: '{cams['CAM_F0'].get('data_path', '?')}'")
            print(f"  CAM_F0 keys: {sorted(cams['CAM_F0'].keys())}")
        
        # ego_dynamic_state format
        eds = f0.get("ego_dynamic_state", None)
        print(f"  ego_dynamic_state type: {type(eds).__name__}, value: {eds}")
    
    # Scan all: collect tokens, scene_names, log_names, occ paths
    all_tokens = []
    all_scenes = set()
    all_logs = set()
    all_occ_paths = set()
    token_to_scene = {}
    token_to_occ = {}
    
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            log_data = pickle.load(f)
        flist = log_data if isinstance(log_data, list) else []
        for fr in flist:
            tok = fr.get("token", "")
            scene = fr.get("scene_name", "")
            log = fr.get("log_name", "")
            occ_path = fr.get("occ_gt_final_path", "")
            all_tokens.append(tok)
            all_scenes.add(scene)
            all_logs.add(log)
            if occ_path:
                all_occ_paths.add(occ_path)
            token_to_scene[tok] = scene
            token_to_occ[tok] = occ_path
    
    print(f"\n  Total frames: {len(all_tokens)}")
    print(f"  Unique tokens: {len(set(all_tokens))}")
    print(f"  Unique scenes: {len(all_scenes)}")
    print(f"  Unique logs: {len(all_logs)}")
    print(f"  Frames with occ_gt_path: {sum(1 for p in all_occ_paths if p)}")
    
    # Show scene name format
    scene_samples = sorted(all_scenes)[:5]
    print(f"  Scene samples: {scene_samples}")
    
    # Show occ path format
    occ_samples = sorted(all_occ_paths)[:3]
    print(f"  Occ path samples: {occ_samples}")
    
    return {
        "pkl_files": pkl_files,
        "tokens": set(all_tokens),
        "token_list": all_tokens,
        "scenes": all_scenes,
        "logs": all_logs,
        "token_to_scene": token_to_scene,
        "token_to_occ": token_to_occ,
        "occ_paths": all_occ_paths,
    }


# ============================================================
# 2. NAVSIM LOGS (mini)
# ============================================================
def inspect_navsim_logs():
    sep("2. NAVSIM LOGS")
    
    if not os.path.isdir(NAVSIM_LOG_ROOT):
        print(f"  NOT FOUND: {NAVSIM_LOG_ROOT}")
        return None
    
    print(f"  Path: {NAVSIM_LOG_ROOT}")
    files = sorted(os.listdir(NAVSIM_LOG_ROOT))
    pkl_files = [f for f in files if f.endswith('.pkl')]
    db_files = [f for f in files if f.endswith('.db')]
    other = [f for f in files if not f.endswith(('.pkl', '.db'))]
    
    print(f"  Files: {len(pkl_files)} pkl, {len(db_files)} db, {len(other)} other")
    if pkl_files:
        print(f"  Pkl samples: {pkl_files[:3]}")
    if db_files:
        print(f"  DB samples: {db_files[:3]}")
    
    # Check if navsim log pkls have the same names as metadata pkls
    navsim_log_names = set(Path(f).stem for f in pkl_files)
    
    # Load one pkl to see format
    if pkl_files:
        sample = os.path.join(NAVSIM_LOG_ROOT, pkl_files[0])
        print(f"\n  --- Loading {pkl_files[0]} ---")
        try:
            with open(sample, "rb") as f:
                data = pickle.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, list):
                print(f"  Length: {len(data)}")
                if data and isinstance(data[0], dict):
                    print(f"  Frame keys: {sorted(data[0].keys())}")
                    # Show a few key fields
                    f0 = data[0]
                    for k in ["token", "lidar_path", "scene_name", "log_name", "ego_dynamic_state"]:
                        if k in f0:
                            v = f0[k]
                            if isinstance(v, str):
                                print(f"    {k}: '{v}'")
                            elif isinstance(v, np.ndarray):
                                print(f"    {k}: ndarray shape={v.shape}")
                            elif isinstance(v, (list, dict)):
                                print(f"    {k}: {type(v).__name__} len={len(v)}")
                            else:
                                print(f"    {k}: {v}")
            elif isinstance(data, dict):
                print(f"  Keys: {sorted(data.keys())}")
        except Exception as e:
            print(f"  Error loading: {e}")
    
    return {
        "pkl_names": navsim_log_names,
        "db_files": db_files,
    }


# ============================================================
# 3. SENSOR_BLOBS
# ============================================================
def inspect_sensors():
    sep("3. SENSOR_BLOBS")
    
    if not os.path.isdir(SENSOR_ROOT):
        print(f"  NOT FOUND: {SENSOR_ROOT}")
        return None
    
    print(f"  Path: {SENSOR_ROOT}")
    log_dirs = sorted([d for d in os.listdir(SENSOR_ROOT) if os.path.isdir(os.path.join(SENSOR_ROOT, d))])
    print(f"  Log directories: {len(log_dirs)}")
    if log_dirs:
        print(f"  Samples: {log_dirs[:3]}")
    
    # Inspect first log
    all_sensor_tokens = set()
    total_cam = 0
    total_lidar = 0
    
    for i, log_dir in enumerate(log_dirs):
        log_path = os.path.join(SENSOR_ROOT, log_dir)
        subdirs = sorted(os.listdir(log_path))
        
        # Count CAM_F0
        cam_f0 = os.path.join(log_path, "CAM_F0")
        n_cam = 0
        if os.path.isdir(cam_f0):
            imgs = [f for f in os.listdir(cam_f0) if f.endswith(('.jpg', '.png'))]
            n_cam = len(imgs)
            for img in imgs:
                all_sensor_tokens.add(Path(img).stem)
        
        # Count LiDAR
        lidar_dir = os.path.join(log_path, "MergedPointCloud")
        n_lidar = 0
        if os.path.isdir(lidar_dir):
            pcds = [f for f in os.listdir(lidar_dir)]
            n_lidar = len(pcds)
            for pcd in pcds:
                all_sensor_tokens.add(Path(pcd).stem)
        
        total_cam += n_cam
        total_lidar += n_lidar
        
        if i < 3:
            print(f"    {log_dir}: subdirs={subdirs}, CAM_F0={n_cam}, LiDAR={n_lidar}")
    
    print(f"\n  Total camera frames (CAM_F0): {total_cam}")
    print(f"  Total LiDAR frames: {total_lidar}")
    print(f"  Unique sensor tokens: {len(all_sensor_tokens)}")
    
    return {
        "log_dirs": set(log_dirs),
        "tokens": all_sensor_tokens,
        "total_cam": total_cam,
        "total_lidar": total_lidar,
    }


# ============================================================
# 4. OCCUPANCY LABELS
# ============================================================
def inspect_occ():
    sep("4. OCCUPANCY LABELS")
    
    if not os.path.isdir(OCC_ROOT):
        print(f"  NOT FOUND: {OCC_ROOT}")
        return None
    
    print(f"  Path: {OCC_ROOT}")
    
    # Structure: mini/log-XXXX-scene-XXXX/occ_gt/NNN_occ_final.npy
    scene_dirs = sorted([d for d in os.listdir(OCC_ROOT) if os.path.isdir(os.path.join(OCC_ROOT, d))])
    print(f"  Scene directories: {len(scene_dirs)}")
    if scene_dirs:
        print(f"  Samples: {scene_dirs[:5]}")
    
    # For each scene, check occ_gt contents
    occ_files = {}  # scene_name -> list of (frame_idx, path)
    total_occ = 0
    total_flow = 0
    
    for scene_dir in scene_dirs:
        occ_gt_dir = os.path.join(OCC_ROOT, scene_dir, "occ_gt")
        if not os.path.isdir(occ_gt_dir):
            continue
        
        files = sorted(os.listdir(occ_gt_dir))
        occ_npy = [f for f in files if f.endswith("_occ_final.npy")]
        flow_npy = [f for f in files if f.endswith("_flow_final.npy")]
        total_occ += len(occ_npy)
        total_flow += len(flow_npy)
        
        occ_files[scene_dir] = occ_npy
    
    print(f"\n  Total occ label files: {total_occ}")
    print(f"  Total flow label files: {total_flow}")
    print(f"  Scenes with occ labels: {len(occ_files)}")
    
    # Inspect naming: NNN_occ_final.npy where NNN is frame index
    if scene_dirs:
        sample_scene = scene_dirs[0]
        occ_gt_dir = os.path.join(OCC_ROOT, sample_scene, "occ_gt")
        if os.path.isdir(occ_gt_dir):
            sample_files = sorted(os.listdir(occ_gt_dir))[:6]
            print(f"\n  Sample ({sample_scene}/occ_gt/):")
            for f in sample_files:
                print(f"    {f}")
    
    # Load one occ label
    first_occ = None
    for scene_dir in scene_dirs:
        occ_gt_dir = os.path.join(OCC_ROOT, scene_dir, "occ_gt")
        if os.path.isdir(occ_gt_dir):
            for f in sorted(os.listdir(occ_gt_dir)):
                if f.endswith("_occ_final.npy"):
                    first_occ = os.path.join(occ_gt_dir, f)
                    break
        if first_occ:
            break
    
    if first_occ:
        print(f"\n  --- Loading sample occ: {os.path.basename(first_occ)} ---")
        arr = np.load(first_occ)
        print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  Min: {arr.min()}, Max: {arr.max()}")
        unique_vals = np.unique(arr)
        if len(unique_vals) <= 20:
            print(f"  Unique values: {unique_vals}")
            print(f"  Value counts: {dict(zip(*np.unique(arr, return_counts=True)))}")
        else:
            print(f"  Unique values: {len(unique_vals)} (range {unique_vals.min()}-{unique_vals.max()})")
    
    # Load one flow label
    first_flow = None
    for scene_dir in scene_dirs:
        occ_gt_dir = os.path.join(OCC_ROOT, scene_dir, "occ_gt")
        if os.path.isdir(occ_gt_dir):
            for f in sorted(os.listdir(occ_gt_dir)):
                if f.endswith("_flow_final.npy"):
                    first_flow = os.path.join(occ_gt_dir, f)
                    break
        if first_flow:
            break
    
    if first_flow:
        print(f"\n  --- Loading sample flow: {os.path.basename(first_flow)} ---")
        arr = np.load(first_flow)
        print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    
    return {
        "scene_dirs": scene_dirs,
        "occ_files": occ_files,
        "total_occ": total_occ,
        "total_flow": total_flow,
    }


# ============================================================
# 5. CROSS-REFERENCE EVERYTHING
# ============================================================
def cross_reference(meta, navsim_logs, sensors, occ):
    sep("5. CROSS-REFERENCE: How everything connects")
    
    # 5a. Meta log names vs Sensor log dirs vs NavSim log names
    print("\n  --- Log-level matching ---")
    meta_logs = meta["logs"] if meta else set()
    sensor_logs = sensors["log_dirs"] if sensors else set()
    navsim_log_names = navsim_logs["pkl_names"] if navsim_logs else set()
    
    print(f"  Meta logs: {len(meta_logs)}")
    print(f"  Sensor log dirs: {len(sensor_logs)}")
    print(f"  NavSim log pkls: {len(navsim_log_names)}")
    
    if meta_logs and sensor_logs:
        overlap = meta_logs & sensor_logs
        print(f"  Meta ∩ Sensor: {len(overlap)} logs match")
        if len(overlap) < len(meta_logs):
            missing = meta_logs - sensor_logs
            print(f"  Meta logs missing from sensors: {len(missing)}")
            if len(missing) <= 5:
                print(f"    {missing}")
    
    if meta_logs and navsim_log_names:
        overlap = meta_logs & navsim_log_names
        print(f"  Meta ∩ NavSim logs: {len(overlap)} match")
    
    if navsim_log_names and sensor_logs:
        overlap = navsim_log_names & sensor_logs
        print(f"  NavSim logs ∩ Sensor: {len(overlap)} match")
    
    # 5b. Meta vs NavSim logs: are they duplicates?
    if meta and navsim_logs and navsim_log_names:
        print("\n  --- Are metadata and navsim_logs duplicates? ---")
        # Compare a shared log
        shared = meta_logs & navsim_log_names
        if shared:
            test_log = list(shared)[0]
            # Load from meta
            meta_pkl = None
            for p in meta["pkl_files"]:
                if Path(p).stem == test_log:
                    meta_pkl = p
                    break
            navsim_pkl = os.path.join(NAVSIM_LOG_ROOT, test_log + ".pkl")
            
            if meta_pkl and os.path.exists(navsim_pkl):
                with open(meta_pkl, "rb") as f:
                    meta_data = pickle.load(f)
                with open(navsim_pkl, "rb") as f:
                    navsim_data = pickle.load(f)
                
                mframes = meta_data if isinstance(meta_data, list) else []
                nframes = navsim_data if isinstance(navsim_data, list) else []
                
                print(f"  Test log: {test_log}")
                print(f"  Meta frames: {len(mframes)}")
                print(f"  NavSim frames: {len(nframes)}")
                
                if mframes and nframes:
                    mkeys = sorted(mframes[0].keys())
                    nkeys = sorted(nframes[0].keys())
                    print(f"  Meta frame keys:   {mkeys}")
                    print(f"  NavSim frame keys: {nkeys}")
                    
                    common_keys = set(mkeys) & set(nkeys)
                    only_meta = set(mkeys) - set(nkeys)
                    only_navsim = set(nkeys) - set(mkeys)
                    
                    print(f"\n  Common keys: {sorted(common_keys)}")
                    if only_meta:
                        print(f"  Only in meta: {sorted(only_meta)}")
                    if only_navsim:
                        print(f"  Only in navsim: {sorted(only_navsim)}")
                    
                    # Check if tokens match
                    m_tokens = [f.get("token", "") for f in mframes[:5]]
                    n_tokens = [f.get("token", "") for f in nframes[:5]]
                    print(f"\n  Meta tokens (first 5):   {m_tokens}")
                    print(f"  NavSim tokens (first 5): {n_tokens}")
                    tokens_match = m_tokens == n_tokens
                    print(f"  Tokens match: {tokens_match}")
    
    # 5c. Meta tokens vs Sensor tokens
    if meta and sensors:
        print("\n  --- Token-level: Meta ↔ Sensors ---")
        meta_tokens = meta["tokens"]
        sensor_tokens = sensors["tokens"]
        overlap = meta_tokens & sensor_tokens
        print(f"  Meta tokens: {len(meta_tokens)}")
        print(f"  Sensor tokens: {len(sensor_tokens)}")
        print(f"  Overlap: {len(overlap)}")
        print(f"  Meta without sensors: {len(meta_tokens - sensor_tokens)}")
        print(f"  Sensors without meta: {len(sensor_tokens - meta_tokens)}")
    
    # 5d. Meta occ_gt_path ↔ actual occ files
    if meta and occ:
        print("\n  --- Meta occ_gt_final_path ↔ actual occ files ---")
        
        # Meta has paths like: dataset/openscene-v1.0/occupancy/mini/log-0001-scene-0001/occ_gt/000_occ_final.npy
        # Actual files are at: OCC_ROOT/log-0001-scene-0001/occ_gt/000_occ_final.npy
        
        # Extract scene+frame from meta occ paths
        meta_occ_scene_frames = set()
        for occ_path in meta["occ_paths"]:
            # Parse: .../mini/log-XXXX-scene-XXXX/occ_gt/NNN_occ_final.npy
            parts = occ_path.replace("\\", "/").split("/")
            # Find the scene part
            for i, p in enumerate(parts):
                if p.startswith("log-") and "scene" in p:
                    scene_name = p
                    if i + 2 < len(parts):
                        frame_file = parts[i + 2]  # e.g., 000_occ_final.npy
                        meta_occ_scene_frames.add((scene_name, frame_file))
                    break
        
        # Actual occ files
        actual_occ_scene_frames = set()
        for scene_dir, occ_list in occ["occ_files"].items():
            for occ_file in occ_list:
                actual_occ_scene_frames.add((scene_dir, occ_file))
        
        overlap = meta_occ_scene_frames & actual_occ_scene_frames
        print(f"  Meta occ references (scene, file): {len(meta_occ_scene_frames)}")
        print(f"  Actual occ files (scene, file): {len(actual_occ_scene_frames)}")
        print(f"  Overlap: {len(overlap)}")
        
        if meta_occ_scene_frames and actual_occ_scene_frames:
            m_sample = list(meta_occ_scene_frames)[:3]
            a_sample = list(actual_occ_scene_frames)[:3]
            print(f"  Meta samples: {m_sample}")
            print(f"  Actual samples: {a_sample}")
        
        # Check: how many meta frames have a matching occ file on disk?
        matched = 0
        unmatched = 0
        for tok, occ_path in list(meta["token_to_occ"].items())[:1000]:
            if not occ_path:
                continue
            # Try to resolve path
            # Remove prefix up to and including 'mini/'
            parts = occ_path.replace("\\", "/").split("/")
            mini_idx = None
            for i, p in enumerate(parts):
                if p == "mini":
                    mini_idx = i
                    break
            if mini_idx is not None:
                rel_path = "/".join(parts[mini_idx + 1:])
                full_path = os.path.join(OCC_ROOT, rel_path)
                if os.path.exists(full_path):
                    matched += 1
                else:
                    if unmatched < 3:
                        print(f"    Missing: {full_path}")
                    unmatched += 1
        
        print(f"\n  Verified (first 1000 meta frames):")
        print(f"    Occ file exists on disk: {matched}")
        print(f"    Occ file MISSING: {unmatched}")
    
    # 5e. Verify sensor file exists for meta tokens
    if meta and sensors:
        print("\n  --- Verify sensor files exist for meta tokens ---")
        # Check first log
        test_log = list(meta["logs"])[0]
        test_tokens = [tok for tok, scene in meta["token_to_scene"].items() 
                       if any(test_log in (meta["token_to_occ"].get(tok, "") or "")
                              for _ in [1])][:10]
        
        # Just check first few meta tokens
        sample_tokens = list(meta["tokens"])[:10]
        found_cam = 0
        found_lidar = 0
        for tok in sample_tokens:
            # Search in sensor blobs
            for log_dir in (sensors["log_dirs"] if sensors else []):
                cam_path = os.path.join(SENSOR_ROOT, log_dir, "CAM_F0", tok + ".jpg")
                lidar_path = os.path.join(SENSOR_ROOT, log_dir, "MergedPointCloud", tok + ".pcd")
                if os.path.exists(cam_path):
                    found_cam += 1
                    break
            for log_dir in (sensors["log_dirs"] if sensors else []):
                lidar_path = os.path.join(SENSOR_ROOT, log_dir, "MergedPointCloud", tok + ".pcd")
                if os.path.exists(lidar_path):
                    found_lidar += 1
                    break
        
        print(f"  Sample of {len(sample_tokens)} meta tokens:")
        print(f"    CAM_F0 image found: {found_cam}")
        print(f"    LiDAR pcd found: {found_lidar}")


# ============================================================
# 6. SUMMARY
# ============================================================
def summary(meta, navsim_logs, sensors, occ):
    sep("6. SUMMARY")
    
    print(f"\n  DATA SOURCES:")
    if meta:
        print(f"    Meta (OpenScene v1.1): {len(meta['tokens'])} frames, {len(meta['pkl_files'])} logs")
        print(f"      Has: token, lidar_path, cams (with calibration), ego_dynamic_state,")
        print(f"           occ_gt_final_path, anns (3D boxes), ego2global, lidar2ego")
    if navsim_logs:
        print(f"    NavSim logs: {len(navsim_logs['pkl_names'])} log pkls")
    if sensors:
        print(f"    Sensors: {len(sensors['log_dirs'])} logs, {sensors['total_cam']} cam, {sensors['total_lidar']} lidar")
    if occ:
        print(f"    Occupancy: {len(occ['scene_dirs'])} scenes, {occ['total_occ']} occ labels, {occ['total_flow']} flow labels")
    
    print(f"\n  KEY QUESTIONS:")
    print(f"    1. Do meta tokens match sensor file hashes? (needed to load images/lidar)")
    print(f"    2. Do meta occ_gt paths resolve to actual files? (needed for Phase 1 training)")
    print(f"    3. Are navsim_logs and meta_datas duplicates? (if so, we only need one)")
    print(f"    4. Can we build one combined pkl from meta that maps each frame to")
    print(f"       its sensor files + occ labels? (the openscene_infos_train.pkl)")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Dataset Inspector v2 — Correct Paths")
    print(f"Date: {__import__('datetime').datetime.now()}")
    
    meta = inspect_metadata()
    navsim_logs = inspect_navsim_logs()
    sensors = inspect_sensors()
    occ = inspect_occ()
    cross_reference(meta, navsim_logs, sensors, occ)
    summary(meta, navsim_logs, sensors, occ)