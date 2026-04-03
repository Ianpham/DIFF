#!/usr/bin/env python3
"""
Dataset Inspector: Check all data sources and how they connect.

Inspects:
  1. meta_datas (per-log pkl files) — the index
  2. sensor_blobs (cameras + LiDAR) — the actual sensor data
  3. occupancy labels — the ground truth
  4. maps — HD map data
  
Shows: what fields exist, how tokens/paths link across sources,
and whether the data can be properly concatenated for training.
"""

import os
import sys
import pickle
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# PATHS — edit these to match your machine
# ============================================================
DATA_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/"
OCC_LABEL_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene-v1.0/occupancy/"
META_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/occ_mini/openscene_metadata_mini"
NAVSIM_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/"
MAPS_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/maps/"

CAMERA_NAMES = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ============================================================
# 1. INSPECT META_DATAS
# ============================================================
def inspect_metadata():
    sep("1. META_DATAS (OpenScene per-log pkl files)")
    
    if not os.path.isdir(META_ROOT):
        print(f"  NOT FOUND: {META_ROOT}")
        return None
    
    # List contents
    contents = sorted(os.listdir(META_ROOT))
    pkl_files = [f for f in contents if f.endswith('.pkl')]
    subdirs = [f for f in contents if os.path.isdir(os.path.join(META_ROOT, f))]
    
    print(f"  Path: {META_ROOT}")
    print(f"  Direct pkl files: {len(pkl_files)}")
    print(f"  Subdirectories: {subdirs}")
    
    # If pkls are in subdirectories
    all_pkls = []
    if pkl_files:
        all_pkls = [os.path.join(META_ROOT, f) for f in pkl_files]
    else:
        for sd in subdirs:
            sd_path = os.path.join(META_ROOT, sd)
            sd_pkls = sorted(glob.glob(os.path.join(sd_path, "*.pkl")))
            print(f"    {sd}/: {len(sd_pkls)} pkl files")
            all_pkls.extend(sd_pkls)
    
    if not all_pkls:
        # Try recursive
        all_pkls = sorted(glob.glob(os.path.join(META_ROOT, "**", "*.pkl"), recursive=True))
        print(f"  Recursive search: {len(all_pkls)} pkl files")
    
    print(f"\n  Total pkl files found: {len(all_pkls)}")
    
    if not all_pkls:
        return None
    
    # Load first pkl and inspect structure
    print(f"\n  --- Inspecting first pkl: {os.path.basename(all_pkls[0])} ---")
    with open(all_pkls[0], "rb") as f:
        data = pickle.load(f)
    
    print(f"  Type: {type(data)}")
    
    if isinstance(data, list):
        print(f"  Length: {len(data)} frames")
        if data:
            frame0 = data[0]
            print(f"  Frame type: {type(frame0)}")
            if isinstance(frame0, dict):
                print(f"  Frame keys: {sorted(frame0.keys())}")
                # Print each key's type and sample value
                for k, v in sorted(frame0.items()):
                    if isinstance(v, np.ndarray):
                        print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype}")
                    elif isinstance(v, dict):
                        print(f"    {k}: dict with keys={sorted(v.keys())}")
                        # Go one level deeper for important fields
                        if k == "cams" and v:
                            first_cam = list(v.keys())[0]
                            print(f"      {first_cam}: {sorted(v[first_cam].keys())}")
                        elif k == "ego_dynamic_state":
                            for ek, ev in v.items():
                                print(f"      {ek}: {type(ev).__name__} = {ev if not isinstance(ev, np.ndarray) else f'shape={ev.shape}'}")
                        elif k == "anns":
                            for ak, av in v.items():
                                if isinstance(av, (list, np.ndarray)):
                                    length = len(av)
                                    print(f"      {ak}: len={length}")
                                else:
                                    print(f"      {ak}: {type(av).__name__}")
                    elif isinstance(v, list):
                        print(f"    {k}: list len={len(v)}")
                    elif isinstance(v, str):
                        print(f"    {k}: str = '{v[:80]}{'...' if len(v)>80 else ''}'")
                    else:
                        print(f"    {k}: {type(v).__name__} = {v}")
    
    elif isinstance(data, dict):
        print(f"  Keys: {sorted(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list):
                print(f"    {k}: list len={len(v)}")
            elif isinstance(v, dict):
                print(f"    {k}: dict keys={list(v.keys())[:5]}")
            else:
                print(f"    {k}: {type(v).__name__}")
    
    # Count total frames across all pkls
    total_frames = 0
    all_tokens = set()
    all_log_names = set()
    frame_sample = None
    
    print(f"\n  --- Scanning all {len(all_pkls)} pkl files ---")
    for pkl_path in all_pkls:
        with open(pkl_path, "rb") as f:
            log_data = pickle.load(f)
        
        frames = log_data if isinstance(log_data, list) else log_data.get("infos", log_data.get("data_list", []))
        total_frames += len(frames)
        
        for fr in frames:
            if isinstance(fr, dict):
                if "token" in fr:
                    all_tokens.add(fr["token"])
                if "log_name" in fr:
                    all_log_names.add(fr["log_name"])
                if frame_sample is None:
                    frame_sample = fr
    
    print(f"  Total frames across all pkls: {total_frames}")
    print(f"  Unique tokens: {len(all_tokens)}")
    print(f"  Unique log names: {len(all_log_names)}")
    if all_log_names:
        print(f"  Log names sample: {list(all_log_names)[:3]}")
    if all_tokens:
        print(f"  Token sample: {list(all_tokens)[:3]}")
    
    return {
        "pkl_files": all_pkls,
        "total_frames": total_frames,
        "tokens": all_tokens,
        "log_names": all_log_names,
        "sample_frame": frame_sample,
    }


# ============================================================
# 2. INSPECT SENSOR_BLOBS
# ============================================================
def inspect_sensor_blobs():
    sep("2. SENSOR_BLOBS (cameras + LiDAR)")
    
    sensor_root = os.path.join(DATA_ROOT, "sensor_blobs")
    if not os.path.isdir(sensor_root):
        print(f"  NOT FOUND: {sensor_root}")
        # Check if sensor_blobs is at a different level
        alt = os.path.join(NAVSIM_ROOT, "download", "sensor_blobs")
        if os.path.isdir(alt):
            sensor_root = alt
            print(f"  Found at: {sensor_root}")
        else:
            return None
    
    print(f"  Path: {sensor_root}")
    splits = sorted([d for d in os.listdir(sensor_root) if os.path.isdir(os.path.join(sensor_root, d))])
    print(f"  Splits: {splits}")
    
    all_sensor_info = {}
    
    for split in splits:
        split_path = os.path.join(sensor_root, split)
        log_dirs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        print(f"\n  --- {split}/ ({len(log_dirs)} logs) ---")
        
        total_cam_frames = 0
        total_lidar_frames = 0
        cam_tokens = set()
        lidar_tokens = set()
        
        for log_dir in log_dirs[:3]:  # Sample first 3 logs
            log_path = os.path.join(split_path, log_dir)
            subdirs = sorted(os.listdir(log_path))
            
            # Count CAM_F0 images
            cam_f0 = os.path.join(log_path, "CAM_F0")
            if os.path.isdir(cam_f0):
                imgs = sorted([f for f in os.listdir(cam_f0) if f.endswith(('.jpg', '.png'))])
                total_cam_frames += len(imgs)
                for img in imgs:
                    cam_tokens.add(Path(img).stem)
                print(f"    {log_dir}: CAM_F0={len(imgs)} imgs, subdirs={subdirs}")
            
            # Count LiDAR
            lidar_dir = os.path.join(log_path, "MergedPointCloud")
            if os.path.isdir(lidar_dir):
                pcds = sorted(os.listdir(lidar_dir))
                total_lidar_frames += len(pcds)
                for pcd in pcds:
                    lidar_tokens.add(Path(pcd).stem)
                ext = pcds[0].split('.')[-1] if pcds else "?"
                print(f"    {log_dir}: MergedPointCloud={len(pcds)} files (.{ext})")
        
        # Count ALL logs
        for log_dir in log_dirs:
            log_path = os.path.join(split_path, log_dir)
            cam_f0 = os.path.join(log_path, "CAM_F0")
            if os.path.isdir(cam_f0):
                imgs = [f for f in os.listdir(cam_f0) if f.endswith(('.jpg', '.png'))]
                total_cam_frames += len(imgs)
                for img in imgs:
                    cam_tokens.add(Path(img).stem)
            lidar_dir = os.path.join(log_path, "MergedPointCloud")
            if os.path.isdir(lidar_dir):
                pcds = os.listdir(lidar_dir)
                total_lidar_frames += len(pcds)
                for pcd in pcds:
                    lidar_tokens.add(Path(pcd).stem)
        
        # Avoid double-counting the first 3 logs
        # Actually let's just do a clean count
        total_cam_frames = 0
        total_lidar_frames = 0
        cam_tokens = set()
        lidar_tokens = set()
        
        for log_dir in log_dirs:
            log_path = os.path.join(split_path, log_dir)
            cam_f0 = os.path.join(log_path, "CAM_F0")
            if os.path.isdir(cam_f0):
                imgs = [f for f in os.listdir(cam_f0) if f.endswith(('.jpg', '.png'))]
                total_cam_frames += len(imgs)
                for img in imgs:
                    cam_tokens.add(Path(img).stem)
            lidar_dir = os.path.join(log_path, "MergedPointCloud")
            if os.path.isdir(lidar_dir):
                pcds = os.listdir(lidar_dir)
                total_lidar_frames += len(pcds)
                for pcd in pcds:
                    lidar_tokens.add(Path(pcd).stem)
        
        print(f"\n  {split} totals:")
        print(f"    Camera frames (CAM_F0): {total_cam_frames}")
        print(f"    LiDAR frames: {total_lidar_frames}")
        print(f"    Unique cam tokens: {len(cam_tokens)}")
        print(f"    Unique lidar tokens: {len(lidar_tokens)}")
        
        all_sensor_info[split] = {
            "log_dirs": log_dirs,
            "cam_tokens": cam_tokens,
            "lidar_tokens": lidar_tokens,
        }
    
    return all_sensor_info


# ============================================================
# 3. INSPECT OCCUPANCY LABELS
# ============================================================
def inspect_occ_labels():
    sep("3. OCCUPANCY LABELS")
    
    if not os.path.isdir(OCC_LABEL_ROOT):
        print(f"  NOT FOUND: {OCC_LABEL_ROOT}")
        return None
    
    print(f"  Path: {OCC_LABEL_ROOT}")
    
    # Check structure
    contents = sorted(os.listdir(OCC_LABEL_ROOT))
    dirs = [c for c in contents if os.path.isdir(os.path.join(OCC_LABEL_ROOT, c))]
    files = [c for c in contents if os.path.isfile(os.path.join(OCC_LABEL_ROOT, c))]
    
    print(f"  Top-level dirs: {len(dirs)}")
    print(f"  Top-level files: {len(files)}")
    if dirs:
        print(f"  Dir samples: {dirs[:5]}")
    if files:
        print(f"  File samples: {files[:5]}")
    
    # Find all label files
    npz_files = sorted(glob.glob(os.path.join(OCC_LABEL_ROOT, "**", "*.npz"), recursive=True))
    npy_files = sorted(glob.glob(os.path.join(OCC_LABEL_ROOT, "**", "*.npy"), recursive=True))
    
    print(f"\n  Total .npz files (recursive): {len(npz_files)}")
    print(f"  Total .npy files (recursive): {len(npy_files)}")
    
    occ_tokens = set()
    occ_paths = {}  # token -> path
    
    # Determine structure
    if npz_files:
        label_files = npz_files
        ext = ".npz"
    elif npy_files:
        label_files = npy_files
        ext = ".npy"
    else:
        # Check deeper
        all_files = sorted(glob.glob(os.path.join(OCC_LABEL_ROOT, "**", "*"), recursive=True))
        file_exts = Counter(Path(f).suffix for f in all_files if os.path.isfile(f))
        print(f"  All file extensions: {dict(file_exts)}")
        
        # Check if there's a different structure (e.g., occ_gt folders)
        if dirs:
            first_dir = os.path.join(OCC_LABEL_ROOT, dirs[0])
            print(f"  Inspecting {dirs[0]}/:")
            sub_contents = sorted(os.listdir(first_dir))
            print(f"    Contents: {sub_contents[:10]}")
            if sub_contents:
                deeper = os.path.join(first_dir, sub_contents[0])
                if os.path.isdir(deeper):
                    deep_contents = sorted(os.listdir(deeper))
                    print(f"    {sub_contents[0]}/: {deep_contents[:10]}")
        return None
    
    # Map tokens to paths
    for f in label_files:
        token = Path(f).stem
        # Remove suffixes like _occ_final, _occ_gt etc.
        clean_token = token
        for suffix in ["_occ_final", "_occ_gt", "_occ"]:
            if clean_token.endswith(suffix):
                clean_token = clean_token[:-len(suffix)]
        occ_tokens.add(clean_token)
        occ_tokens.add(token)  # Also add raw filename stem
        occ_paths[token] = f
    
    print(f"  Unique occ tokens: {len(occ_tokens)}")
    if label_files:
        print(f"  Sample paths:")
        for f in label_files[:3]:
            rel = os.path.relpath(f, OCC_LABEL_ROOT)
            print(f"    {rel}")
    
    # Load one and check shape
    if label_files:
        sample_file = label_files[0]
        print(f"\n  --- Inspecting sample: {os.path.basename(sample_file)} ---")
        if ext == ".npz":
            data = np.load(sample_file, allow_pickle=True)
            print(f"  Keys: {list(data.keys())}")
            for k in data.keys():
                arr = data[k]
                print(f"    {k}: shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}")
        elif ext == ".npy":
            arr = np.load(sample_file, allow_pickle=True)
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
    
    return {
        "tokens": occ_tokens,
        "paths": occ_paths,
        "n_files": len(label_files),
    }


# ============================================================
# 4. INSPECT MAPS
# ============================================================
def inspect_maps():
    sep("4. MAPS")
    
    if not os.path.isdir(MAPS_ROOT):
        print(f"  NOT FOUND: {MAPS_ROOT}")
        return None
    
    print(f"  Path: {MAPS_ROOT}")
    contents = sorted(os.listdir(MAPS_ROOT))
    print(f"  Contents ({len(contents)}): {contents[:10]}")
    
    for item in contents[:3]:
        item_path = os.path.join(MAPS_ROOT, item)
        if os.path.isdir(item_path):
            sub = os.listdir(item_path)
            print(f"    {item}/: {len(sub)} items ({sub[:5]})")
    
    return True


# ============================================================
# 5. INSPECT NAVSIM LOGS
# ============================================================
def inspect_navsim_logs():
    sep("5. NAVSIM LOGS")
    
    log_candidates = [
        os.path.join(NAVSIM_ROOT, "navsim_logs"),
        os.path.join(NAVSIM_ROOT, "download", "navsim_logs"),
        os.path.join(DATA_ROOT, "navsim_logs"),
    ]
    
    log_root = None
    for candidate in log_candidates:
        if os.path.isdir(candidate):
            log_root = candidate
            break
    
    if log_root is None:
        print(f"  NOT FOUND (checked: {log_candidates})")
        return None
    
    print(f"  Path: {log_root}")
    splits = sorted(os.listdir(log_root))
    print(f"  Contents: {splits}")
    
    for sp in splits:
        sp_path = os.path.join(log_root, sp)
        if os.path.isdir(sp_path):
            files = sorted(os.listdir(sp_path))
            pkl_count = sum(1 for f in files if f.endswith('.pkl'))
            db_count = sum(1 for f in files if f.endswith('.db'))
            print(f"    {sp}/: {len(files)} files ({pkl_count} pkl, {db_count} db)")
            if files:
                print(f"      Sample: {files[:3]}")
    
    return log_root


# ============================================================
# 6. CROSS-REFERENCE: How do tokens connect?
# ============================================================
def cross_reference(meta_info, sensor_info, occ_info):
    sep("6. CROSS-REFERENCE: How data sources connect")
    
    if meta_info is None:
        print("  No meta_data to cross-reference")
        return
    
    meta_tokens = meta_info["tokens"]
    
    # Check: meta tokens vs sensor tokens
    if sensor_info:
        for split, sinfo in sensor_info.items():
            cam_tokens = sinfo["cam_tokens"]
            lidar_tokens = sinfo["lidar_tokens"]
            
            meta_in_cam = meta_tokens & cam_tokens
            meta_in_lidar = meta_tokens & lidar_tokens
            cam_not_in_meta = cam_tokens - meta_tokens
            meta_not_in_cam = meta_tokens - cam_tokens
            
            print(f"\n  Meta ↔ Sensor ({split}):")
            print(f"    Meta tokens in camera data: {len(meta_in_cam)} / {len(meta_tokens)}")
            print(f"    Meta tokens in LiDAR data: {len(meta_in_lidar)} / {len(meta_tokens)}")
            print(f"    Camera tokens NOT in meta: {len(cam_not_in_meta)}")
            print(f"    Meta tokens NOT in camera: {len(meta_not_in_cam)}")
            
            if meta_not_in_cam and len(meta_not_in_cam) < 10:
                print(f"      Missing: {meta_not_in_cam}")
    
    # Check: meta tokens vs occ tokens
    if occ_info:
        occ_tokens = occ_info["tokens"]
        meta_in_occ = meta_tokens & occ_tokens
        occ_not_in_meta = occ_tokens - meta_tokens
        meta_not_in_occ = meta_tokens - occ_tokens
        
        print(f"\n  Meta ↔ Occupancy:")
        print(f"    Meta tokens with occ labels: {len(meta_in_occ)} / {len(meta_tokens)}")
        print(f"    Occ tokens NOT in meta: {len(occ_not_in_meta)}")
        print(f"    Meta tokens WITHOUT occ: {len(meta_not_in_occ)}")
        
        # Try to understand the token format difference
        if meta_in_occ == set() and len(meta_tokens) > 0 and len(occ_tokens) > 0:
            print(f"\n    *** NO OVERLAP — checking token formats ***")
            meta_samples = list(meta_tokens)[:5]
            occ_samples = list(occ_tokens)[:5]
            print(f"    Meta token samples: {meta_samples}")
            print(f"    Occ token samples: {occ_samples}")
            
            # Check if occ uses a different naming (e.g., with log prefix)
            # Check if there's a path-based connection
            if occ_info["paths"]:
                sample_paths = list(occ_info["paths"].values())[:3]
                print(f"    Occ file path samples:")
                for p in sample_paths:
                    print(f"      {os.path.relpath(p, OCC_LABEL_ROOT)}")
    
    # Check: sample frame paths vs actual files
    if meta_info.get("sample_frame"):
        frame = meta_info["sample_frame"]
        print(f"\n  --- Verifying sample frame paths ---")
        print(f"  Token: {frame.get('token', '?')}")
        
        # Check lidar path
        lidar_path = frame.get("lidar_path", "")
        if lidar_path:
            full_lidar = os.path.join(DATA_ROOT, "sensor_blobs", lidar_path)
            exists = os.path.exists(full_lidar)
            print(f"  LiDAR: {lidar_path}")
            print(f"    Full path: {full_lidar}")
            print(f"    EXISTS: {exists}")
            if not exists:
                # Try alternative paths
                for alt_base in [DATA_ROOT, os.path.join(DATA_ROOT, "sensor_blobs", "mini")]:
                    alt_path = os.path.join(alt_base, lidar_path)
                    if os.path.exists(alt_path):
                        print(f"    Found at: {alt_path}")
                        break
        
        # Check camera paths
        cams = frame.get("cams", {})
        for cam_name in ["CAM_F0"]:  # Just check one
            cam_data = cams.get(cam_name, {})
            cam_path = cam_data.get("data_path", "")
            if cam_path:
                full_cam = os.path.join(DATA_ROOT, "sensor_blobs", cam_path)
                exists = os.path.exists(full_cam)
                print(f"  {cam_name}: {cam_path}")
                print(f"    Full path: {full_cam}")
                print(f"    EXISTS: {exists}")
                if not exists:
                    for alt_base in [DATA_ROOT, os.path.join(DATA_ROOT, "sensor_blobs", "mini")]:
                        alt_path = os.path.join(alt_base, cam_path)
                        if os.path.exists(alt_path):
                            print(f"    Found at: {alt_path}")
                            break


# ============================================================
# 7. SUMMARY & RECOMMENDATIONS
# ============================================================
def summary(meta_info, sensor_info, occ_info):
    sep("7. SUMMARY & NEXT STEPS")
    
    if meta_info:
        print(f"  Meta: {meta_info['total_frames']} frames from {len(meta_info['pkl_files'])} pkl files")
    else:
        print(f"  Meta: NOT AVAILABLE")
    
    if sensor_info:
        for split, sinfo in sensor_info.items():
            print(f"  Sensors ({split}): {len(sinfo['log_dirs'])} logs, {len(sinfo['cam_tokens'])} cam frames, {len(sinfo['lidar_tokens'])} lidar frames")
    else:
        print(f"  Sensors: NOT AVAILABLE")
    
    if occ_info:
        print(f"  Occupancy: {occ_info['n_files']} label files, {len(occ_info['tokens'])} tokens")
    else:
        print(f"  Occupancy: NOT AVAILABLE")
    
    print(f"\n  For Phase 1 training you need:")
    print(f"    [{'OK' if meta_info else 'MISSING'}] Meta-data (frame index with calibration)")
    print(f"    [{'OK' if sensor_info else 'MISSING'}] Camera images + LiDAR point clouds")
    print(f"    [{'OK' if occ_info else 'MISSING'}] Occupancy ground truth labels")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Dataset Inspector — Checking all data sources")
    print(f"Date: {__import__('datetime').datetime.now()}")
    
    meta_info = inspect_metadata()
    sensor_info = inspect_sensor_blobs()
    occ_info = inspect_occ_labels()
    inspect_maps()
    inspect_navsim_logs()
    cross_reference(meta_info, sensor_info, occ_info)
    summary(meta_info, sensor_info, occ_info)