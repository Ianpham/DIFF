#!/usr/bin/env python3
"""
Step-by-step PCD File Diagnostics
Find and test actual PCD files in your dataset
"""

from pathlib import Path
import os

print("="*70)
print("Step 1: Find PCD Files")
print("="*70)

# Your data root
data_root = Path("/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download")
sensor_blob_path = data_root / "mini_sensor_blobs" / "mini"

print(f"\nSearching in: {sensor_blob_path}")
print(f"Path exists: {sensor_blob_path.exists()}")

if not sensor_blob_path.exists():
    print(f"ERROR: Path doesn't exist!")
    exit(1)

# Find all date folders
date_folders = sorted([d for d in sensor_blob_path.iterdir() if d.is_dir()])
print(f"\nFound {len(date_folders)} date folders")

# List first 5 date folders
print("\nFirst 5 date folders:")
for i, folder in enumerate(date_folders[:5]):
    print(f"  {i+1}. {folder.name}")

# Check one date folder for MergedPointCloud
if date_folders:
    sample_folder = date_folders[0]
    print(f"\n{'='*70}")
    print(f"Step 2: Inspect First Date Folder")
    print("="*70)
    print(f"Folder: {sample_folder.name}")
    
    # List all subfolders
    subfolders = [d.name for d in sample_folder.iterdir() if d.is_dir()]
    print(f"\nSubfolders found: {len(subfolders)}")
    for subfolder in sorted(subfolders):
        print(f"  - {subfolder}")
    
    # Check for MergedPointCloud
    merged_pc_path = sample_folder / "MergedPointCloud"
    print(f"\nMergedPointCloud path: {merged_pc_path}")
    print(f"Exists: {merged_pc_path.exists()}")
    
    if merged_pc_path.exists():
        # List PCD files
        pcd_files = list(merged_pc_path.glob("*.pcd"))
        print(f"\nFound {len(pcd_files)} .pcd files")
        
        if pcd_files:
            print("\nFirst 5 PCD files:")
            for i, pcd_file in enumerate(pcd_files[:5]):
                file_size = pcd_file.stat().st_size
                print(f"  {i+1}. {pcd_file.name}")
                print(f"      Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
                print(f"      Full path: {pcd_file}")
            
            # Test loading first PCD file
            print(f"\n{'='*70}")
            print(f"Step 3: Test Loading First PCD File")
            print("="*70)
            
            test_file = pcd_files[0]
            print(f"\nTesting: {test_file.name}")
            print(f"Full path: {test_file}")
            print(f"File exists: {test_file.exists()}")
            print(f"File size: {test_file.stat().st_size:,} bytes")
            
            # Check file permissions
            print(f"Readable: {os.access(test_file, os.R_OK)}")
            
            # Try to read first few bytes
            print("\n--- First 500 bytes of file ---")
            try:
                with open(test_file, 'rb') as f:
                    first_bytes = f.read(500)
                    # Try to decode as ASCII
                    try:
                        text = first_bytes.decode('ascii', errors='replace')
                        print(text)
                    except:
                        print("Binary data (non-ASCII)")
                        print(f"Hex preview: {first_bytes[:100].hex()}")
            except Exception as e:
                print(f"ERROR reading file: {e}")
            
            # Try open3d
            print(f"\n{'='*70}")
            print(f"Step 4: Test open3d Loading")
            print("="*70)
            
            try:
                import open3d as o3d
                import numpy as np
                
                print(f"\nopen3d version: {o3d.__version__}")
                print(f"Attempting to load: {test_file}")
                
                pcd = o3d.io.read_point_cloud(str(test_file))
                points = np.asarray(pcd.points)
                
                print(f"\n✓ SUCCESS!")
                print(f"Loaded {len(points)} points")
                print(f"Point cloud shape: {points.shape}")
                
                if len(points) > 0:
                    print(f"\nFirst 5 points:")
                    print(points[:5])
                    print(f"\nPoint statistics:")
                    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
                    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                else:
                    print("⚠ WARNING: Point cloud is empty!")
                
            except Exception as e:
                print(f"\n✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
            
            # Try manual binary parsing
            print(f"\n{'='*70}")
            print(f"Step 5: Test Manual Binary Parsing")
            print("="*70)
            
            try:
                with open(test_file, 'rb') as f:
                    # Read and parse header
                    header_lines = []
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        try:
                            line_str = line.decode('ascii', errors='ignore').strip()
                            header_lines.append(line_str)
                            if line_str.startswith('DATA'):
                                break
                        except:
                            break
                    
                    print("\nPCD Header:")
                    for line in header_lines[:15]:  # Print first 15 lines
                        print(f"  {line}")
                    
                    # Read binary data
                    binary_data = f.read()
                    print(f"\nBinary data size: {len(binary_data):,} bytes")
                    
                    # Try to parse as float32
                    points = np.frombuffer(binary_data, dtype=np.float32)
                    print(f"As float32 array: {len(points)} values")
                    
                    # Try reshaping to (N, 4) - typical format is x,y,z,intensity
                    if len(points) % 4 == 0:
                        points = points.reshape(-1, 4)
                        xyz = points[:, :3]  # Take only x,y,z
                        
                        # Filter valid points
                        valid = ~np.isnan(xyz).any(axis=1)
                        xyz = xyz[valid]
                        
                        print(f"\n✓ SUCCESS!")
                        print(f"Parsed {len(xyz)} valid points")
                        print(f"Shape: {xyz.shape}")
                        
                        if len(xyz) > 0:
                            print(f"\nFirst 5 points:")
                            print(xyz[:5])
                            print(f"\nPoint statistics:")
                            print(f"  X range: [{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
                            print(f"  Y range: [{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
                            print(f"  Z range: [{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
                    else:
                        print(f"⚠ WARNING: Data size doesn't divide evenly by 4")
                        print(f"Remainder: {len(points) % 4}")
                        
            except Exception as e:
                print(f"\n✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        else:
            print("\n✗ No .pcd files found in MergedPointCloud folder!")
    else:
        print("\n✗ MergedPointCloud folder doesn't exist!")
        print("\nAvailable folders:")
        for subfolder in sorted(subfolders):
            print(f"  - {subfolder}")
else:
    print("\n✗ No date folders found!")

print(f"\n{'='*70}")
print("Diagnostics Complete")
print("="*70)