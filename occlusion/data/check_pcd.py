#!/usr/bin/env python3
"""Quick check: what's inside the PCD files from NavSim?"""
import os
import numpy as np

SENSOR_ROOT = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/download/mini_sensor_blobs/mini"

# Find first PCD file
first_log = sorted(os.listdir(SENSOR_ROOT))[0]
lidar_dir = os.path.join(SENSOR_ROOT, first_log, "MergedPointCloud")
first_pcd = sorted(os.listdir(lidar_dir))[0]
pcd_path = os.path.join(lidar_dir, first_pcd)

print(f"File: {pcd_path}")
print(f"Size: {os.path.getsize(pcd_path)} bytes")

# Read header
with open(pcd_path, "rb") as f:
    header_lines = []
    while True:
        line = f.readline()
        decoded = line.decode("utf-8", errors="ignore").strip()
        header_lines.append(decoded)
        if decoded.startswith("DATA"):
            break
    
    header_end_pos = f.tell()
    remaining_bytes = os.path.getsize(pcd_path) - header_end_pos
    
    # Read first 100 bytes of data
    sample_data = f.read(min(200, remaining_bytes))

print("\n--- HEADER ---")
for line in header_lines:
    print(f"  {line}")

print(f"\nHeader ends at byte: {header_end_pos}")
print(f"Remaining data bytes: {remaining_bytes}")

# Parse header
header = {}
for line in header_lines:
    parts = line.split(None, 1)
    if len(parts) == 2:
        header[parts[0]] = parts[1]

fields = header.get("FIELDS", "").split()
sizes = header.get("SIZE", "").split()
types = header.get("TYPE", "").split()
counts = header.get("COUNT", "").split()
n_points = int(header.get("POINTS", 0))
data_type = header_lines[-1].split()[-1]

print(f"\nFields: {fields}")
print(f"Sizes:  {sizes}")
print(f"Types:  {types}")
print(f"Counts: {counts}")
print(f"Points: {n_points}")
print(f"Data:   {data_type}")

# Calculate expected bytes per point
bytes_per_point = sum(int(s) * int(c) for s, c in zip(sizes, counts))
expected_total = bytes_per_point * n_points
print(f"\nBytes per point: {bytes_per_point}")
print(f"Expected data bytes: {expected_total}")
print(f"Actual data bytes:   {remaining_bytes}")
print(f"Ratio: {remaining_bytes / expected_total if expected_total > 0 else 'N/A':.3f}")

if data_type == "binary_compressed":
    import struct
    with open(pcd_path, "rb") as f:
        f.seek(header_end_pos)
        compressed_size = struct.unpack("I", f.read(4))[0]
        uncompressed_size = struct.unpack("I", f.read(4))[0]
        print(f"\nCompressed size: {compressed_size}")
        print(f"Uncompressed size: {uncompressed_size}")
        print(f"Expected uncompressed: {expected_total}")
elif data_type == "binary":
    print(f"\nFirst bytes of data (hex): {sample_data[:50].hex()}")
    # Try reading as float32
    if remaining_bytes >= bytes_per_point:
        first_point_bytes = sample_data[:bytes_per_point]
        vals = []
        offset = 0
        for sz, tp, fn in zip(sizes, types, fields):
            sz = int(sz)
            if tp == "F" and sz == 4:
                vals.append((fn, np.frombuffer(first_point_bytes[offset:offset+sz], dtype=np.float32)[0]))
            elif tp == "U" and sz == 4:
                vals.append((fn, np.frombuffer(first_point_bytes[offset:offset+sz], dtype=np.uint32)[0]))
            elif tp == "U" and sz == 1:
                vals.append((fn, first_point_bytes[offset]))
            offset += sz
        print(f"First point: {vals}")