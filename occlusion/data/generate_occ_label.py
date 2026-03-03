"""
Occupancy Label Generation Pipeline.

Generates 200×200×16 semantic occupancy labels from:
  - Multi-sweep LiDAR point clouds (accumulated via ego poses)
  - 3D bounding box annotations (dynamic objects)
  - HD map polygons (static surfaces)

This replicates the approach used by OpenScene to create occupancy
ground truth, adapted for arbitrary sensor configurations (any number
of cameras/LiDARs).

Usage:
    python -m data.generate_occ_labels \
        --data-root /data/new_dataset/ \
        --output-dir /data/new_dataset/occupancy/ \
        --split train \
        --num-workers 8

Pipeline per sample:
    1. Accumulate N seconds of LiDAR sweeps in ego reference frame
    2. Voxelize the accumulated point cloud → binary occupied mask
    3. Assign semantics from 3D bounding boxes (dynamic objects)
    4. Assign semantics from HD map (static surfaces: road, sidewalk, etc.)
    5. Ray-trace from sensor origin → mark free/observed voxels
    6. Save as compressed .npz: occ_label (X,Y,Z) + visibility (X,Y,Z)
"""

import os
import argparse
import numpy as np

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import json

######################## Configuration ##############################

@dataclass
class OccLabelConfig:
    """Configuration for occupancy label generation"""

    # voxel grid geometry
    point_cloud_range: List[float] = field(
        default_factory = lambda : [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    )

    occ_size: List[int] = field(default_factory=lambda : [200, 200, 16])

    # multi-sweep accumulation
    accumulation_seconds: float = 20.0
    lidar_freq_hz: float = 10.0
    max_sweeps: int = 200

    # OpenScene 17-class taxonomy:
    #   0: empty       1: barrier       2: bicycle       3: bus
    #   4: car         5: construction  6: motorcycle    7: pedestrian
    #   8: traffic_cone 9: trailer     10: truck
    #  11: driveable   12: other_flat  13: sidewalk
    #  14: terrain     15: manmade     16: vegetatio

    num_classes: int = 17
    empty_label: int = 0

    # new dataset's box classes -> openscene occupancy index
    # modify mapping for new dataset
    box_class_mapping: Dict[str, int] = field(default_factory= lambda: {
        "car": 4, "truck": 10, "bus": 3, "trailer": 9,
        "construction_vehicle": 5, "motorcycle": 6, "bicycle": 2,
        "pedestrian": 7, "traffic_cone": 8, "barrier": 1,
    })

    # new dataset map layer names -> Openscence occupancy index
    # modify this mapping for new dataset
    # this is problem with road surface, that is when we can add more sub class to identify this.
    map_class_mapping: Dict[str, int] = field(default_factory= lambda : {
        "driveable_area": 11, "road": 11, "lane": 11,
        "crosswalk": 12, "sidewalk": 13, "walkway": 13,
        "terrain": 14, "grass": 14,
        "building": 15, "wall": 15, "fence": 15,
        "vegetation": 16, "tree": 16,      
    })

    # ray tracing
    min_points_per_voxel: int = 1
    max_ray_trace_points: int = 30000

    # ground-level z-range for map polygon assignment
    ground_z_min: float = -1.0
    ground_z_max: float = 0.5

# coordinate utilities

def compute_voxel_size(pc_range, occ_size):
    """ Derive voxel size from range and grid dimensions."""
    pc = np.array(pc_range, dtype = np.float32)
    return np.array([
        (pc[3] - pc[1])/occ_size[0],
        (pc[4] - pc[1])/occ_size[1],
        (pc[5] - pc[2])/occ_size[2],
    ], dtype = np.float32)


