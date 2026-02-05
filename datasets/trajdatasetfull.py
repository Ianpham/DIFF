"""
Complete implementation of build_trajectory_dataset()
This can be used as a placeholder or actual implementation
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from typing import Dict, List, Optional


def build_trajectory_dataset(args):
    """
    Build trajectory dataset from args.
    
    Args:
        args: Argparse namespace with the following attributes:
            - data_path: Path to dataset directory
            - traj_len: Number of future waypoints
            - traj_dim: Trajectory dimension (x, y, vel_x, vel_y, heading)
            - ego_dim: Ego state dimension
            - modality_names: List of modality names to load
    
    Returns:
        TrajectoryDataset instance
    """
    
    # Extract arguments with defaults
    data_path = args.data_path
    traj_len = getattr(args, 'traj_len', 10)
    traj_dim = getattr(args, 'traj_dim', 5)
    ego_dim = getattr(args, 'ego_dim', 8)
    modality_names = getattr(args, 'modality_names', ['lidar', 'img', 'BEV'])
    
    # Check if data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data path {data_path} does not exist. "
            f"Please create it or generate dummy data."
        )
    
    # Create dataset
    dataset = TrajectoryDataset(
        data_path=data_path,
        traj_len=traj_len,
        traj_dim=traj_dim,
        ego_dim=ego_dim,
        modality_names=modality_names,
        normalize_trajectory=True
    )
    
    return dataset


class TrajectoryDataset(Dataset):
    """
    Dataset for multi-modal trajectory prediction.
    
    Expected directory structure:
    data_path/
        scene_0001/
            lidar.npy          # (2, H, W) or (T, 2, H, W)
            img.npy            # (3, H, W) or (T, 3, H, W)
            BEV.npy            # (7, H, W) or (T, 7, H, W)
            ego_state.npy      # (ego_dim,) or (T, ego_dim)
            action_history.npy # (history_len, action_dim)
            trajectory.npy     # (traj_len, traj_dim) - GROUND TRUTH
            metadata.json      # Optional metadata
        scene_0002/
            ...
    """
    
    def __init__(
        self,
        data_path: str,
        traj_len: int = 10,
        traj_dim: int = 5,
        ego_dim: int = 8,
        modality_names: List[str] = ['lidar', 'img', 'BEV'],
        normalize_trajectory: bool = True,
        transform=None
    ):
        self.data_path = data_path
        self.traj_len = traj_len
        self.traj_dim = traj_dim
        self.ego_dim = ego_dim
        self.modality_names = modality_names
        self.transform = transform
        self.normalize_trajectory = normalize_trajectory
        
        # Get all scene directories
        self.scene_dirs = self._get_scene_directories()
        
        if len(self.scene_dirs) == 0:
            raise ValueError(
                f"No scenes found in {data_path}. "
                f"Please check your data path or generate dummy data."
            )
        
        print(f"[TrajectoryDataset] Found {len(self.scene_dirs)} scenes")
        
        # Compute trajectory statistics for normalization
        if self.normalize_trajectory:
            self._compute_trajectory_stats()
    
    def _get_scene_directories(self):
        """Get all valid scene directories."""
        scene_dirs = []
        
        for item in os.listdir(self.data_path):
            scene_path = os.path.join(self.data_path, item)
            
            # Check if it's a directory
            if not os.path.isdir(scene_path):
                continue
            
            # Check if it has trajectory.npy (required)
            traj_file = os.path.join(scene_path, 'trajectory.npy')
            if not os.path.exists(traj_file):
                continue
            
            scene_dirs.append(scene_path)
        
        return sorted(scene_dirs)
    
    def _compute_trajectory_stats(self):
        """Compute mean and std for trajectory normalization."""
        print("[TrajectoryDataset] Computing trajectory statistics...")
        
        # Sample a subset to compute statistics
        sample_size = min(1000, len(self.scene_dirs))
        trajectories = []
        
        for i in range(sample_size):
            traj_path = os.path.join(self.scene_dirs[i], 'trajectory.npy')
            try:
                traj = np.load(traj_path)
                trajectories.append(traj)
            except Exception as e:
                print(f"Warning: Failed to load {traj_path}: {e}")
                continue
        
        if len(trajectories) == 0:
            print("Warning: No trajectories loaded for statistics. Using default normalization.")
            self.traj_mean = np.zeros(self.traj_dim)
            self.traj_std = np.ones(self.traj_dim)
            return
        
        # Stack and compute statistics
        trajectories = np.concatenate(trajectories, axis=0)
        self.traj_mean = trajectories.mean(axis=0).astype(np.float32)
        self.traj_std = (trajectories.std(axis=0) + 1e-8).astype(np.float32)
        
        print(f"  - Trajectory mean: {self.traj_mean}")
        print(f"  - Trajectory std: {self.traj_std}")
    
    def __len__(self):
        return len(self.scene_dirs)
    
    def __getitem__(self, idx):
        """
        Load one scene's data.
        
        Returns:
            batch: Dict with keys:
                - 'lidar': (2, H, W) tensor
                - 'img': (3, H, W) tensor
                - 'BEV': (7, H, W) tensor
                - 'ego_state': (ego_dim,) tensor
                - 'action_history': (history_len, action_dim) tensor
                - 'future_trajectory': (traj_len, traj_dim) tensor
                - 'metadata': dict (optional)
        """
        scene_dir = self.scene_dirs[idx]
        batch = {}
        
        # Load each modality
        for modality in self.modality_names:
            modality_path = os.path.join(scene_dir, f'{modality}.npy')
            
            if os.path.exists(modality_path):
                try:
                    data = np.load(modality_path)
                    
                    # Handle temporal dimension
                    if len(data.shape) == 4:  # (T, C, H, W)
                        data = data[-1]  # Take last frame (current)
                    
                    batch[modality] = torch.from_numpy(data).float()
                except Exception as e:
                    print(f"Warning: Failed to load {modality_path}: {e}")
                    # Create dummy data as fallback
                    if modality == 'lidar':
                        batch[modality] = torch.zeros(2, 64, 64)
                    elif modality == 'img':
                        batch[modality] = torch.zeros(3, 64, 64)
                    elif modality == 'BEV':
                        batch[modality] = torch.zeros(7, 64, 64)
        
        # Load ego state
        ego_path = os.path.join(scene_dir, 'ego_state.npy')
        if os.path.exists(ego_path):
            try:
                ego_state = np.load(ego_path)
                
                # Handle temporal dimension
                if len(ego_state.shape) == 2:  # (T, ego_dim)
                    ego_state = ego_state[-1]  # Current state
                
                batch['ego_state'] = torch.from_numpy(ego_state).float()
            except Exception as e:
                print(f"Warning: Failed to load ego state: {e}")
                batch['ego_state'] = torch.zeros(self.ego_dim)
        else:
            batch['ego_state'] = torch.zeros(self.ego_dim)
        
        # Load action history
        action_path = os.path.join(scene_dir, 'action_history.npy')
        if os.path.exists(action_path):
            try:
                action_history = np.load(action_path)
                batch['action_history'] = torch.from_numpy(action_history).float()
            except Exception as e:
                print(f"Warning: Failed to load action history: {e}")
                batch['action_history'] = torch.zeros(10, 4)  # Default size
        else:
            batch['action_history'] = torch.zeros(10, 4)
        
        # Load future trajectory (REQUIRED)
        traj_path = os.path.join(scene_dir, 'trajectory.npy')
        try:
            trajectory = np.load(traj_path).astype(np.float32)
            
            # Normalize trajectory
            if self.normalize_trajectory:
                trajectory = (trajectory - self.traj_mean) / self.traj_std
            
            batch['future_trajectory'] = torch.from_numpy(trajectory).float()
        except Exception as e:
            print(f"Error: Failed to load trajectory {traj_path}: {e}")
            # Create dummy trajectory
            trajectory = np.zeros((self.traj_len, self.traj_dim), dtype=np.float32)
            batch['future_trajectory'] = torch.from_numpy(trajectory).float()
        
        # Load metadata (optional)
        metadata_path = os.path.join(scene_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                batch['metadata'] = metadata
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")
        
        # Apply transforms if provided
        if self.transform is not None:
            batch = self.transform(batch)
        
        return batch


# ============================================================================
# UTILITY FUNCTIONS FOR DATA GENERATION
# ============================================================================

def generate_dummy_trajectory_data(output_path, num_scenes=100, H=64, W=64):
    """
    Generate dummy trajectory data for testing.
    
    Args:
        output_path: Where to save the data
        num_scenes: Number of scenes to generate
        H, W: Height and width of spatial data
    """
    print(f"Generating {num_scenes} dummy scenes at {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    for i in range(num_scenes):
        scene_dir = os.path.join(output_path, f'scene_{i:04d}')
        os.makedirs(scene_dir, exist_ok=True)
        
        # LiDAR: (2, H, W)
        lidar = np.random.randn(2, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'lidar.npy'), lidar)
        
        # Image: (3, H, W)
        img = np.random.randn(3, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'img.npy'), img)
        
        # BEV: (7, H, W)
        bev = np.random.randn(7, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'BEV.npy'), bev)
        
        # Ego state: (8,)
        ego_state = np.random.randn(8).astype(np.float32)
        np.save(os.path.join(scene_dir, 'ego_state.npy'), ego_state)
        
        # Action history: (10, 4)
        action_history = np.random.randn(10, 4).astype(np.float32)
        np.save(os.path.join(scene_dir, 'action_history.npy'), action_history)
        
        # Future trajectory: (10, 5) - (x, y, vel_x, vel_y, heading)
        trajectory = np.random.randn(10, 5).astype(np.float32)
        np.save(os.path.join(scene_dir, 'trajectory.npy'), trajectory)
        
        # Metadata
        metadata = {
            'scene_id': f'scene_{i:04d}',
            'weather': 'clear',
            'time_of_day': 'day',
            'location': 'test'
        }
        with open(os.path.join(scene_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    print(f"✓ Generated {num_scenes} scenes successfully!")
    print(f"  Data saved to: {output_path}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the dataset implementation."""
    
    print("=" * 80)
    print("Testing TrajectoryDataset")
    print("=" * 80)
    
    # Step 1: Generate dummy data
    dummy_path = '/tmp/test_trajectory_dataset'
    generate_dummy_trajectory_data(dummy_path, num_scenes=20)
    
    # Step 2: Create args object
    class Args:
        data_path = dummy_path
        traj_len = 10
        traj_dim = 5
        ego_dim = 8
        modality_names = ['lidar', 'img', 'BEV']
    
    args = Args()
    
    # Step 3: Build dataset
    print("\n" + "=" * 80)
    print("Building dataset...")
    print("=" * 80)
    dataset = build_trajectory_dataset(args)
    
    # Step 4: Test loading
    print("\n" + "=" * 80)
    print("Testing data loading...")
    print("=" * 80)
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Load first batch
    batch = dataset[0]
    print("\nBatch keys:", list(batch.keys()))
    
    print("\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.shape}")
        else:
            print(f"  {key:20s}: {type(value)}")
    
    # Step 5: Test DataLoader
    print("\n" + "=" * 80)
    print("Testing DataLoader...")
    print("=" * 80)
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print("\nBatched data shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.shape}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    
    print("\nYou can use this dataset with:")
    print("  dataset = build_trajectory_dataset(args)")
    print("  loader = DataLoader(dataset, batch_size=32, ...)")