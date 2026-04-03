"""
MINIMAL PLACEHOLDER for build_trajectory_dataset()
Copy this into your code to get started quickly
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os


def build_trajectory_dataset(args):
    """
    Minimal placeholder - loads trajectory data from disk.
    
    Required args attributes:
        - data_path: Path to dataset
        - traj_len: Trajectory length (default: 10)
        - traj_dim: Trajectory dimension (default: 5)
        - ego_dim: Ego state dimension (default: 8)
        - modality_names: List of modalities (default: ['lidar', 'img', 'BEV'])
    """
    return TrajectoryDataset(
        data_path=args.data_path,
        traj_len=getattr(args, 'traj_len', 10),
        traj_dim=getattr(args, 'traj_dim', 5),
        ego_dim=getattr(args, 'ego_dim', 8),
        modality_names=getattr(args, 'modality_names', ['lidar', 'img', 'BEV'])
    )


class TrajectoryDataset(Dataset):
    """Loads data from: data_path/scene_XXXX/*.npy"""
    
    def __init__(self, data_path, traj_len=10, traj_dim=5, ego_dim=8, 
                 modality_names=['lidar', 'img', 'BEV']):
        self.data_path = data_path
        self.traj_len = traj_len
        self.traj_dim = traj_dim
        self.ego_dim = ego_dim
        self.modality_names = modality_names
        
        # Get scene directories
        self.scene_dirs = sorted([
            os.path.join(data_path, d) 
            for d in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, d)) and 
               os.path.exists(os.path.join(data_path, d, 'trajectory.npy'))
        ])
        
        print(f"Loaded {len(self.scene_dirs)} scenes")
        
        # Compute normalization stats
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute trajectory mean/std for normalization."""
        trajectories = []
        for scene_dir in self.scene_dirs[:min(100, len(self.scene_dirs))]:
            traj = np.load(os.path.join(scene_dir, 'trajectory.npy'))
            trajectories.append(traj)
        
        trajectories = np.concatenate(trajectories, axis=0)
        self.traj_mean = trajectories.mean(axis=0).astype(np.float32)
        self.traj_std = (trajectories.std(axis=0) + 1e-8).astype(np.float32)
    
    def __len__(self):
        return len(self.scene_dirs)
    
    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]
        batch = {}
        
        # Load modalities
        for mod in self.modality_names:
            path = os.path.join(scene_dir, f'{mod}.npy')
            if os.path.exists(path):
                data = np.load(path)
                # Take last frame if temporal
                if len(data.shape) == 4:
                    data = data[-1]
                batch[mod] = torch.from_numpy(data).float()
        
        # Load ego state
        ego_path = os.path.join(scene_dir, 'ego_state.npy')
        if os.path.exists(ego_path):
            ego = np.load(ego_path)
            if len(ego.shape) == 2:
                ego = ego[-1]
            batch['ego_state'] = torch.from_numpy(ego).float()
        else:
            batch['ego_state'] = torch.zeros(self.ego_dim)
        
        # Load trajectory (REQUIRED)
        traj = np.load(os.path.join(scene_dir, 'trajectory.npy')).astype(np.float32)
        
        # Normalize
        traj = (traj - self.traj_mean) / self.traj_std
        batch['future_trajectory'] = torch.from_numpy(traj).float()
        
        return batch


# Quick test
if __name__ == "__main__":
    # Generate dummy data
    import os
    output_path = '/home/phamtamadas/DPJI/diffusion/DDPM/datasets'
    os.makedirs(output_path, exist_ok=True)
    
    for i in range(10):
        scene_dir = os.path.join(output_path, f'scene_{i:04d}')
        os.makedirs(scene_dir, exist_ok=True)
        
        np.save(os.path.join(scene_dir, 'lidar.npy'), np.random.randn(2, 64, 64))
        np.save(os.path.join(scene_dir, 'img.npy'), np.random.randn(3, 64, 64))
        np.save(os.path.join(scene_dir, 'BEV.npy'), np.random.randn(7, 64, 64))
        np.save(os.path.join(scene_dir, 'ego_state.npy'), np.random.randn(8))
        np.save(os.path.join(scene_dir, 'trajectory.npy'), np.random.randn(10, 5))
    
    # Test
    class Args:
        data_path = output_path
        traj_len = 10
        traj_dim = 5
        ego_dim = 8
        modality_names = ['lidar', 'img', 'BEV']
    
    dataset = build_trajectory_dataset(Args())
    batch = dataset[0]
    
    print("  Loaded batch with keys:", batch.keys())
    print("  Trajectory shape:", batch['future_trajectory'].shape)