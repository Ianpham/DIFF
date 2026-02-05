"""
Multi-Agent Trajectory Dataset for TransDiffuser
Loads multi-modal driving data and multi-agent future trajectories
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from typing import Dict, List, Optional


class TrajectoryDataset(Dataset):
    """
    Dataset for loading multi-modal driving data and multi-agent trajectories.
    
    Expected data structure:
    data_path/
        scene_0001/
            lidar.npy              # (2, H, W) - LiDAR at current timestep
            img.npy                # (3, H, W) - RGB image at current timestep
            BEV.npy                # (7, H, W) - Bird's eye view at current timestep
            agent_states.npy       # (N, 5) - Current agent states [x, y, vx, vy, heading]
            agent_history.npy      # (N, 30, 5) - Agent history (30 frames)
            agent_future.npy       # (N, 20, 5) - Future trajectories (20 waypoints)
            metadata.json          # Scene metadata
        scene_0002/
            ...
    """
    
    def __init__(
        self,
        data_path: str,
        future_horizon: int = 20,
        history_length: int = 30,
        traj_dim: int = 5,
        max_agents: int = 32,
        modality_names: List[str] = ['lidar', 'img', 'BEV'],
        transform=None,
        normalize_trajectory: bool = True
    ):
        self.data_path = data_path
        self.future_horizon = future_horizon
        self.history_length = history_length
        self.traj_dim = traj_dim
        self.max_agents = max_agents
        self.modality_names = modality_names
        self.transform = transform
        self.normalize_trajectory = normalize_trajectory
        
        # Get all scene directories
        self.scene_dirs = sorted([
            os.path.join(data_path, d) 
            for d in os.listdir(data_path) 
            if os.path.isdir(os.path.join(data_path, d)) and d.startswith('scene_')
        ])
        
        print(f"Found {len(self.scene_dirs)} scenes")
        
        # Compute trajectory statistics for normalization
        if self.normalize_trajectory:
            self._compute_trajectory_stats()
    
    def _compute_trajectory_stats(self):
        """Compute mean and std for trajectory normalization"""
        sample_size = min(100, len(self.scene_dirs))
        all_states = []
        all_futures = []
        
        for i in range(sample_size):
            # Agent states
            states_path = os.path.join(self.scene_dirs[i], 'agent_states.npy')
            if os.path.exists(states_path):
                states = np.load(states_path)
                all_states.append(states)
            
            # Future trajectories
            future_path = os.path.join(self.scene_dirs[i], 'agent_future.npy')
            if os.path.exists(future_path):
                future = np.load(future_path)
                all_futures.append(future)
        
        # Concatenate and compute stats
        all_states = np.concatenate(all_states, axis=0)  # (total_agents, 5)
        all_futures = np.concatenate(all_futures, axis=0)  # (total_agents, 20, 5)
        
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8
        
        self.future_mean = all_futures.mean(axis=(0, 1))
        self.future_std = all_futures.std(axis=(0, 1)) + 1e-8
        
        print(f"State mean: {self.state_mean}")
        print(f"State std: {self.state_std}")
        print(f"Future mean: {self.future_mean}")
        print(f"Future std: {self.future_std}")
    
    def __len__(self):
        return len(self.scene_dirs)
    
    def _pad_or_truncate_agents(self, data, target_shape):
        """
        Pad or truncate agent dimension to fixed size.
        
        Args:
            data: (N, ...) array
            target_shape: (max_agents, ...)
        """
        N = data.shape[0]
        
        if N < self.max_agents:
            # Pad with zeros
            pad_shape = (self.max_agents - N,) + data.shape[1:]
            padding = np.zeros(pad_shape, dtype=data.dtype)
            data = np.concatenate([data, padding], axis=0)
        elif N > self.max_agents:
            # Truncate
            data = data[:self.max_agents]
        
        return data
    
    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]
        
        batch = {}
        
        # ========== LOAD CONTEXT MODALITIES (SHARED) ==========
        for modality in self.modality_names:
            modality_path = os.path.join(scene_dir, f'{modality}.npy')
            if os.path.exists(modality_path):
                data = np.load(modality_path)  # (C, H, W)
                batch[modality] = torch.from_numpy(data).float()
        
        # ========== LOAD AGENT DATA ==========
        # Current agent states: (N, 5)
        states_path = os.path.join(scene_dir, 'agent_states.npy')
        if os.path.exists(states_path):
            agent_states = np.load(states_path)  # (N, 5)
            
            # Pad/truncate to max_agents
            agent_states = self._pad_or_truncate_agents(agent_states, (self.max_agents, 5))
            
            # Normalize
            if self.normalize_trajectory:
                agent_states = (agent_states - self.state_mean) / self.state_std
            
            batch['agent_states'] = torch.from_numpy(agent_states).float()
        else:
            # If no agent data, create dummy single agent
            batch['agent_states'] = torch.zeros(self.max_agents, 5)
        
        # Agent history: (N, 30, 5)
        history_path = os.path.join(scene_dir, 'agent_history.npy')
        if os.path.exists(history_path):
            agent_history = np.load(history_path)  # (N, 30, 5)
            
            # Pad/truncate
            agent_history = self._pad_or_truncate_agents(
                agent_history, (self.max_agents, self.history_length, 5)
            )
            
            # Normalize
            if self.normalize_trajectory:
                agent_history = (agent_history - self.state_mean) / self.state_std
            
            batch['agent_history'] = torch.from_numpy(agent_history).float()
        else:
            batch['agent_history'] = torch.zeros(self.max_agents, self.history_length, 5)
        
        # Future trajectories: (N, 20, 5)
        future_path = os.path.join(scene_dir, 'agent_future.npy')
        if os.path.exists(future_path):
            agent_future = np.load(future_path)  # (N, 20, 5)
            
            # Pad/truncate
            agent_future = self._pad_or_truncate_agents(
                agent_future, (self.max_agents, self.future_horizon, 5)
            )
            
            # Normalize
            if self.normalize_trajectory:
                agent_future = (agent_future - self.future_mean) / self.future_std
            
            batch['agent_future'] = torch.from_numpy(agent_future).float()
        else:
            batch['agent_future'] = torch.zeros(self.max_agents, self.future_horizon, 5)
        
        # ========== LOAD METADATA ==========
        metadata_path = os.path.join(scene_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            batch['metadata'] = metadata
            batch['num_agents'] = metadata.get('num_agents', 1)  # Track real agent count
        else:
            batch['num_agents'] = 1
        
        # Apply transforms
        if self.transform is not None:
            batch = self.transform(batch)
        
        return batch


def build_trajectory_dataset(args):
    """Build trajectory dataset from args."""
    dataset = TrajectoryDataset(
        data_path=args.data_path,
        future_horizon=args.future_horizon,
        history_length=args.history_length,
        traj_dim=args.traj_dim,
        max_agents=args.max_agents,
        modality_names=args.modality_names,
        normalize_trajectory=True
    )
    
    return dataset


# Example data generation for testing
def generate_dummy_data(output_path, num_scenes=100, num_agents_range=(4, 12)):
    """
    Generate dummy multi-agent data for testing.
    
    Args:
        output_path: Where to save data
        num_scenes: Number of scenes to generate
        num_agents_range: (min, max) number of agents per scene
    """
    os.makedirs(output_path, exist_ok=True)
    
    for i in range(num_scenes):
        scene_dir = os.path.join(output_path, f'scene_{i:04d}')
        os.makedirs(scene_dir, exist_ok=True)
        
        # Random number of agents for this scene
        num_agents = np.random.randint(num_agents_range[0], num_agents_range[1] + 1)
        
        # ========== CONTEXT (SHARED) ==========
        H, W = 64, 64
        
        # LiDAR: (2, H, W)
        lidar = np.random.randn(2, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'lidar.npy'), lidar)
        
        # Image: (3, H, W)
        img = np.random.randn(3, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'img.npy'), img)
        
        # BEV: (7, H, W)
        bev = np.random.randn(7, H, W).astype(np.float32)
        np.save(os.path.join(scene_dir, 'BEV.npy'), bev)
        
        # ========== AGENT DATA ==========
        # Current states: (N, 5) - [x, y, vx, vy, heading]
        agent_states = np.random.randn(num_agents, 5).astype(np.float32)
        # Make it more realistic
        agent_states[:, 0:2] *= 50.0  # x, y positions in range [-50, 50]
        agent_states[:, 2:4] *= 10.0  # velocities in range [-10, 10] m/s
        agent_states[:, 4] = np.random.uniform(-np.pi, np.pi, num_agents)  # heading
        np.save(os.path.join(scene_dir, 'agent_states.npy'), agent_states)
        
        # History: (N, 30, 5)
        agent_history = np.random.randn(num_agents, 30, 5).astype(np.float32)
        # Make history somewhat continuous with current state
        for j in range(num_agents):
            # Linear interpolation from past to current
            for k in range(30):
                alpha = k / 30.0
                agent_history[j, k] = agent_states[j] * alpha + np.random.randn(5) * 0.5
        np.save(os.path.join(scene_dir, 'agent_history.npy'), agent_history)
        
        # Future: (N, 20, 5)
        agent_future = np.random.randn(num_agents, 20, 5).astype(np.float32)
        # Make future somewhat continuous with current state
        for j in range(num_agents):
            for k in range(20):
                # Simple forward projection with noise
                dt = (k + 1) * 0.5  # 0.5s per step (2Hz)
                agent_future[j, k, 0:2] = agent_states[j, 0:2] + agent_states[j, 2:4] * dt
                agent_future[j, k, 0:2] += np.random.randn(2) * 2.0  # position noise
                agent_future[j, k, 2:4] = agent_states[j, 2:4] + np.random.randn(2) * 0.5  # velocity
                agent_future[j, k, 4] = agent_states[j, 4] + np.random.randn() * 0.1  # heading
        np.save(os.path.join(scene_dir, 'agent_future.npy'), agent_future)
        
        # ========== METADATA ==========
        metadata = {
            'scene_id': f'scene_{i:04d}',
            'num_agents': num_agents,
            'weather': np.random.choice(['clear', 'rain', 'fog']),
            'time_of_day': np.random.choice(['day', 'night', 'dawn', 'dusk'])
        }
        with open(os.path.join(scene_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
    
    print(f"Generated {num_scenes} dummy scenes at {output_path}")


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    # Generate dummy data
    dummy_path = './test_data'
    generate_dummy_data(dummy_path, num_scenes=10, num_agents_range=(4, 8))
    
    # Test loading
    class Args:
        data_path = dummy_path
        future_horizon = 20
        history_length = 30
        traj_dim = 5
        max_agents = 32
        modality_names = ['lidar', 'img', 'BEV']
    
    args = Args()
    dataset = build_trajectory_dataset(args)
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test getitem
    batch = dataset[0]
    print("\nBatch keys:", batch.keys())
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*50)
    print("Sample data statistics:")
    print("="*50)
    print(f"Agent states range: [{batch['agent_states'].min():.2f}, {batch['agent_states'].max():.2f}]")
    print(f"Agent history range: [{batch['agent_history'].min():.2f}, {batch['agent_history'].max():.2f}]")
    print(f"Agent future range: [{batch['agent_future'].min():.2f}, {batch['agent_future'].max():.2f}]")
    print(f"Actual number of agents: {batch['num_agents']}")
    
    print("\nDataset test passed!")