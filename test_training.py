"""
Training script for adapted TransDiffuser.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from typing import Dict, Any, Tuple

from datasets.navsim.navsim_utilize.data import NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
from adapters import EncoderAdapter
from engine import create_transdiffuser_adapted


class DemoNavsimDataset(Dataset):
    """
    Demo/synthetic dataset that mimics NavsimDataset structure.
    Useful for quick pipeline testing without loading real data.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        bev_size: Tuple[int, int] = (200, 200),
        uniad_bev_size: Tuple[int, int] = (64, 64),
        bev_channels: int = 256,
        history_length: int = 4,
        future_horizon: int = 8,
        compute_acceleration: bool = True,
        use_uniad_bev: bool = True,
        interpolate_bev: bool = False,
        **kwargs,  # Accept extra kwargs for compatibility
    ):
        """
        Args:
            num_samples: Number of synthetic samples to generate
            bev_size: Target BEV size for labels/LiDAR (default: 200x200)
            uniad_bev_size: UniAD BEV feature size (default: 64x64)
            bev_channels: Number of UniAD BEV channels (default: 256)
            history_length: Number of history frames (default: 4)
            future_horizon: Number of future frames (default: 8)
            compute_acceleration: Include acceleration in agent state
            use_uniad_bev: Simulate UniAD BEV features
            interpolate_bev: Whether UniAD is upsampled to bev_size
        """
        self.num_samples = num_samples
        self.bev_size = bev_size
        self.uniad_bev_size = uniad_bev_size
        self.bev_channels = bev_channels
        self.history_length = history_length
        self.future_horizon = future_horizon
        self.compute_acceleration = compute_acceleration
        self.use_uniad_bev = use_uniad_bev
        self.interpolate_bev = interpolate_bev
        
        # Determine camera BEV output size
        if use_uniad_bev and interpolate_bev:
            self.camera_bev_size = bev_size
        elif use_uniad_bev:
            self.camera_bev_size = uniad_bev_size
        else:
            self.camera_bev_size = bev_size
        
        # Agent state dimension
        self.agent_dim = 7 if compute_acceleration else 5
        
        print(f"✓ DemoNavsimDataset initialized:")
        print(f"  Samples: {num_samples}")
        print(f"  LiDAR/Labels BEV: {bev_size}")
        print(f"  Camera BEV: {self.camera_bev_size} ({bev_channels} channels)")
        print(f"  Agent state dim: {self.agent_dim}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Generate a synthetic sample matching NavsimDataset output."""
        
        # Set seed for reproducibility per sample
        torch.manual_seed(idx)
        
        # 1. LiDAR BEV [2, H, W] - density + height channels
        lidar_bev = torch.rand(2, *self.bev_size)
        # Make it more realistic - sparse with some structure
        lidar_bev = lidar_bev * (torch.rand(2, *self.bev_size) > 0.7).float()
        
        # 2. Camera BEV [C, H, W] - UniAD features or placeholder
        camera_bev = torch.randn(self.bev_channels, *self.camera_bev_size) * 0.1
        
        # 3. BEV Labels - 12 channel semantic labels
        labels_tensor = self._generate_labels()
        
        # 4. Agent state [1, agent_dim]
        # [x, y, vx, vy, heading] or [x, y, vx, vy, ax, ay, heading]
        if self.compute_acceleration:
            agent_states = torch.tensor([[
                torch.randn(1).item() * 5,      # x position
                torch.randn(1).item() * 5,      # y position
                torch.randn(1).item() * 10,     # vx velocity
                torch.randn(1).item() * 2,      # vy velocity
                torch.randn(1).item() * 2,      # ax acceleration
                torch.randn(1).item() * 0.5,    # ay acceleration
                torch.randn(1).item() * 3.14,   # heading
            ]], dtype=torch.float32)
        else:
            agent_states = torch.tensor([[
                torch.randn(1).item() * 5,
                torch.randn(1).item() * 5,
                torch.randn(1).item() * 10,
                torch.randn(1).item() * 2,
                torch.randn(1).item() * 3.14,
            ]], dtype=torch.float32)
        
        # 5. Agent history [1, history_length, 5]
        agent_history = torch.randn(1, self.history_length, 5)
        # Make trajectory somewhat smooth
        for t in range(1, self.history_length):
            agent_history[0, t, :2] = agent_history[0, t-1, :2] + agent_history[0, t-1, 2:4] * 0.1
        
        # 6. GT trajectory [1, future_horizon, 5]
        gt_trajectory = torch.zeros(1, self.future_horizon, 5)
        # Start from current position and simulate forward
        gt_trajectory[0, 0, :2] = agent_states[0, :2]
        gt_trajectory[0, 0, 2:4] = agent_states[0, 2:4]
        gt_trajectory[0, 0, 4] = agent_states[0, -1]  # heading
        
        for t in range(1, self.future_horizon):
            # Simple kinematic forward simulation
            gt_trajectory[0, t, :2] = gt_trajectory[0, t-1, :2] + gt_trajectory[0, t-1, 2:4] * 0.5
            gt_trajectory[0, t, 2:4] = gt_trajectory[0, t-1, 2:4] + torch.randn(2) * 0.1
            gt_trajectory[0, t, 4] = gt_trajectory[0, t-1, 4] + torch.randn(1).item() * 0.05
        
        return {
            'camera_bev': camera_bev,
            'lidar_bev': lidar_bev,
            'labels': labels_tensor,
            'agent_states': agent_states,
            'agent_history': agent_history,
            'gt_trajectory': gt_trajectory,
            'token': f'demo_token_{idx:06d}',
        }
    
    def _generate_labels(self) -> Dict[str, torch.Tensor]:
        """Generate synthetic BEV semantic labels."""
        H, W = self.bev_size
        
        # Create road-like structure
        drivable = torch.zeros(H, W)
        # Horizontal road
        drivable[H//2-20:H//2+20, :] = 1.0
        # Vertical road
        drivable[:, W//2-20:W//2+20] = 1.0
        # Add some noise
        drivable = drivable * (1 - torch.rand(H, W) * 0.1)
        
        # Lane boundaries at road edges
        lane_boundaries = torch.zeros(H, W)
        lane_boundaries[H//2-20, :] = 1.0
        lane_boundaries[H//2+19, :] = 1.0
        lane_boundaries[:, W//2-20] = 1.0
        lane_boundaries[:, W//2+19] = 1.0
        
        # Lane dividers in middle
        lane_dividers = torch.zeros(H, W)
        lane_dividers[H//2, :] = 1.0
        lane_dividers[:, W//2] = 1.0
        
        # Random vehicle occupancy
        vehicle_occupancy = (torch.rand(H, W) > 0.995).float()
        
        # Random pedestrian occupancy (near roads)
        pedestrian_occupancy = (torch.rand(H, W) > 0.998).float()
        
        # Velocity fields (smooth)
        velocity_x = torch.randn(H, W) * 0.5
        velocity_y = torch.randn(H, W) * 0.5
        
        # Ego mask (center region)
        ego_mask = torch.zeros(H, W)
        ego_mask[H//2-5:H//2+5, W//2-3:W//2+3] = 1.0
        
        return {
            'drivable_area': drivable,
            'lane_boundaries': lane_boundaries,
            'lane_dividers': lane_dividers,
            'vehicle_occupancy': vehicle_occupancy,
            'pedestrian_occupancy': pedestrian_occupancy,
            'velocity_x': velocity_x,
            'velocity_y': velocity_y,
            'ego_mask': ego_mask,
            'traffic_lights': torch.zeros(H, W, dtype=torch.uint8),
            'vehicle_classes': torch.zeros(H, W, dtype=torch.uint8),
            'crosswalks': (torch.rand(H, W) > 0.99).float(),
            'stop_lines': (torch.rand(H, W) > 0.995).float(),
        }
    
    def get_contract(self):
        """Return a mock contract for compatibility with EncoderAdapter."""
        # Import here to avoid circular imports
        from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
        
        builder = ContractBuilder(dataset_name="DemoNavsimDataset")
        
        builder.add_feature(
            FeatureType.LIDAR_BEV,
            shape=(2, *self.bev_size),
            dtype="float32",
            description=f"Synthetic LiDAR BEV at {self.bev_size}",
        )
        
        builder.add_feature(
            FeatureType.CAMERA_BEV,
            shape=(self.bev_channels, *self.camera_bev_size),
            dtype="float32",
            description=f"Synthetic camera BEV at {self.camera_bev_size}",
        )
        
        builder.add_feature(
            FeatureType.BEV_LABELS,
            shape=(12, *self.bev_size),
            dtype="float32",
            description="Synthetic HD map labels",
        )
        
        builder.add_feature(
            FeatureType.AGENT_STATE,
            shape=(1, self.agent_dim),
            dtype="float32",
            description="Synthetic agent state",
        )
        
        builder.add_feature(
            FeatureType.AGENT_HISTORY,
            shape=(1, self.history_length, 5),
            dtype="float32",
            description="Synthetic agent history",
        )
        
        builder.add_feature(
            FeatureType.GT_TRAJECTORY,
            shape=(1, self.future_horizon, 5),
            dtype="float32",
            description="Synthetic GT trajectory",
        )
        
        builder.set_physical_limits(max_batch_size=32, memory_footprint_mb=20.0)
        
        builder.set_semantic_info(
            num_cameras=0,
            bev_channels=12,
            agent_state_dim=self.agent_dim,
            history_length=self.history_length,
            has_acceleration=self.compute_acceleration,
            has_nearby_agents=False,
            has_vector_maps=False,
        )
        
        return builder.build()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("🚀 TransDiffuser Training with Adapter System")
    print("=" * 70)
    
    # 1. Create dataset
    print("\n📦 Creating dataset...")
    if args.dataset == 'demo':
        # Fast synthetic dataset for pipeline testing
        dataset = DemoNavsimDataset(
            num_samples=args.num_demo_samples,
            bev_size=(200, 200),
            uniad_bev_size=(64, 64),
            bev_channels=256,
            compute_acceleration=True,
            use_uniad_bev=True,
            interpolate_bev=False,
        )
    elif args.dataset == 'basic':
        dataset = NavsimDataset(
            data_split=args.data_split,
            extract_labels=True,
            compute_acceleration=True,
        )
    elif args.dataset == 'enhanced':
        dataset = EnhancedNavsimDataset(
            data_split=args.data_split,
            extract_labels=True,
            extract_route_info=True,
        )
    elif args.dataset == 'phase':
        dataset = PhaseNavsimDataset(
            data_split=args.data_split,
            enable_phase_0=True,
            extract_labels=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"✓ Dataset: {len(dataset)} samples")
    
    # 2. Create adapter
    print("\n🔧 Creating adapter...")
    adapter = EncoderAdapter(dataset, mode=args.mode)
    adapter.print_summary()
    
    # 3. Create model
    print("\n🤖 Creating TransDiffuser...")
    model = create_transdiffuser_adapted(
        adapter=adapter,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        max_agents=args.max_agents,
    ).to(device)
    
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Create dataloader
    batch_size = min(adapter.get_optimal_batch_size(), args.batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
    )
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 6. Training loop
    print("\n" + "=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Adapt batch
            adapted_batch = adapter.adapt_batch(batch)
            if args.debug:
                print('adapted_batch lidar:', adapted_batch['lidar'].shape)
            
            # Move to device
            for key in adapted_batch:
                if isinstance(adapted_batch[key], torch.Tensor):
                    adapted_batch[key] = adapted_batch[key].to(device)
            
            # Forward pass
            loss_dict = model(adapted_batch)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % args.log_interval == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Diffusion: {loss_dict['diffusion_loss']:.4f} "
                    f"Decorr: {loss_dict['decorr_loss']:.4f}"
                )
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✓ Epoch {epoch+1} complete: Avg Loss={avg_loss:.4f}\n")
    
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='basic', 
                        choices=['demo', 'basic', 'enhanced', 'phase'],
                        help="Dataset type: 'demo' for fast synthetic testing")
    parser.add_argument('--data_split', type=str, default='mini')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'minimal', 'efficient', 'full'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--max_agents', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_interval', type=int, default=10)
    
    # Demo dataset specific args
    parser.add_argument('--num_demo_samples', type=int, default=100,
                        help="Number of synthetic samples for demo dataset")
    parser.add_argument('--debug', action='store_true',
                        help="Print debug info like tensor shapes")
    
    args = parser.parse_args()
    main(args)