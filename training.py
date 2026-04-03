"""
Training script for adapted TransDiffuser.
"""

import torch
from torch.utils.data import DataLoader
import argparse

from datasets.navsim.navsim_utilize.data import NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
from adapters import EncoderAdapter
from engine import create_transdiffuser_adapted

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("  TransDiffuser Training with Adapter System")
    print("=" * 70)
    
    # 1. Create dataset
    print("\n  Creating dataset...")
    if args.dataset == 'basic':
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
    print("\n Creating adapter...")
    adapter = EncoderAdapter(dataset, mode=args.mode)
    adapter.print_summary()
    
    # 3. Create model
    print("\n Creating TransDiffuser...")
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
    print(" STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Adapt batch
            adapted_batch = adapter.adapt_batch(batch)
            # print('adapted_batch',adapted_batch['lidar'].shape)
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
    print("TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='basic', choices=['basic', 'enhanced', 'phase'])
    parser.add_argument('--data_split', type=str, default='mini')
    parser.add_argument('--mode', type=str, default='efficient', choices=['auto', 'minimal', 'efficient', 'full'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--max_agents', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()
    main(args)