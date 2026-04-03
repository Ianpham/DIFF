#!/usr/bin/env python3
"""
BEVFusion Checkpoint Analyzer
==============================

Analyzes a BEVFusion checkpoint to understand its structure
and help with loading.
"""

import torch
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint structure."""
    print("="*70)
    print(f"Analyzing: {checkpoint_path}")
    print("="*70)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("  Checkpoint loaded")
    except Exception as e:
        print(f"  Failed to load: {e}")
        return
    
    # Top-level structure
    print("\n" + "="*70)
    print("Top-level Keys")
    print("="*70)
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  {key:20s} dict with {len(value)} entries")
            elif isinstance(value, torch.Tensor):
                print(f"  {key:20s} tensor {tuple(value.shape)}")
            elif isinstance(value, list):
                print(f"  {key:20s} list with {len(value)} items")
            else:
                print(f"  {key:20s} {type(value).__name__}")
    else:
        print(f"Checkpoint is {type(checkpoint).__name__}, not a dict")
        return
    
    # Analyze state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume the checkpoint is the state_dict
        state_dict = checkpoint
    
    print("\n" + "="*70)
    print("State Dict Analysis")
    print("="*70)
    print(f"Total parameters: {len(state_dict)}")
    
    # Group by module
    groups = defaultdict(list)
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 0:
            # Get top-level module
            group = parts[0]
            groups[group].append(key)
    
    print(f"\nTop-level modules: {len(groups)}")
    for group, keys in sorted(groups.items()):
        total_params = sum(state_dict[k].numel() for k in keys)
        print(f"\n  {group:20s} {len(keys):4d} parameters, {total_params:12,d} values")
        
        # Show first few
        for key in sorted(keys)[:3]:
            shape = tuple(state_dict[key].shape)
            dtype = state_dict[key].dtype
            print(f"    {key:60s} {str(shape):20s} {dtype}")
        
        if len(keys) > 3:
            print(f"    ... and {len(keys) - 3} more")
    
    # Detailed encoder analysis
    print("\n" + "="*70)
    print("Encoder Analysis")
    print("="*70)
    
    encoder_keys = [k for k in state_dict.keys() if k.startswith('encoders.')]
    
    # Group by sensor
    encoder_groups = defaultdict(list)
    for key in encoder_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            sensor = parts[1]  # camera, lidar, radar
            encoder_groups[sensor].append(key)
    
    for sensor, keys in sorted(encoder_groups.items()):
        print(f"\n{sensor.upper()} Encoder: {len(keys)} parameters")
        
        # Sub-modules
        sub_modules = defaultdict(list)
        for key in keys:
            parts = key.split('.')
            if len(parts) >= 3:
                sub_module = parts[2]  # backbone, neck, vtransform, etc.
                sub_modules[sub_module].append(key)
        
        for sub_module, sub_keys in sorted(sub_modules.items()):
            total_params = sum(state_dict[k].numel() for k in sub_keys)
            print(f"  {sub_module:15s} {len(sub_keys):4d} params, {total_params:12,d} values")
    
    # Decoder analysis
    print("\n" + "="*70)
    print("Decoder Analysis")
    print("="*70)
    
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
    print(f"Decoder parameters: {len(decoder_keys)}")
    
    decoder_groups = defaultdict(list)
    for key in decoder_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            module = parts[1]  # backbone, neck
            decoder_groups[module].append(key)
    
    for module, keys in sorted(decoder_groups.items()):
        total_params = sum(state_dict[k].numel() for k in keys)
        print(f"  {module:15s} {len(keys):4d} params, {total_params:12,d} values")
    
    # Head analysis
    print("\n" + "="*70)
    print("Head Analysis")
    print("="*70)
    
    head_keys = [k for k in state_dict.keys() if k.startswith('heads.')]
    
    head_groups = defaultdict(list)
    for key in head_keys:
        parts = key.split('.')
        if len(parts) >= 2:
            head_type = parts[1]  # map, object, etc.
            head_groups[head_type].append(key)
    
    for head_type, keys in sorted(head_groups.items()):
        total_params = sum(state_dict[k].numel() for k in keys)
        print(f"  {head_type:15s} {len(keys):4d} params, {total_params:12,d} values")
    
    # Fuser analysis
    print("\n" + "="*70)
    print("Fuser Analysis")
    print("="*70)
    
    fuser_keys = [k for k in state_dict.keys() if k.startswith('fuser.')]
    if fuser_keys:
        total_params = sum(state_dict[k].numel() for k in fuser_keys)
        print(f"Fuser parameters: {len(fuser_keys)}, {total_params:,} values")
    else:
        print("No fuser found")
    
    # Config analysis
    if 'meta' in checkpoint:
        print("\n" + "="*70)
        print("Meta Information")
        print("="*70)
        meta = checkpoint['meta']
        for key, value in meta.items():
            print(f"  {key:20s} {value}")
    
    # Model config
    if 'model_config' in checkpoint or 'config' in checkpoint:
        print("\n" + "="*70)
        print("Model Configuration")
        print("="*70)
        
        config = checkpoint.get('model_config', checkpoint.get('config'))
        print_dict(config, indent=2)
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {len(state_dict)}")
    print(f"Total values: {total_params:,}")
    print(f"Checkpoint size: {Path(checkpoint_path).stat().st_size / 1024 / 1024:.1f} MB")


def print_dict(d, indent=0):
    """Pretty print nested dict."""
    if not isinstance(d, dict):
        print(f"{' ' * indent}{d}")
        return
    
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            print_dict(value, indent + 2)
        elif isinstance(value, list) and len(value) > 5:
            print(f"{' ' * indent}{key}: list with {len(value)} items")
        else:
            print(f"{' ' * indent}{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Analyze BEVFusion checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--save-keys', type=str, default=None,
                       help='Save all parameter keys to file')
    
    args = parser.parse_args()
    
    # Analyze
    analyze_checkpoint(args.checkpoint)
    
    # Save keys if requested
    if args.save_keys:
        print(f"\nSaving parameter keys to: {args.save_keys}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        with open(args.save_keys, 'w') as f:
            for key in sorted(state_dict.keys()):
                shape = tuple(state_dict[key].shape)
                f.write(f"{key:80s} {str(shape)}\n")
        
        print(f"  Saved {len(state_dict)} keys")


if __name__ == "__main__":
    main()