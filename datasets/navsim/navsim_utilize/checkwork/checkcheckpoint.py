#!/usr/bin/env python3
"""
Minimal BEVFusion Checkpoint Diagnostic
No dependencies, just checks what's in the checkpoint file
"""

import torch
from pathlib import Path

print("="*80)
print("BEVFusion Checkpoint Minimal Diagnostic")
print("="*80)

checkpoint_path = "/home/phamtamadas/DPJI/transdiffuser/DDPM/datasets/navsim/navsim_utilize/checkpoints/bevfusion-seg.pth"

print(f"\n1. LOADING CHECKPOINT")
print("-"*80)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Successfully loaded checkpoint")
    print(f"  File: {checkpoint_path}")
    print(f"  Size: {Path(checkpoint_path).stat().st_size / (1024*1024):.1f} MB")
except Exception as e:
    print(f"✗ FAILED to load: {e}")
    exit(1)

print(f"\n2. CHECKPOINT TYPE & STRUCTURE")
print("-"*80)

if isinstance(checkpoint, dict):
    print(f"Type: Dictionary with {len(checkpoint)} top-level keys")
    print(f"\nTop-level keys:")
    for i, key in enumerate(list(checkpoint.keys())[:10]):
        val = checkpoint[key]
        if isinstance(val, dict):
            print(f"  [{i+1}] '{key}': dict ({len(val)} items)")
        elif isinstance(val, torch.Tensor):
            print(f"  [{i+1}] '{key}': tensor {val.shape}")
        elif isinstance(val, list):
            print(f"  [{i+1}] '{key}': list ({len(val)} items)")
        else:
            print(f"  [{i+1}] '{key}': {type(val).__name__}")
    
    if len(checkpoint) > 10:
        print(f"  ... and {len(checkpoint) - 10} more keys")

elif isinstance(checkpoint, torch.nn.Module):
    print(f"Type: PyTorch Model ({type(checkpoint).__name__})")
else:
    print(f"Type: {type(checkpoint).__name__}")

print(f"\n3. STATE DICT EXTRACTION")
print("-"*80)

state_dict = None

if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"✓ Found 'state_dict' key")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"✓ Found 'model' key")
    else:
        # Try to use the dict directly as state_dict
        if all(isinstance(v, torch.Tensor) for v in list(checkpoint.values())[:5]):
            state_dict = checkpoint
            print(f"✓ Using checkpoint dict directly as state_dict")
        else:
            print(f"⚠ Could not find standard state_dict format")
            print(f"  Available keys: {list(checkpoint.keys())[:5]}")

elif isinstance(checkpoint, torch.nn.Module):
    state_dict = checkpoint.state_dict()
    print(f"✓ Extracted state_dict from model")

if state_dict is None:
    print(f"✗ Could not extract state_dict")
    exit(1)

print(f"\n4. STATE DICT ANALYSIS")
print("-"*80)

if isinstance(state_dict, dict):
    print(f"State dict has {len(state_dict)} keys")
    print(f"Total parameters: {sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)):,}")
    
    # Analyze keys
    all_keys = list(state_dict.keys())
    print(f"\nFirst 10 keys:")
    for i, key in enumerate(all_keys[:10]):
        val = state_dict[key]
        if isinstance(val, torch.Tensor):
            print(f"  {i+1}. {key}")
            print(f"     Shape: {val.shape}, Dtype: {val.dtype}")
            print(f"     Range: [{val.min():.4f}, {val.max():.4f}]")
        else:
            print(f"  {i+1}. {key}: {type(val).__name__}")
    
    # Component analysis
    print(f"\nComponent keywords in layer names:")
    keywords = {
        'camera': 0,
        'lidar': 0,
        'backbone': 0,
        'encoder': 0,
        'decoder': 0,
        'fusion': 0,
        'head': 0,
        'seg': 0,
        'classifier': 0,
        'conv': 0,
        'bn': 0,
        'norm': 0,
    }
    
    for key in all_keys:
        key_lower = key.lower()
        for keyword in keywords:
            if keyword in key_lower:
                keywords[keyword] += 1
    
    for keyword, count in sorted(keywords.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  '{keyword}': {count} layers")

print(f"\n5. WEIGHT STATISTICS")
print("-"*80)

if isinstance(state_dict, dict):
    weight_layers = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.ndim >= 2]
    
    if weight_layers:
        means = [v.float().mean().item() for v in weight_layers]
        stds = [v.float().std().item() for v in weight_layers]
        
        print(f"Weight layers: {len(weight_layers)}")
        print(f"Means: avg={sum(means)/len(means):.4f}, range=[{min(means):.4f}, {max(means):.4f}]")
        print(f"Stds:  avg={sum(stds)/len(stds):.4f}, range=[{min(stds):.4f}, {max(stds):.4f}]")
        
        # Check for dead weights
        dead_count = sum(1 for m, s in zip(means, stds) if abs(m) < 0.0001 and abs(s) < 0.0001)
        if dead_count > 0:
            print(f"\n⚠ WARNING: {dead_count} layers appear to be uninitialized (mean≈0, std≈0)")
        
        # Check for NaN
        nan_count = sum(1 for v in weight_layers if torch.isnan(v.float()).any())
        if nan_count > 0:
            print(f"✗ ERROR: {nan_count} layers contain NaN values!")
        else:
            print(f"✓ No NaN values detected")

print(f"\n6. DIAGNOSIS")
print("-"*80)

issues = []

# Check 1: File size
if Path(checkpoint_path).stat().st_size < 10 * 1024 * 1024:
    issues.append("File is very small (<10 MB) - may be incomplete")

# Check 2: State dict
if state_dict is None:
    issues.append("Could not extract state_dict - format issue")
elif len(state_dict) < 10:
    issues.append("State dict has very few keys (<10) - may be incomplete")

# Check 3: Parameters
if isinstance(state_dict, dict):
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    if total_params < 1000000:  # < 1M
        issues.append("Total parameters <1M - model might be too small")
    elif total_params > 500000000:  # > 500M
        issues.append("Total parameters >500M - model might be too large")

if issues:
    print("Potential Issues Detected:")
    for issue in issues:
        print(f"  ⚠ {issue}")
else:
    print("✓ No obvious issues detected with checkpoint file")

print(f"\n7. RECOMMENDATION")
print("-"*80)

if len(state_dict) if isinstance(state_dict, dict) else 0 > 50:
    print("""
RECOMMENDATION: Use the checkpoint directly with BEVFusion code

This checkpoint appears to be complete. However:
1. It may need official BEVFusion code to load properly
2. Our simplified model architecture may not match
3. It's trained on nuScenes, not NAVSIM

OPTIONS:
A) Use official BEVFusion repository:
   - Clone: https://github.com/mit-han-lab/bevfusion
   - Use their code to load checkpoint
   - Adapt for NAVSIM

B) Use heuristic-only approach (RECOMMENDED):
   - Heuristics already working well (89% coverage)
   - Skip neural model entirely
   - Faster and more reliable

C) Fine-tune on NAVSIM:
   - Load pretrained weights
   - Train on NAVSIM labels
   - Better results but requires labeled data
""")
else:
    print("""
RECOMMENDATION: This checkpoint may be incomplete or incorrect

The checkpoint has fewer components than expected for BEVFusion.
Options:
1. Download official checkpoint from MIT-Han-Lab
2. Use heuristic-only approach (already working)
3. Train custom model on your data
""")

print("="*80)