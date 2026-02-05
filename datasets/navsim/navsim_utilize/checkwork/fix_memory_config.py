"""
Quick Start Script - Minimal Memory Configuration
For RTX 5050 8GB - Prevents OOM crashes

This will modify your bev_process.py config section to use minimal memory.
"""

import re

def update_config_to_minimal(filepath='bev_process.py'):
    """Update bev_process.py with minimal memory config."""
    
    print("="*80)
    print("Updating bev_process.py to MINIMAL MEMORY configuration")
    print("="*80)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find and replace the config section
    config_pattern = r"config = \{[^}]*'bev_size': \([0-9]+, [0-9]+\),[^}]*'batch_size': [0-9]+,[^}]*\}"
    
    new_config = """config = {
        'bev_checkpoint': './checkpoints/bevfusion-seg.pth',
        'bev_size': (32, 32),      # REDUCED: 64->32 (4x less memory!)
        'history_length': 4,
        'future_horizon': 8,
        'use_cache': True,
        'precompute_bev': True,
        'force_recompute': False,  # Set to True to regenerate
        'batch_size': 1,           # REDUCED: 16->1 (process one scene at a time)
        'device': 'auto',
    }"""
    
    if re.search(config_pattern, content, re.DOTALL):
        content = re.sub(config_pattern, new_config, content, flags=re.DOTALL)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print("\n✓ Updated successfully!")
        print("\nChanges made:")
        print("  - bev_size: (64, 64) → (32, 32)  [4x less memory]")
        print("  - batch_size: 16 → 1              [16x less memory per iteration]")
        print("\nEstimated peak memory: ~3-4GB (safe for 8GB GPU)")
        print("\nYou can now run:")
        print("  python bev_process.py")
        
    else:
        print("\n⚠️  Could not automatically update config")
        print("\nPlease manually edit bev_process.py:")
        print("\n" + "="*80)
        print("Find this section:")
        print("-"*80)
        print("""
    config = {
        'bev_checkpoint': './checkpoints/bevfusion-seg.pth',
        'bev_size': (64, 64),
        ...
        'batch_size': 16,
        ...
    }
""")
        print("-"*80)
        print("\nChange to:")
        print("-"*80)
        print("""
    config = {
        'bev_checkpoint': './checkpoints/bevfusion-seg.pth',
        'bev_size': (32, 32),      # Changed!
        ...
        'batch_size': 1,           # Changed!
        ...
    }
""")
        print("="*80)
    
    print("\n" + "="*80)
    print("Memory Optimization Tips")
    print("="*80)
    print("""
1. Monitor GPU memory while running:
   watch -n 1 nvidia-smi

2. If still OOM, try:
   - Close browser/other GPU apps
   - Reduce bev_size to (24, 24)
   - Use device='cpu' (slower but won't crash)

3. Once it works, you can increase gradually:
   - batch_size: 1 → 2 → 4
   - bev_size: (32,32) → (48,48) → (64,64)

4. Typical memory usage:
   - batch_size=1, bev=(32,32): ~3-4GB ✓ Safe
   - batch_size=2, bev=(48,48): ~4-5GB ✓ Should work
   - batch_size=4, bev=(64,64): ~5-6GB ✓ Good balance
   - batch_size=8, bev=(64,64): ~6-7GB ⚠️  Risky
   - batch_size=16, bev=(64,64): ~7-8GB ❌ Will crash
""")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'bev_process.py'
    
    try:
        update_config_to_minimal(filepath)
    except FileNotFoundError:
        print(f"\n❌ File not found: {filepath}")
        print("\nPlease run from the same directory as bev_process.py")
        print("Or specify path: python fix_memory_config.py /path/to/bev_process.py")