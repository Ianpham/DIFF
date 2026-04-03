"""
BEVFusion Preprocessing Crash Analyzer - Smart Path Detection
Automatically finds NAVSIM data and analyzes crash state
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch

def find_navsim_data():
    """Search for NAVSIM data directory."""
    print("🔍 Searching for NAVSIM data directory...")
    print("-"*80)
    
    # Common locations to check
    search_paths = [
        Path('./data'),
        Path('../data'),
        Path('../../data'),
        Path('./'),
        Path(os.environ.get('OPENSCENE_DATA_ROOT', '')),
    ]
    
    found_paths = []
    
    for base_path in search_paths:
        if not base_path.exists():
            continue
        
        # Look for NAVSIM indicators
        indicators = [
            base_path / 'mini_navsim_logs',
            base_path / 'mini_sensor_blobs',
            base_path / 'cache',
        ]
        
        for indicator in indicators:
            if indicator.exists():
                print(f"  Found: {base_path} (contains {indicator.name})")
                found_paths.append(base_path)
                break
    
    if not found_paths:
        print("\n  No NAVSIM data found in common locations")
        print("\nSearched:")
        for p in search_paths:
            if p.exists():
                print(f"  - {p.absolute()}")
        return None
    
    return found_paths[0]


def analyze_crash(data_root=None):
    """Analyze the state of preprocessing to determine crash point."""
    
    # Auto-detect data root if not provided
    if data_root is None:
        data_root = find_navsim_data()
        if data_root is None:
            print("\n" + "="*80)
            print("   MANUAL PATH REQUIRED")
            print("="*80)
            print("\nCouldn't auto-detect NAVSIM data directory.")
            print("\nPlease run with explicit path:")
            print("  python analyze_crash.py /path/to/your/navsim/data")
            print("\nOr set environment variable:")
            print("  export OPENSCENE_DATA_ROOT=/path/to/your/navsim/data")
            print("  python analyze_crash.py")
            print("="*80 + "\n")
            return
    else:
        data_root = Path(data_root)
    
    print("\n" + "="*80)
    print("BEVFusion Preprocessing Crash Analysis")
    print("="*80)
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data root: {data_root.absolute()}")
    print("="*80 + "\n")
    
    cache_dir = data_root / 'cache' / 'transdiffuser_mini'
    bev_cache_dir = cache_dir / 'bevfusion_features'
    
    # 1. Check if cache directories exist
    print("📁 CACHE DIRECTORY STATUS")
    print("-"*80)
    
    if not cache_dir.exists():
        print(f"  Cache directory does not exist: {cache_dir}")
        print("\n   FINDING: Preprocessing never started")
        print("\nPossible reasons:")
        print("  1. Script crashed before creating cache directory")
        print("  2. Different data_root was used")
        print("  3. Preprocessing was never actually run")
        
        # Check if NAVSIM data exists
        navsim_logs = data_root / 'mini_navsim_logs' / 'mini'
        sensor_blobs = data_root / 'mini_sensor_blobs' / 'mini'
        
        print("\n📦 Checking for NAVSIM source data:")
        if navsim_logs.exists():
            print(f"    Found navsim logs: {navsim_logs}")
            log_files = list(navsim_logs.glob('*.json'))
            print(f"    Contains {len(log_files)} log files")
        else:
            print(f"    Missing navsim logs: {navsim_logs}")
        
        if sensor_blobs.exists():
            print(f"    Found sensor blobs: {sensor_blobs}")
        else:
            print(f"    Missing sensor blobs: {sensor_blobs}")
        
        print("\n" + "="*80)
        print("LIKELY CAUSE")
        print("="*80)
        print("\n🔴 Preprocessing script never successfully started")
        print("\nWhat probably happened:")
        print("  1. Script crashed during initialization (before cache creation)")
        print("  2. Error in BEVFusion checkpoint loading")
        print("  3. Python import error or dependency issue")
        print("  4. CUDA/GPU initialization failure")
        
        print("\nRecommended actions:")
        print("  1. Check terminal output for error messages")
        print("  2. Verify BEVFusion checkpoint exists:")
        print("     ls -lh ./checkpoints/bevfusion-seg.pth")
        print("  3. Test CUDA availability:")
        print("     python -c 'import torch; print(torch.cuda.is_available())'")
        print("  4. Try running with verbose error catching:")
        print("     python bev_process.py 2>&1 | tee preprocessing.log")
        print("="*80 + "\n")
        return
    
    print(f"  Cache directory exists: {cache_dir}")
    
    if not bev_cache_dir.exists():
        print(f"  BEV cache directory does not exist: {bev_cache_dir}")
        print("\n   FINDING: Cache initialized but BEV extraction never started")
        
        print("\n" + "="*80)
        print("LIKELY CAUSE")
        print("="*80)
        print("\n🟡 Script initialized but crashed before BEV extraction")
        print("\nWhat probably happened:")
        print("  1. BEVFusion model loading failed")
        print("  2. GPU out of memory during model initialization")
        print("  3. Checkpoint file corrupted or wrong format")
        
        print("\nRecommended actions:")
        print("  1. Check BEVFusion checkpoint:")
        print("     python -c 'import torch; torch.load(\"./checkpoints/bevfusion-seg.pth\")'")
        print("  2. Try CPU-only mode first:")
        print("     Edit config: device='cpu'")
        print("  3. Check GPU memory:")
        print("     nvidia-smi")
        print("="*80 + "\n")
        return
    
    print(f"  BEV cache directory exists: {bev_cache_dir}")
    
    # 2. Analyze BEV features
    print("\n  BEV FEATURE EXTRACTION STATUS")
    print("-"*80)
    
    cached_files = list(bev_cache_dir.glob('*.pt'))
    num_cached = len(cached_files)
    
    if num_cached == 0:
        print("  No BEV features extracted")
        print("\n   FINDING: Crash occurred during first feature extraction")
        
        print("\n" + "="*80)
        print("LIKELY CAUSE")
        print("="*80)
        print("\n🔴 GPU out of memory during first forward pass")
        print("\nWhat probably happened:")
        print("  1. BEVFusion model loaded successfully")
        print("  2. First image forward pass triggered OOM")
        print("  3. System crashed/froze due to memory exhaustion")
        
        print("\nRecommended actions:")
        print("  1. REDUCE batch_size in config (try 4 or 8)")
        print("  2. Check GPU memory before running:")
        print("     nvidia-smi")
        print("  3. Close other GPU applications")
        print("  4. Try smaller BEV size: bev_size=(32, 32)")
        print("  5. Monitor memory during run:")
        print("     watch -n 1 nvidia-smi")
        print("="*80 + "\n")
        return
    
    print(f"  Found {num_cached} cached BEV features")
    
    # Calculate sizes
    total_size_bytes = sum(f.stat().st_size for f in cached_files)
    total_size_mb = total_size_bytes / (1024 * 1024)
    total_size_gb = total_size_mb / 1024
    avg_size_mb = total_size_mb / num_cached if num_cached > 0 else 0
    
    print(f"  Total cache size: {total_size_gb:.2f} GB ({total_size_mb:.1f} MB)")
    print(f"  Average per file: {avg_size_mb:.2f} MB")
    
    # 3. Check for incomplete extractions
    print("\n🔍 CHECKING FOR CORRUPTION")
    print("-"*80)
    
    corrupted = []
    valid = []
    sample_size = min(20, len(cached_files))
    
    print(f"Sampling {sample_size} files...")
    
    for cache_file in cached_files[:sample_size]:
        try:
            data = torch.load(cache_file, map_location='cpu')
            valid.append(cache_file.name)
            # Clean up
            del data
        except Exception as e:
            corrupted.append((cache_file.name, str(e)))
    
    if corrupted:
        print(f"  Found {len(corrupted)} corrupted cache files:")
        for name, error in corrupted[:5]:
            print(f"   - {name}")
            print(f"     Error: {error[:80]}...")
        print("\n   FINDING: Files may be corrupted from interrupted writes")
        print("\nRecommended: Delete corrupted files or set force_recompute=True")
    else:
        print(f"  All sampled files are valid ({len(valid)}/{sample_size} checked)")
    
    # 4. Analyze last modified times
    print("\n⏰ TIMELINE ANALYSIS")
    print("-"*80)
    
    file_times = [(f, f.stat().st_mtime) for f in cached_files]
    file_times.sort(key=lambda x: x[1])
    
    first_file, first_time = file_times[0]
    last_file, last_time = file_times[-1]
    
    first_dt = datetime.fromtimestamp(first_time)
    last_dt = datetime.fromtimestamp(last_time)
    duration = last_time - first_time
    
    print(f"First file:  {first_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"             {first_file.name[:50]}...")
    print(f"\nLast file:   {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"             {last_file.name[:50]}...")
    print(f"\nDuration:    {duration:.0f} seconds ({duration/60:.1f} minutes)")
    
    if duration > 0:
        rate = num_cached / duration
        print(f"Rate:        {rate:.2f} scenes/second")
        print(f"             {rate * 60:.1f} scenes/minute")
    
    # Time since last file
    time_since_last = datetime.now().timestamp() - last_time
    print(f"\nTime since last file: {time_since_last/60:.1f} minutes ago")
    
    # 5. Check error log
    print("\n📋 ERROR LOG CHECK")
    print("-"*80)
    
    error_log_file = cache_dir / 'bevfusion_extraction_errors.log'
    
    if error_log_file.exists():
        print(f"  Error log found: {error_log_file}")
        
        with open(error_log_file, 'r') as f:
            errors = f.readlines()
        
        print(f"  Total errors logged: {len(errors)}")
        
        if errors:
            print("\n  Last 10 errors:")
            for error in errors[-10:]:
                print(f"    {error.strip()}")
    else:
        print("ℹ️  No error log found (normal if no errors before crash)")
    
    # 6. Try to estimate total scenes
    print("\n📈 PROGRESS ESTIMATION")
    print("-"*80)
    
    # Try to count scenes from navsim logs
    navsim_logs = data_root / 'mini_navsim_logs' / 'mini'
    estimated_total = None
    
    if navsim_logs.exists():
        try:
            # Count log files as proxy for scenes
            log_files = list(navsim_logs.glob('*.json'))
            # Each log might have multiple scenes, but this is a rough estimate
            estimated_total = len(log_files) * 10  # rough multiplier
            print(f"Found {len(log_files)} log files")
        except:
            pass
    
    if estimated_total is None:
        estimated_total = 5000  # Fallback for mini split
        print(f"Using default estimate for mini split")
    
    completion_pct = (num_cached / estimated_total) * 100
    
    print(f"\nProcessed:        {num_cached:,} scenes")
    print(f"Estimated total:  {estimated_total:,} scenes")
    print(f"Completion:       ~{completion_pct:.1f}%")
    
    if num_cached < estimated_total:
        remaining = estimated_total - num_cached
        print(f"Remaining:        ~{remaining:,} scenes")
        
        if duration > 0 and rate > 0:
            remaining_time = remaining / rate
            print(f"Est. time left:   ~{remaining_time/60:.1f} minutes")
    
    # 7. System diagnostics
    print("\n💻 SYSTEM DIAGNOSTICS")
    print("-"*80)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        try:
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU: {device_name}")
            
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU memory: {total_mem:.2f} GB")
            
            # Try to get current memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"Currently allocated: {allocated:.2f} GB")
            print(f"Currently reserved:  {reserved:.2f} GB")
        except Exception as e:
            print(f"   Could not query GPU info: {e}")
    
    # 8. Final diagnosis
    print("\n" + "="*80)
    print("CRASH DIAGNOSIS & RECOMMENDATIONS")
    print("="*80)
    
    if num_cached < 10:
        print("\n🔴 CRITICAL: Very few scenes processed")
        print("\nMost likely cause: GPU out of memory")
        print("\nWhat to do:")
        print("  1. Reduce batch_size to 4 or 8")
        print("  2. Reduce bev_size to (32, 32)")
        print("  3. Run nvidia-smi to check memory usage")
        print("  4. Close other GPU applications")
        
    elif completion_pct < 25:
        print(f"\n🟡 EARLY CRASH: {completion_pct:.1f}% complete")
        print("\nMost likely causes:")
        print("  1. Gradual memory leak")
        print("  2. Thermal throttling")
        print("  3. Specific problematic scene")
        print("\nWhat to do:")
        print("  1. Simply re-run (will skip existing files)")
        print("  2. Monitor temperature: watch sensors")
        print("  3. Check error log for patterns")
        
    elif completion_pct < 75:
        print(f"\n🟡 MID-CRASH: {completion_pct:.1f}% complete")
        print("\nMost likely causes:")
        print("  1. Random system instability")
        print("  2. Power/thermal issue")
        print("  3. Slow memory leak")
        print("\nWhat to do:")
        print("  1. Re-run to continue (auto-resumes)")
        print("  2. Monitor system during run")
        
    else:
        print(f"\n🟢 NEARLY COMPLETE: {completion_pct:.1f}% done!")
        print("\nMost likely cause: Random crash near end")
        print("\nWhat to do:")
        print("  1. Simply re-run to finish")
        print("  2. Should complete quickly")
    
    print("\n" + "-"*80)
    print("NEXT STEPS")
    print("-"*80)
    print("\n  Good news: Preprocessing can resume automatically!")
    print("\n1. To continue where it left off:")
    print("   python bev_process.py")
    print("\n2. To start fresh (if corruption suspected):")
    print("   Edit config: force_recompute=True")
    print(f"\n3. Monitor progress:")
    print("   watch -n 5 \"ls -1 {bev_cache_dir} | wc -l\"")
    
    print("\n" + "="*80)
    print(f"Cache location: {bev_cache_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = None
    
    try:
        analyze_crash(data_root)
    except Exception as e:
        print(f"\n  Analyzer error: {e}")
        print("\nPlease provide data root manually:")
        print("  python analyze_crash.py /path/to/navsim/data")