# """
# Compare capabilities of different datasets.
# """

# from data import NavsimDataset, EnhancedNavsimDataset
# from adapters import EncoderAdapter

# def compare_datasets():
#     """Compare capabilities."""
    
#     # Create both datasets
#     basic = NavsimDataset(
#         data_split="mini",
#         extract_labels=True,
#         compute_acceleration=False,
#     )
    
#     enhanced = EnhancedNavsimDataset(
#         data_split="mini",
#         extract_labels=True,
#         extract_vector_maps=False,
#     )
    
#     # Get contracts
#     basic_contract = basic.get_contract()
#     enhanced_contract = enhanced.get_contract()
    
#     print("=" * 70)
#     print("DATASET COMPARISON")
#     print("=" * 70)
    
#     print("\n📦 NavsimDataset (Basic):")
#     print(f"  Features: {len(basic_contract.features)}")
#     print(f"  Agent state dim: {basic_contract.agent_state_dim}D")
#     print(f"  Cameras: {basic_contract.num_cameras}")
#     print(f"  Has acceleration: {basic_contract.has_acceleration}")
#     print(f"  Multi-agent: {basic_contract.has_nearby_agents}")
#     print(f"  Max batch size: {basic_contract.max_batch_size}")
#     print(f"  Memory: {basic_contract.memory_footprint_mb} MB/sample")
    
#     print("\n📦 EnhancedNavsimDataset:")
#     print(f"  Features: {len(enhanced_contract.features)}")
#     print(f"  Agent state dim: {enhanced_contract.agent_state_dim}D")
#     print(f"  Cameras: {enhanced_contract.num_cameras}")
#     print(f"  Has acceleration: {enhanced_contract.has_acceleration}")
#     print(f"  Multi-agent: {enhanced_contract.has_nearby_agents}")
#     print(f"  Max batch size: {enhanced_contract.max_batch_size}")
#     print(f"  Memory: {enhanced_contract.memory_footprint_mb} MB/sample")
    
#     # Create adapters
#     print("\n" + "=" * 70)
#     print("ADAPTER CONFIGURATIONS")
#     print("=" * 70)
    
#     print("\n🔧 Basic Dataset Adapter:")
#     basic_adapter = EncoderAdapter(basic, mode="auto")
#     print(f"  Selected mode: {basic_adapter.mode}")
#     print(f"  Encoders: {list(basic_adapter.config.encoders.keys())}")
#     print(f"  Adaptations needed: {len(basic_adapter.config.adaptations)}")
    
#     print("\n🔧 Enhanced Dataset Adapter:")
#     enhanced_adapter = EncoderAdapter(enhanced, mode="auto")
#     print(f"  Selected mode: {enhanced_adapter.mode}")
#     print(f"  Encoders: {list(enhanced_adapter.config.encoders.keys())}")
#     print(f"  Adaptations needed: {len(enhanced_adapter.config.adaptations)}")
    
#     print("\n" + "=" * 70)

# if __name__ == "__main__":
#     compare_datasets()

"""
Compare all three dataset variants.
"""

from datas import NavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
from adapters import EncoderAdapter
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType

def compare_all():
    """Compare all dataset capabilities."""
    
    # Create all three datasets
    basic = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,
    )
    
    enhanced = EnhancedNavsimDataset(
        data_split="mini",
        extract_labels=True,
        extract_vector_maps=False,
    )
    
    phase = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=False,
        enable_phase_2=False,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    
    datasets = {
        'NavsimDataset (Basic)': basic,
        'EnhancedNavsimDataset': enhanced,
        'PhaseNavsimDataset (P0)': phase,
    }
    
    print("=" * 70)
    print("DATASET CAPABILITY COMPARISON")
    print("=" * 70)
    
    # Feature comparison
    features_to_check = [
        ('Raw LiDAR Points', FeatureType.LIDAR_POINTS),
        ('LiDAR BEV', FeatureType.LIDAR_BEV),
        ('Camera Images', FeatureType.CAMERA_IMAGES),
        ('BEV Labels', FeatureType.BEV_LABELS),
        ('Vector Maps', FeatureType.VECTOR_MAP),
        ('Nearby Agents', FeatureType.AGENT_NEARBY),
        ('Route Info', FeatureType.ROUTE),
    ]
    
    print("\n📊 Feature Availability:")
    print(f"{'Feature':<20} {'Basic':<10} {'Enhanced':<12} {'Phase':<10}")
    print("-" * 52)
    
    for feature_name, feature_type in features_to_check:
        row = f"{feature_name:<20}"
        for dataset in [basic, enhanced, phase]:
            contract = dataset.get_contract()
            has_it = "✓" if contract.has(feature_type) else "✗"
            row += f" {has_it:<10}"
        print(row)
    
    # Dimensional comparison
    print("\n📐 Dimensions & Constraints:")
    print(f"{'Property':<25} {'Basic':<10} {'Enhanced':<12} {'Phase':<10}")
    print("-" * 57)
    
    properties = [
        ('Agent State Dim', lambda c: f"{c.agent_state_dim}D"),
        ('History Length', lambda c: str(c.history_length)),
        ('Num Cameras', lambda c: str(c.num_cameras)),
        ('BEV Channels', lambda c: str(c.bev_channels)),
        ('Has Acceleration', lambda c: "✓" if c.has_acceleration else "✗"),
        ('Multi-Agent', lambda c: "✓" if c.has_nearby_agents else "✗"),
        ('Max Batch Size', lambda c: str(c.max_batch_size)),
        ('Memory (MB)', lambda c: f"{c.memory_footprint_mb:.0f}"),
    ]
    
    for prop_name, prop_fn in properties:
        row = f"{prop_name:<25}"
        for dataset in [basic, enhanced, phase]:
            contract = dataset.get_contract()
            value = prop_fn(contract)
            row += f" {value:<10}"
        print(row)
    
    # Adapter comparison
    print("\n" + "=" * 70)
    print("ADAPTER AUTO-CONFIGURATION")
    print("=" * 70)
    
    for name, dataset in datasets.items():
        print(f"\n{name}:")
        adapter = EncoderAdapter(dataset, mode="auto")
        print(f"  Selected mode: {adapter.mode}")
        print(f"  Encoders: {list(adapter.config.encoders.keys())}")
        print(f"  Adaptations: {len(adapter.config.adaptations)}")
        if adapter.config.adaptations:
            for adapt in adapter.config.adaptations:
                print(f"    - {adapt}")

if __name__ == "__main__":
    compare_all()