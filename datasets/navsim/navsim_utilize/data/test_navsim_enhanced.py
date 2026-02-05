"""
Test EnhancedNavsimDataset with adapter.
"""

from data import EnhancedNavsimDataset
from adapters import EncoderAdapter
from datasets.navsim.navsim_utilize.contract import DataContract, ContractBuilder, FeatureType
from navsim.common.dataclasses import Scene, Frame, SceneFilter, SensorConfig
def test_enhanced_dataset():
    """Test EnhancedNavsimDataset."""
    
    dataset = EnhancedNavsimDataset(
        data_split="mini",
        extract_labels=True,
        extract_route_info=True,
        extract_vector_maps=False,  # Disable for speed
    )
    
    # Check contract
    contract = dataset.get_contract()
    print(contract)
    
    # Verify features
    assert contract.has(FeatureType.LIDAR_POINTS)  # ✓ Raw points
    assert contract.has(FeatureType.LIDAR_BEV)     # ✓ BEV
    assert contract.has(FeatureType.CAMERA_IMAGES) # ✓ 8 cameras
    assert contract.has(FeatureType.BEV_LABELS)    # ✓ Labels
    assert contract.has(FeatureType.AGENT_NEARBY)  # ✓ Multi-agent
    
    assert contract.agent_state_dim == 7           # ✓ Full 7D
    assert contract.has_acceleration               # ✓ Has acceleration
    assert contract.has_nearby_agents              # ✓ Multi-agent
    assert contract.num_cameras == 8               # ✓ All cameras
    
    print("\n✓ EnhancedNavsimDataset contract test passed!")

def test_enhanced_with_adapter():
    """Test adapter with EnhancedNavsimDataset."""
    
    dataset = EnhancedNavsimDataset(
        data_split="mini",
        extract_labels=True,
        extract_route_info=True,
        extract_vector_maps=False,
    )
    
    # Create adapter
    adapter = EncoderAdapter(dataset, mode="full")
    adapter.print_summary()
    
    # Check configuration
    assert 'lidar' in adapter.config.encoders
    assert adapter.config.encoders['lidar'].encoder_type == 'PointPillars'  # Should use raw points
    
    assert 'camera' in adapter.config.encoders
    assert adapter.config.encoders['camera'].encoder_type == 'MultiCamera'
    
    assert 'agent' in adapter.config.encoders
    assert not adapter.config.encoders['agent'].needs_padding  # Already 7D!
    
    print("\n✓ Adapter with EnhancedNavsimDataset test passed!")

def test_sample():
    """Test getting a sample."""
    
    """Test loading a single sample."""
    print("\n=== Testing Single Sample ===")
    
    print("\n=== Testing Single Sample ===")
    
    # Create sensor config with all sensors enabled
    sensor_config = SensorConfig.build_all_sensors(include=True)
    
    # Debug: Print sensor config
    print(f"Sensor config created: {sensor_config}")
    print(f"Sensors at iteration 0: {sensor_config.get_sensors_at_iteration(0)}")
    print(f"Sensors at iteration 3: {sensor_config.get_sensors_at_iteration(3)}")
    
    # Create dataset
    dataset = EnhancedNavsimDataset(
        data_split="mini",
        extract_labels=True,
        extract_route_info=True,
        extract_vector_maps=False,  # Disable for speed
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    
    print("\nSample keys:", sample.keys())
    print(f"LiDAR points shape: {sample['lidar_original'].shape}")
    print(f"Camera images: {len(sample['camera_images'])} cameras")
    print(f"Agent states shape: {sample['agent_states'].shape}")
    print(f"Nearby agents shape: {sample['nearby_agents'].shape}")
    
    # Verify shapes
    assert sample['agent_states'].shape[-1] == 7  # Full 7D
    assert len(sample['camera_images']) == 8      # All cameras
    assert sample['nearby_agents'].shape[1] == 10 # Up to 10 agents
    
    print("\n✓ Sample test passed!")

if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Enhanced Dataset Contract")
    print("=" * 70)
    test_enhanced_dataset()
    
    print("\n" + "=" * 70)
    print("TEST 2: Adapter with Enhanced Dataset")
    print("=" * 70)
    test_enhanced_with_adapter()
    
    print("\n" + "=" * 70)
    print("TEST 3: Sample Extraction")
    print("=" * 70)
    test_sample()
    
    print("\n" + "=" * 70)
    print("✓ ALL ENHANCED DATASET TESTS PASSED!")
    print("=" * 70)