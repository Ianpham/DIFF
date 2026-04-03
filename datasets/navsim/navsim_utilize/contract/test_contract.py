"""
Test the contract system before refactoring datasets.
"""

from data_contract import FeatureType, DataContract, ContractBuilder

def test_basic_contract():
    """Test creating a simple contract."""
    
    builder = ContractBuilder(dataset_name="TestDataset")
    
    contract = (
        builder
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200), dtype="float32")
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5), dtype="float32")
        .set_physical_limits(max_batch_size=16, memory_footprint_mb=50.0)
        .set_semantic_info(agent_state_dim=5, history_length=4)
        .build()
    )
    
    print(contract)
    
    # Test queries
    assert contract.has(FeatureType.LIDAR_BEV)
    assert not contract.has(FeatureType.CAMERA_IMAGES)
    assert contract.agent_state_dim == 5
    
    print("\n  Basic contract test passed!")

def test_contract_with_fallback():
    """Test contract with fallback features."""
    
    builder = ContractBuilder(dataset_name="FlexibleDataset")
    
    contract = (
        builder
        .add_feature(
            FeatureType.LIDAR_POINTS,
            shape=(-1, 3),
            dtype="float32",
            fallback=FeatureType.LIDAR_BEV,
        )
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200), dtype="float32")
        .set_physical_limits(max_batch_size=8, memory_footprint_mb=100.0)
        .build()
    )
    
    print(contract)
    
    # Check fallback
    spec = contract.get_spec(FeatureType.LIDAR_POINTS)
    assert spec.fallback == FeatureType.LIDAR_BEV
    
    print("\n  Fallback contract test passed!")

def test_invalid_contract():
    """Test that invalid contracts are caught."""
    
    builder = ContractBuilder(dataset_name="InvalidDataset")
    
    try:
        contract = (
            builder
            .add_feature(
                FeatureType.LIDAR_POINTS,
                shape=(-1, 3),
                dtype="float32",
                fallback=FeatureType.CAMERA_IMAGES,  # Not available!
            )
            .build()
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly caught invalid contract: {e}")

if __name__ == "__main__":
    test_basic_contract()
    print("\n" + "=" * 70 + "\n")
    test_contract_with_fallback()
    print("\n" + "=" * 70 + "\n")
    test_invalid_contract()
    print("\n  All contract tests passed!")