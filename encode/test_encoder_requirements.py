"""
Test encoder requirements system.
"""

from requirements import (
    EncoderRequirements,
    StandardRequirements,
    RequirementValidator,
)
from datasets.navsim.navsim_utilize.contract.data_contract import FeatureType, DataContract, ContractBuilder
from datasets.navsim.navsim_utilize.data import NavsimDataset

def test_basic_requirements():
    """Test creating and checking basic requirements."""
    
    # Create a requirement
    req = EncoderRequirements(
        name="TestEncoder",
        required={FeatureType.LIDAR_BEV},
        min_agent_state_dim=5,
    )
    
    print(req)
    print()
    
    # Create a compatible contract
    contract = (
        ContractBuilder("TestDataset")
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .set_physical_limits(16, 50.0)
        .set_semantic_info(agent_state_dim=5)
        .build()
    )
    
    # Check compatibility
    compatible, errors, warnings = req.check_compatibility(contract)
    
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    
    assert compatible
    assert len(errors) == 0
    
    print("\n✓ Basic requirements test passed!")

def test_fallback_requirements():
    """Test requirements with fallback."""
    
    # PointPillars prefers raw points, falls back to BEV
    req = StandardRequirements.POINTPILLARS
    
    print(req)
    print()
    
    # Contract only has BEV (no raw points)
    contract = (
        ContractBuilder("BEVOnlyDataset")
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .set_physical_limits(16, 50.0)
        .build()
    )
    
    compatible, errors, warnings = req.check_compatibility(contract)
    
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    
    assert compatible  # Should be compatible via fallback
    assert len(errors) == 0
    assert len(warnings) > 0  # Should have warning about fallback
    
    print("\n✓ Fallback requirements test passed!")

def test_dimension_constraints():
    """Test dimension constraint checking."""
    
    # Require 7D agent state
    req = StandardRequirements.AGENT_FULL
    
    print(req)
    print()
    
    # Contract only has 5D
    contract = (
        ContractBuilder("5DDataset")
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 30, 5))
        .set_physical_limits(16, 50.0)
        .set_semantic_info(agent_state_dim=5, history_length=30)
        .build()
    )
    
    compatible, errors, warnings = req.check_compatibility(contract)
    
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    
    # Should be compatible (with warning) because fallback is allowed
    assert compatible
    assert len(errors) == 0
    assert len(warnings) > 0  # Warning about padding needed
    
    print("\n✓ Dimension constraint test passed!")

def test_validator_with_real_dataset():
    """Test validator with a real dataset."""
    
    # Create NavsimDataset
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,  # 5D only
    )
    
    contract = dataset.get_contract()
    print(contract)
    print()
    
    # # Create multi-modal requirements
    # requirements = StandardRequirements.create_multi_modal(
    #     use_lidar=True,
    #     use_camera=False,  # NavsimDataset doesn't have real cameras
    #     use_bev=True,
    #     use_vector_map=False,  # NavsimDataset doesn't have vector maps
    # )
    requirements = {
    'lidar': StandardRequirements.POINTPILLARS,
    'bev': StandardRequirements.BEV_SEMANTIC,
    'agent': StandardRequirements.AGENT_BASIC,  # ← Use BASIC instead of FULL
    }
    
    print("Requirements:")
    for name, req in requirements.items():
        print(f"\n{req}")
    print()
    
    # Validate
    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    
    validator.print_report(report)
    
    # Should be valid with some warnings
    assert is_valid
    
    print("\n✓ Real dataset validation test passed!")

def test_incompatible_requirements():
    """Test detecting incompatibilities."""
    
    # Dataset with minimal features
    contract = (
        ContractBuilder("MinimalDataset")
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .set_physical_limits(16, 50.0)
        .build()
    )
    
    # Require features that don't exist
    requirements = {
        'camera': StandardRequirements.MULTI_CAMERA,  # Not available
        'vector_map': StandardRequirements.VECTOR_MAP,  # Not available
    }
    
    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    
    validator.print_report(report)
    
    # Should be invalid
    assert not is_valid
    assert report['summary']['incompatible'] == 2
    
    print("\n✓ Incompatibility detection test passed!")

if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Basic Requirements")
    print("=" * 70)
    test_basic_requirements()
    
    print("\n" + "=" * 70)
    print("TEST 2: Fallback Requirements")
    print("=" * 70)
    test_fallback_requirements()
    
    print("\n" + "=" * 70)
    print("TEST 3: Dimension Constraints")
    print("=" * 70)
    test_dimension_constraints()
    
    print("\n" + "=" * 70)
    print("TEST 4: Real Dataset Validation")
    print("=" * 70)
    test_validator_with_real_dataset()
    
    print("\n" + "=" * 70)
    print("TEST 5: Incompatibility Detection")
    print("=" * 70)
    test_incompatible_requirements()
    
    print("\n" + "=" * 70)
    print("✓ ALL ENCODER REQUIREMENT TESTS PASSED!")
    print("=" * 70)