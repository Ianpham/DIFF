"""
Test the refactored NavsimDataset.
"""

from datasets import NavsimDataset
from contract import FeatureType

def test_navsim_dataset():
    """Test NavsimDataset with contract system."""
    
    # Create dataset WITHOUT acceleration
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,
    )
    
    # Check contract
    contract = dataset.get_contract()
    print(contract)
    
    assert contract.has(FeatureType.LIDAR_BEV)
    assert contract.has(FeatureType.BEV_LABELS)
    assert contract.agent_state_dim == 5
    assert not contract.has_acceleration
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Agent states shape: {sample['agent_states'].shape}")
    
    assert sample['agent_states'].shape == (1, 5)  # No acceleration
    
    print("\n✓ NavsimDataset (no acceleration) test passed!")
    
    # Now try WITH acceleration
    dataset2 = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=True,  # Enable acceleration
    )
    
    contract2 = dataset2.get_contract()
    print(f"\n{contract2}")
    
    assert contract2.agent_state_dim == 7
    assert contract2.has_acceleration
    
    sample2 = dataset2[0]
    print(f"\nSample2 agent states shape: {sample2['agent_states'].shape}")
    assert sample2['agent_states'].shape == (1, 7)  # With acceleration
    
    print("\n✓ NavsimDataset (with acceleration) test passed!")

if __name__ == "__main__":
    test_navsim_dataset()