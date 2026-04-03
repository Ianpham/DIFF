"""
Test the encoder adapter system.
"""
from datasets.navsim.navsim_utilize.contract import DataContract, FeatureType
from datasets.navsim.navsim_utilize.data import NavsimDataset
from adapters import EncoderAdapter, create_adapter, quick_build
import torch

def test_basic_adapter():
    """Test basic adapter creation."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,
    )
    
    # DEBUG: Check the contract
    contract = dataset.get_contract()
    print(f"\n=== CONTRACT DEBUG ===")
    print(f"Contract type: {type(contract)}")
    print(f"Contract.__dict__: {contract.__dict__}")
    
    # Try different ways to check features
    print(f"\nTrying contract.has():")
    print(f"  LIDAR_BEV: {contract.has(FeatureType.LIDAR_BEV)}")
    print(f"  BEV_LABELS: {contract.has(FeatureType.BEV_LABELS)}")
    print(f"  AGENT_STATE: {contract.has(FeatureType.AGENT_STATE)}")
    
    # Check if there's a features dict
    if hasattr(contract, 'features'):
        print(f"\nContract.features: {contract.features}")
    
    if hasattr(contract, '_features'):
        print(f"\nContract._features: {contract._features}")
    
    print("=" * 70)

    # print(f"\n=== REQUIREMENT CHECK DEBUG ===")
    # contract = dataset.get_contract()

    # # Manually test what check_compatibility does
    # from encode.requirements import StandardRequirements
    # agent_req = StandardRequirements.agent_basic()

    # print(f"Agent Basic Requirements:")
    # print(f"  Required features: {agent_req.required}")
    # print(f"\nChecking each required feature:")
    # for feat in agent_req.required:
    #     has_it = contract.has(feat)
    #     in_set = feat in contract.features
    #     print(f"  {feat.name}:")
    #     print(f"    contract.has(): {has_it}")
    #     print(f"    in contract.features: {in_set}")
    #     print(f"    Feature object id: {id(feat)}")

    # print(f"\nhas_required = any(...): {any(contract.has(feat) for feat in agent_req.required)}")
        
    adapter = EncoderAdapter(dataset, mode="auto")
    adapter.print_summary()
def test_adapter_modes():
    """Test different adapter modes."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,
    )
    
    # Test each mode
    for mode in ['minimal', 'efficient', 'full', 'auto']:
        print(f"\n{'='*70}")
        print(f"Testing mode: {mode.upper()}")
        print('='*70)
        
        adapter = EncoderAdapter(dataset, mode=mode)
        adapter.print_summary()
        
        # Check that configuration exists
        assert len(adapter.config.encoders) > 0
        
        # Auto should select a mode
        if mode == 'auto':
            assert adapter.mode in ['minimal', 'efficient', 'full']
            print(f"Auto-selected mode: {adapter.mode}")
    
    print("\n  Adapter modes test passed!")

def test_batch_adaptation():
    """Test batch adaptation (padding, etc.)."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=False,  # 5D states
    )
    
    adapter = EncoderAdapter(dataset, mode="full")
    
    # Get a sample
    sample = dataset[0]
    
    # Create fake batch
    batch = {
        'lidar_bev': sample['lidar_bev'].unsqueeze(0),
        'labels': {k: v.unsqueeze(0) for k, v in sample['labels'].items()},
        'agent_states': sample['agent_states'].unsqueeze(0),  # (1, 1, 5)
    }
    
    print("\nOriginal batch:")
    print(f"  Agent states shape: {batch['agent_states'].shape}")
    print(f"  Agent states dim: {batch['agent_states'].shape[-1]}")
    
    # Adapt batch
    adapted = adapter.adapt_batch(batch)
    
    print("\nAdapted batch:")
    print(f"  Agent states shape: {adapted['agent'].shape}")
    print(f"  Agent states dim: {adapted['agent'].shape[-1]}")
    
    # Check padding
    assert adapted['agent'].shape[-1] == 7, "Should be padded to 7D"
    
    # Check that acceleration is zero
    ax = adapted['agent'][0, 0, 4]
    ay = adapted['agent'][0, 0, 5]
    assert ax == 0.0 and ay == 0.0, "Padded acceleration should be zero"
    
    print("\n  Batch adaptation test passed!")

def test_with_acceleration():
    """Test adapter with dataset that has acceleration."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
        compute_acceleration=True,  # 7D states
    )
    
    adapter = EncoderAdapter(dataset, mode="full")
    adapter.print_summary()
    
    # Check that NO padding is needed
    agent_config = adapter.config.encoders['agent']
    assert not agent_config.needs_padding, "Should NOT need padding (already 7D)"
    
    # Get sample and adapt
    sample = dataset[0]
    batch = {
        'agent_states': sample['agent_states'].unsqueeze(0),  # (1, 1, 7)
    }
    
    adapted = adapter.adapt_batch(batch)
    
    # Check that it's still 7D (no padding)
    assert adapted['agent'].shape[-1] == 7
    
    print("\n  Adapter with acceleration test passed!")

def test_quick_build():
    """Test quick_build convenience function."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
    )
    
    # Quick build (NOTE: This will fail if encoder modules aren't available)
    # For now, just test the adapter part
    adapter = create_adapter(dataset, mode="auto")
    
    # Check that it works
    assert len(adapter.config.encoders) > 0
    
    print("\n  Quick build test passed!")

def test_optimal_batch_size():
    """Test optimal batch size calculation."""
    
    dataset = NavsimDataset(
        data_split="mini",
        extract_labels=True,
    )
    
    adapter = EncoderAdapter(dataset, mode="auto")
    
    batch_size = adapter.get_optimal_batch_size()
    print(f"\nOptimal batch size: {batch_size}")
    
    assert batch_size > 0
    assert batch_size <= dataset.get_contract().max_batch_size
    
    print("\n  Optimal batch size test passed!")
    

if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Basic Adapter")
    print("=" * 70)
    test_basic_adapter()
    
    print("\n" + "=" * 70)
    print("TEST 2: Adapter Modes")
    print("=" * 70)
    test_adapter_modes()
    
    print("\n" + "=" * 70)
    print("TEST 3: Batch Adaptation")
    print("=" * 70)
    test_batch_adaptation()
    
    print("\n" + "=" * 70)
    print("TEST 4: With Acceleration")
    print("=" * 70)
    test_with_acceleration()
    
    print("\n" + "=" * 70)
    print("TEST 5: Quick Build")
    print("=" * 70)
    test_quick_build()
    
    print("\n" + "=" * 70)
    print("TEST 6: Optimal Batch Size")
    print("=" * 70)
    test_optimal_batch_size()
    
    print("\n" + "=" * 70)
    print("  ALL ADAPTER TESTS PASSED!")
    print("=" * 70)