"""
Test PhaseNavsimDataset with adapter.
"""

from data import PhaseNavsimDataset
from adapters import EncoderAdapter

def test_phase_dataset_phase0_only():
    """Test Phase dataset with only Phase 0."""
    
    dataset = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=False,
        enable_phase_2=False,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    
    # Check contract
    contract = dataset.get_contract()
    print(contract)
    
    # Verify Phase 0 features
    assert contract.has_acceleration
    assert contract.has_nearby_agents
    assert contract.agent_state_dim == 7
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    assert 'phase_0' in sample
    assert 'phase_1' not in sample
    assert 'phase_2' not in sample
    
    print(f"Phase 0 keys: {sample['phase_0'].keys()}")
    
    print("\n  Phase 0 only test passed!")

def test_phase_dataset_all_phases():
    """Test Phase dataset with all phases enabled."""
    
    dataset = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=True,
        enable_phase_2=True,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    
    assert 'phase_0' in sample
    assert 'phase_1' in sample
    assert 'phase_2' in sample
    
    print(f"Phase 0 keys: {list(sample['phase_0'].keys())[:5]}...")
    print(f"Phase 1 keys: {sample['phase_1'].keys()}")
    print(f"Phase 2 keys: {sample['phase_2'].keys()}")
    
    print("\n  All phases test passed!")

def test_phase_dataset_with_adapter():
    """Test adapter with Phase dataset."""
    
    dataset = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=False,
        enable_phase_2=False,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    
    # Create adapter
    adapter = EncoderAdapter(dataset, mode="full")
    adapter.print_summary()
    
    # Check configuration
    assert 'lidar' in adapter.config.encoders
    assert 'camera' in adapter.config.encoders
    assert 'agent' in adapter.config.encoders
    
    # Agent should NOT need padding (already 7D)
    assert not adapter.config.encoders['agent'].needs_padding
    
    print("\n  Adapter with Phase dataset test passed!")

def test_curriculum_learning():
    """Test curriculum learning progression."""
    
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING SIMULATION")
    print("=" * 70)
    
    # Stage 1: Start with Phase 0
    print("\n📚 Stage 1: Phase 0 (Core Features)")
    dataset_stage1 = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=False,
        enable_phase_2=False,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    adapter_stage1 = EncoderAdapter(dataset_stage1, mode="full")
    print(f"  Encoders: {list(adapter_stage1.config.encoders.keys())}")
    print(f"  Mode: {adapter_stage1.mode}")
    
    # Stage 2: Add Phase 1
    print("\n📚 Stage 2: Phase 0 + Phase 1 (Pretrained Models)")
    dataset_stage2 = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=True,
        enable_phase_2=False,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    sample = dataset_stage2[0]
    print(f"  Available phases: {[k for k in sample.keys() if 'phase' in k]}")
    
    # Stage 3: Add Phase 2
    print("\n📚 Stage 3: All Phases (Full Model)")
    dataset_stage3 = PhaseNavsimDataset(
        data_split="mini",
        enable_phase_0=True,
        enable_phase_1=True,
        enable_phase_2=True,
        extract_labels=True,
        extract_vector_maps=False,
        use_cache=False,
    )
    sample = dataset_stage3[0]
    print(f"  Available phases: {[k for k in sample.keys() if 'phase' in k]}")
    
    print("\n  Curriculum learning test passed!")

if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: Phase 0 Only")
    print("=" * 70)
    test_phase_dataset_phase0_only()
    
    print("\n" + "=" * 70)
    print("TEST 2: All Phases")
    print("=" * 70)
    test_phase_dataset_all_phases()
    
    print("\n" + "=" * 70)
    print("TEST 3: Adapter with Phase Dataset")
    print("=" * 70)
    test_phase_dataset_with_adapter()
    
    print("\n" + "=" * 70)
    print("TEST 4: Curriculum Learning")
    print("=" * 70)
    test_curriculum_learning()
    
    print("\n" + "=" * 70)
    print("  ALL PHASE DATASET TESTS PASSED!")
    print("=" * 70)