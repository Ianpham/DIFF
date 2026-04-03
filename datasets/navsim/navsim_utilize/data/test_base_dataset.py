"""
Test the base dataset class.
"""

from datasets.base import BaseNavsimDataset
from contract.data_contract import DataContract, ContractBuilder, FeatureType
import torch

class DummyDataset(BaseNavsimDataset):
    """Minimal dataset for testing."""
    
    def __init__(self, size: int = 10):
        super().__init__()
        self.size = size
    
    def _build_contract(self) -> DataContract:
        builder = ContractBuilder(dataset_name="DummyDataset")
        return (
            builder
            .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
            .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
            .set_physical_limits(max_batch_size=16, memory_footprint_mb=10.0)
            .build()
        )
    
    def __getitem__(self, idx: int):
        return {
            'lidar_bev': torch.randn(2, 200, 200),
            'agent_states': torch.randn(1, 5),
            'token': f'sample_{idx}',
        }
    
    def __len__(self):
        return self.size

def test_dummy_dataset():
    """Test the dummy dataset."""
    
    dataset = DummyDataset(size=5)
    
    # Check contract
    contract = dataset.get_contract()
    print(contract)
    
    assert contract.has(FeatureType.LIDAR_BEV)
    assert contract.agent_state_dim == 5
    
    # Check sample
    sample = dataset[0]
    assert 'lidar_bev' in sample
    assert sample['lidar_bev'].shape == (2, 200, 200)
    
    # Check collate
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    
    assert batch['lidar_bev'].shape == (2, 2, 200, 200)  # (B, C, H, W)
    assert batch['agent_states'].shape == (2, 1, 5)       # (B, N, D)
    
    print("\n  Dummy dataset test passed!")

if __name__ == "__main__":
    test_dummy_dataset()