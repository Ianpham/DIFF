"""
Base Dataset Class
==================
All NAVSIM datasets must inherit from this base class.
Enforces contract declaration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch.utils.data import Dataset

from navsim_utilize.contract import DataContract

class BaseNavsimDataset(Dataset, ABC):
    """
    Base class for all NAVSIM datasets.
    
    Key responsibilities:
    1. Declare contract (what features are available)
    2. Load and return samples
    3. Provide collate function
    
    Subclasses must implement:
    - _build_contract(): Declare what features this dataset provides
    - __getitem__(): Return a sample
    - __len__(): Return dataset size
    """
    
    def __init__(self):
        super().__init__()
        self._contract = None
    
    @abstractmethod
    def _build_contract(self) -> DataContract:
        """
        Build and return the dataset contract.
        
        This declares what features the dataset provides.
        Called during __init__.
        
        Returns:
            DataContract with full feature specifications
        """
        pass
    
    def get_contract(self) -> DataContract:
        """
        Get the dataset contract.
        
        Returns:
            DataContract declaring available features
        """
        if self._contract is None:
            self._contract = self._build_contract()
            
            # Validate contract
            is_valid, errors = self._contract.validate()
            if not is_valid:
                error_msg = (
                    f"Invalid contract for {self.__class__.__name__}:\n" +
                    "\n".join(f"  - {e}" for e in errors)
                )
                raise ValueError(error_msg)
        
        return self._contract
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dict containing features declared in contract
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    def collate_fn(self, batch: list) -> Dict[str, Any]:
        """
        Default collate function.
        
        Subclasses can override for custom collation logic.
        """
        return self._default_collate(batch)
    
    def _default_collate(self, batch: list) -> Dict[str, Any]:
        """
        Simple default collate that works for most cases.
        
        Override this in subclasses if you need custom logic.
        """
        if len(batch) == 0:
            return {}
        
        collated = {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            # Try to stack tensors
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    # Variable size - keep as list
                    collated[key] = values
            
            # Keep lists as-is
            elif isinstance(values[0], (list, tuple)):
                collated[key] = values
            
            # Keep dicts as list of dicts
            elif isinstance(values[0], dict):
                # Try to collate dict values
                try:
                    collated[key] = self._collate_dict_batch(values)
                except:
                    collated[key] = values
            
            # Other types - keep as list
            else:
                collated[key] = values
        
        return collated
    
    def _collate_dict_batch(self, dict_list: list) -> Dict:
        """Helper to collate a list of dicts."""
        if len(dict_list) == 0:
            return {}
        
        collated = {}
        keys = dict_list[0].keys()
        
        for key in keys:
            values = [d[key] for d in dict_list]
            
            if isinstance(values[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(values)
                except:
                    collated[key] = values
            else:
                collated[key] = values
        
        return collated
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  samples={len(self)},\n"
            f"  contract={self.get_contract().dataset_name}\n"
            f")"
        )