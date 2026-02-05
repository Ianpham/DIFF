"""
Data Contract System
====================
Defines the interface between datasets and encoders.

Think of this as a "specification sheet" - datasets declare what they provide,
encoders declare what they need, and the adapter matches them.
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Literal, Tuple, Dict, Any
from enum import Enum, auto

############### feature type ####################

class FeatureType(Enum):
    """All possible features a dataset might provide"""

    # raw sensors
    LIDAR_POINTS = auto() # raw 3D point cloud: List[(N_i, 3)]
    LIDAR_BEV = auto() # rasterized BEV: (2, H, W)
    CAMERA_IMAGES = auto() # raw rgb dict[str, (3, H, W)]
    CAMERA_BEV = auto() # BEV from cameras (C, H, W)

    # processed features
    BEV_LABELS = auto() # HD map sematic labels: Dict [str, (H, W)]
    VECTOR_MAP = auto() #  structured map data: VectormapFeatures

    # agent data
    AGENT_STATE = auto() # current kinematics (N, D)
    AGENT_HISTORY = auto() # trajectory history (N, T, D)
    AGENT_NEARBY = auto() # other agent (M,M, D)

    # mission
    ROUTE = auto() # navigation route : Dict
    GOAL = auto() # detination (2, )

    # groundtruth
    GT_TRAJECTORY = auto() # ground truth trajectory (N, T, D)

    # Difficulty/metadata
    DIFFICULTY = auto()        # Scene difficulty metrics: Dict
    

################## feature spec ####################

@dataclass
class FeatureSpec:
    """specification for a single feature"""
    feature_type: FeatureType  
    shape: Tuple  # Expected shape (use -1 for variable)
    dtype: str # 'float32', 'int64', etc.       
    optional: bool = False # can be missing
    fallback: Optional[FeatureType] = None # what to use if missing
    description: str = "" # description

    def __str__(self):
        fallback_str = f" (fallback: {self.fallback.name})" if self.fallback else ""
        optional_str = " [optional]" if self.optional else ""

        return (
            f"{self.feature_type.name}"
            f"shape = {self.shape}, dtype = {self.dtype}"
            f"{optional_str} {fallback_str}"
        )

###################### data constract #############################

@dataclass
class DataContract:
    """
    The contract b between dataset and encoder
    Menu dataset declare what they serve, encoder declare what they want to order
    """


    # available features 
    features: Set[FeatureType]

    # feature specification
    specs: Dict[FeatureType, FeatureSpec]

    # physical constraints
    max_batch_size: int 
    memory_footprint_mb: float

    # sematic info
    num_cameras: int = 0
    bev_channels: int = 0
    agent_state_dim: int = 5 # 5 or 7 depended on whether including acceleration
    history_length: int = 4 # number of timestep

    # flags for quick checks
    has_acceleration: bool = False # is acceleration in agent_state?
    has_nearby_agents: bool = False # multi-agent data available?
    has_vector_maps: bool = False # structured map avalable?

    # dataset metadata
    dataset_name: str = "Unknown"
    dataset_version: str = "1.0"

    def has(self, feature: FeatureType) -> bool:
        return feature in self.features
    # eheck id of feature where it is being called.
    # def has(self, feature: FeatureType) -> bool:
    #     print(f"\n=== DEBUG has() called with {feature.name} ===")
    #     print(f"Input feature: {feature} (id: {id(feature)}, module: {feature.__class__.__module__})")
        
    #     result = feature in self.features
    #     print(f"Result of 'feature in self.features': {result}")
        
    #     print(f"\nFeatures in contract:")
    #     for f in self.features:
    #         if f.value == feature.value:
    #             print(f"  Found matching value: {f} (id: {id(f)}, module: {f.__class__.__module__})")
    #             print(f"  f == feature: {f == feature}")
    #             print(f"  f is feature: {f is feature}")
        
    #     return result
    
    def get_spec(self, feature: FeatureType) -> Optional[FeatureSpec]:
        """get sepec for a featuers"""

        return self.specs.get(feature)
    
    def validate(self) -> Tuple[bool, list[str]]:
        """
        Validate contract consitancy

        Returns:
            (is_valid, list_of_errors)
        """

        errors = []

        # check feature has spec
        for feature in self.features:
            if feature not in self.specs:
                errors.append(f"Missing spec for {feature.name}")

        
        # check every spec's feature is in features set
        for feature in self.specs:
            if feature not in self.features:
                errors.append(f"spec for {feature.name} but feature not in set")

        # check fallback are available
        for spec in self.specs.values():
            if spec.fallback and spec.fallback not in self.features:
                errors.append(
                    f"{spec.feature_type.name} fallback {spec.fallback.name}"
                    f"not avaiable"
                )

        # check : semantic info matches specs
        if self.has(FeatureType.CAMERA_IMAGES):
            camera_spec = self.specs.get(FeatureType.CAMERA_IMAGES)
            if camera_spec and self.num_cameras == 0:
                errors.append("CAMERA_IMAGES available but num_cameras = 0")

        if self.has(FeatureType.BEV_LABELS):
            if self.bev_channels == 0:
                errors.append("BEV_LABELS availabel but bev_channels = 0")

        if self.has(FeatureType.AGENT_STATE):
            agent_spec = self.specs.get(FeatureType.AGENT_STATE)
            if agent_spec:
                expected_dim = agent_spec.shape[-1]
                if expected_dim != -1 and expected_dim != self.agent_state_dim:
                    errors.append(f"Agent state dim mismatch: spec says {expected_dim}, "
                                  f"contract says {self.agent_state_dim}")
                    

        is_valid = len(errors) == 0

        return is_valid, errors
    
    def __str__(self) -> str:
        lines = [
            f"=" * 70,
            f"Data contract: {self.dataset_name} v{self.dataset_version}" ,
            f"=" * 70,
            f"physical constraints:",
            f" Max batch size: {self.max_batch_size}",
            f" Memory footprint: {self.memory_footprint_mb: .1f} MB/sample",
            f"",
            f"Semantic Info:",
            f"  Cameras: {self.num_cameras}",
            f"  BEV channels: {self.bev_channels}",
            f"  Agent state dim: {self.agent_state_dim} ({'with' if self.has_acceleration else 'without'} acceleration)",
            f"  History length: {self.history_length}",
            f"",
            f"Available Features ({len(self.features)}):",
        ]

        for feature in sorted(self.features, key = lambda f: f.name):
            spec = self.specs.get(feature)

            if spec:
                lines.append(f" get {spec}")

            else:
                lines.append(f" get {feature.name} (no spec)")

        lines.append("=" * 70)

        return "\n".join(lines)
    

################### contract builder helper ###########################
class ContractBuilder:
    """Helper to build contracts incrementally"""

    def __init__(self, dataset_name: str = "Unknown"):
        self.dataset_name = dataset_name
        self.features = set()
        self.specs = {}
        self.max_batch_size = 16
        self.memory_footprint_mb = 50.0
        self.num_cameras = 0
        self.bev_channels = 0
        self.agent_state_dim = 5
        self.history_length = 5
        self.has_acceleration = False
        self.has_nearby_agents = False
        self.has_vector_maps = False
        self.dataset_version = "1.0"

    def add_feature(
            self, 
            feature_type: FeatureType,
            shape: Tuple,
            dtype: str = "float32",
            optional: bool = False,
            fallback: Optional[FeatureType] = None,
            description: str = ""
    ) -> 'ContractBuilder':
        """Add a feature to the contract"""
        self.features.add(feature_type)
        self.specs[feature_type] = FeatureSpec(
            feature_type= feature_type, 
            shape = shape,
            dtype = dtype,
            optional = optional,
            fallback= fallback,
            description= description
        )

        return self # for chaining
    

    def set_physical_limits(
            self,
            max_batch_size: int, 
            memory_footprint_mb: float,
    )-> 'ContractBuilder':
        """ set physical constraints"""
        self.max_batch_size = max_batch_size
        self.memory_footprint_mb = memory_footprint_mb

        return self
    
    def set_semantic_info(
        self,
        num_cameras: int = 0,
        bev_channels: int = 0,
        agent_state_dim: int = 5,
        history_length: int = 4,
        has_acceleration: bool = False,
        has_nearby_agents: bool = False,
        has_vector_maps: bool = False,
    ) -> 'ContractBuilder':
        """Set semantic information."""
        self.num_cameras = num_cameras
        self.bev_channels = bev_channels
        self.agent_state_dim = agent_state_dim
        self.history_length = history_length
        self.has_acceleration = has_acceleration
        self.has_nearby_agents = has_nearby_agents
        self.has_vector_maps = has_vector_maps
        return self
    
    def build(self) -> DataContract:
        """build the final contract"""
        contract = DataContract(
            features=self.features, 
            specs = self.specs,
            max_batch_size= self.max_batch_size,
            memory_footprint_mb=self.memory_footprint_mb,
            num_cameras= self.num_cameras,
            bev_channels= self.bev_channels, 
            agent_state_dim= self.agent_state_dim,
            history_length = self.history_length,
            has_acceleration= self.has_acceleration, 
            has_nearby_agents= self.has_nearby_agents,
            has_vector_maps= self.has_vector_maps,
            dataset_name=self.dataset_name, 
            dataset_version=self.dataset_version,
        )

        # validate
        is_valid, errors = contract.validate()

        if not is_valid:
            raise ValueError(
                f"invalid contract for {self.dataset_name}:\n" + 
                "\n".join(f" -{e}" for e in errors)
            )
        
        return contract
    
