"""
Encoder Requirements System
============================
Encoders declare what they NEED from datasets.

Think of this as the "order form" - encoders specify what ingredients
they need, and the adapter checks if the dataset (kitchen) can provide them.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Any, Tuple, List, Dict
from enum import Enum

# make sure FeatureType being called in this directory, if not, the comparison is being called will return False.
from datasets.navsim.navsim_utilize.contract import FeatureType, DataContract

# Encoder Requirement Levels

class RequirementLevel(Enum):
    REQUIRED = "required"      # Must have, no fallback
    PREFERRED = "preferred"    # Nice to have, can work without
    OPTIONAL = "optional"      # Bonus feature

#single encoder requirement
@dataclass
class EncoderRequirements:
    # name for debugging
    name: str

    # required features (MUST have at least one)
    required: Set[FeatureType]

    # prefered features (nice to have)
    preferred : Set[FeatureType] = field(default_factory=set)

    
    optional: Set[FeatureType] = field(default_factory=set)

    # dimension constraints
    min_agent_state_dim: Optional[int] = None
    min_history_length: Optional[int] = None
    min_bev_channels: Optional[int] = None
    min_cameras: Optional[int] = None

    # Flags
    needs_acceleration: bool = False
    needs_multi_agent: bool = False
    needs_vector_maps: bool = False

     # Fallback strategy
    fallback_allowed: bool = True
    fallback_features: Dict[FeatureType, FeatureType] = field(default_factory=dict)

    def check_compatibility(
            self, 
            contract: DataContract
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if dataset contract satisfies this encoder's requirements.

        Return:
            (is_compatible, errors, warnings)

        Examples:
            req = EncoderRequirement(
            name = "PointPillars",
            required = {FeatureType.LIDAR_POINTS},
            fallback_features = {FeatureType.LIDAR_POINTS: FeatureType.LIDAR_BEV)})
            compatible, errors, wwarnings = req.check_compatibility(contract)
        """
        errors = []
        warnings = []

        # # check required features
        # has_required = any(
        #     contract.has(feat) for feat in self.required
        # )

        # if not has_required:
        #     # check if fallback available
        #     if self.fallback_allowed:
        #         has_fallback = False
        #         for req_feat, fallback_feat in self.fallback_features.items():
        #             if req_feat in self.required and contract.has(fallback_feat):
        #                 has_fallback = True
        #                 warnings.append(
        #                     f"{self.name}: Using fallback {fallback_feat.name}"
        #                     f"instead of {req_feat.name}"
        #                 )

        #                 break
        #         if not has_fallback:
        #             errors.append(
        #                 f"{self.name}: Missing required features {self.required}"
        #                 f"and no fallback available"
        #             )

        #     else:
        #         errors.append(
        #             f"{self.name}: Missing required features {self.required}"
        #         )

        # FIXED: Check if ALL required features are present
        missing_required = {feat for feat in self.required if not contract.has(feat)}
        
        if missing_required:
            # check if fallback available for ALL missing features
            if self.fallback_allowed:
                unresolved_features = set()
                for req_feat in missing_required:
                    fallback_feat = self.fallback_features.get(req_feat)
                    if fallback_feat and contract.has(fallback_feat):
                        warnings.append(
                            f"{self.name}: Using fallback {fallback_feat.name} "
                            f"instead of {req_feat.name}"
                        )
                    else:
                        unresolved_features.add(req_feat)
                
                if unresolved_features:
                    errors.append(
                        f"{self.name}: Missing required features {unresolved_features} "
                        f"and no fallback available"
                    )
            else:
                errors.append(
                    f"{self.name}: Missing required features {missing_required}"
                )
        # check preferred features
        for feat in self.preferred:
            if not contract.has(feat):
                warnings.append(
                    f"{self.name}: Missing preferred feature {feat.name}"
                )

        # check dimension constraints
        if self.min_agent_state_dim is not None:
            if contract.agent_state_dim is not None and contract.agent_state_dim < self.min_agent_state_dim:
                if self.fallback_allowed:
                    warnings.append(
                        f"{self.name}: Agent state dim {contract.agent_state_dim} < "
                        f"{self.min_agent_state_dim}, will pad with zeros"
                    )

                else:
                    errors.append(
                        f"{self.name}: Agent state dim {contract.agent_state_dim} < "
                        f"{self.min_agent_state_dim} (required)"
                    )

        if self.min_history_length is not None:
            if contract.history_length is not None and contract.history_length < self.min_history_length:
                errors.append(
                    f"{self.name}: History legnth {contract.history_length} < "
                    f"{self.min_history_length}"
                )

        if self.min_bev_channels is not None:
            if contract.bev_channels is not None and contract.bev_channels < self.min_bev_channels:
                errors.append(
                    f"{self.name}: BEV channels {contract.bev_channels} < "
                    f"{self.min_bev_channels}"
                )
        if self.min_cameras is not None:
            if contract.num_cameras is not None and contract.num_cameras < self.min_cameras:
                if self.fallback_allowed:
                    warnings.append(
                        f"{self.name}: Only {contract.num_cameras} cameras available "
                        f"(prefer {self.min_cameras})"
                    )
                else:
                    errors.append(
                        f"{self.name}: Only {contract.num_cameras} cameras available "
                        f"(need {self.min_cameras})"
                    )

        # Check flags
        if self.needs_acceleration and not contract.has_acceleration:
            if self.fallback_allowed:
                warnings.append(
                    f"{self.name}: Acceleration not available, will use zero padding"
                )
            else:
                errors.append(f"{self.name}: Acceleration required but not available")
        
        if self.needs_multi_agent and not contract.has_nearby_agents:
            if self.fallback_allowed:
                warnings.append(
                    f"{self.name}: Multi-agent data not available, "
                    f"will use single-agent mode"
                )
            else:
                errors.append(
                    f"{self.name}: Multi-agent data required but not available"
                )
        
        if self.needs_vector_maps and not contract.has_vector_maps:
            if self.fallback_allowed:
                warnings.append(
                    f"{self.name}: Vector maps not available"
                )
            else:
                errors.append(
                    f"{self.name}: Vector maps required but not available"
                )
        
        is_compatible = len(errors) == 0
        return is_compatible, errors, warnings
    
    def __str__(self) -> str:
        lines = [f"EncoderRequirements(name='{self.name}')"]
        lines.append(f"  Required: {[f.name for f in self.required]}")
        if self.preferred:
            lines.append(f"  Preferred: {[f.name for f in self.preferred]}")
        if self.optional:
            lines.append(f"  Optional: {[f.name for f in self.optional]}")
        
        constraints = []
        if self.min_agent_state_dim:
            constraints.append(f"agent_state_dim≥{self.min_agent_state_dim}")
        if self.min_history_length:
            constraints.append(f"history_length≥{self.min_history_length}")
        if self.min_bev_channels:
            constraints.append(f"bev_channels≥{self.min_bev_channels}")
        if self.min_cameras:
            constraints.append(f"cameras≥{self.min_cameras}")
        
        if constraints:
            lines.append(f"  Constraints: {', '.join(constraints)}")
        
        if self.fallback_allowed:
            lines.append(f"  Fallback: Allowed")
        
        return "\n".join(lines)


# Pre-defined Requirements (Common Encoders)
# =============================================================================

class StandardRequirements:
    """
    Pre-defined requirements for standard encoders.
    
    This is your "recipe book" - common encoder configurations.
    """
    

    # LiDAR Encoders
    # =========================================================================
    
    POINTPILLARS = EncoderRequirements(
        name="PointPillars",
        required={FeatureType.LIDAR_POINTS},
        preferred=set(),
        fallback_allowed=True,
        fallback_features={
            FeatureType.LIDAR_POINTS: FeatureType.LIDAR_BEV
        }
    )
    
    LIDAR_BEV = EncoderRequirements(
        name="LidarBEV",
        required={FeatureType.LIDAR_BEV},
        preferred=set(),
        fallback_allowed=False,
    )
    

    # Camera Encoders
    # =========================================================================
    
    MULTI_CAMERA = EncoderRequirements(
        name="MultiCamera",
        required={FeatureType.CAMERA_IMAGES},
        preferred=set(),
        min_cameras=1,
        fallback_allowed=True,
    )
    
    SINGLE_CAMERA = EncoderRequirements(
        name="SingleCamera",
        required={FeatureType.CAMERA_IMAGES},
        min_cameras=1,
        fallback_allowed=False,
    )
    

    # BEV Encoders
    # =========================================================================
    
    BEV_SEMANTIC = EncoderRequirements(
        name="BEVSemantic",
        required={FeatureType.BEV_LABELS},
        min_bev_channels=5,  # Minimum useful channels
        fallback_allowed=True,
    )
    
    BEV_FULL = EncoderRequirements(
        name="BEVFull",
        required={FeatureType.BEV_LABELS},
        min_bev_channels=12,  # All channels
        fallback_allowed=False,
    )
    

    # Agent Encoders
    # =========================================================================
    
    AGENT_BASIC = EncoderRequirements(
        name="AgentBasic",
        required={FeatureType.AGENT_STATE, FeatureType.AGENT_HISTORY},
        min_agent_state_dim=5,
        min_history_length=4,
        needs_acceleration=False,
        needs_multi_agent=False,
        fallback_allowed=True,
    )
    
    AGENT_FULL = EncoderRequirements(
        name="AgentFull",
        required={FeatureType.AGENT_STATE, FeatureType.AGENT_HISTORY},
        preferred={FeatureType.AGENT_NEARBY},
        min_agent_state_dim=7,
        min_history_length=10,
        needs_acceleration=True,
        needs_multi_agent=True,
        fallback_allowed=True,  # Can pad acceleration if missing
    )
    

    # Specialized Encoders
    # =========================================================================
    
    VECTOR_MAP = EncoderRequirements(
        name="VectorMap",
        required={FeatureType.VECTOR_MAP},
        needs_vector_maps=True,
        fallback_allowed=False,
    )
    
    GOAL_INTENT = EncoderRequirements(
        name="GoalIntent",
        required={FeatureType.ROUTE},
        optional={FeatureType.GOAL},
        fallback_allowed=True,
    )
    
    HISTORY_LSTM = EncoderRequirements(
        name="HistoryLSTM",
        required={FeatureType.AGENT_HISTORY},
        min_history_length=30,
        fallback_allowed=False,
    )

    # Combined Requirements (Multi-encoder systems)
    # =========================================================================
    
    @staticmethod
    def create_multi_modal(
        use_lidar: bool = True,
        use_camera: bool = True,
        use_bev: bool = True,
        use_vector_map: bool = False,
    ) -> Dict[str, EncoderRequirements]:
        """
        Create requirements for a multi-modal encoder system.
        
        Returns:
            Dict mapping encoder names to their requirements
        """
        requirements = {}
        
        if use_lidar:
            requirements['lidar'] = StandardRequirements.POINTPILLARS
        
        if use_camera:
            requirements['camera'] = StandardRequirements.MULTI_CAMERA
        
        if use_bev:
            requirements['bev'] = StandardRequirements.BEV_SEMANTIC
        
        requirements['agent'] = StandardRequirements.AGENT_FULL
        
        if use_vector_map:
            requirements['vector_map'] = StandardRequirements.VECTOR_MAP
        
        return requirements

# Requirement Validator
# =============================================================================

class RequirementValidator:
    """
    Validates a set of encoder requirements against a dataset contract.
    """
    
    def __init__(self, requirements: Dict[str, EncoderRequirements]):
        """
        Args:
            requirements: Dict mapping encoder names to their requirements
        """
        self.requirements = requirements
    
    def validate(
        self, 
        contract: DataContract
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate all requirements against contract.
        
        Returns:
            (is_valid, report)
            
        Report structure:
            {
                'valid': bool,
                'encoders': {
                    'encoder_name': {
                        'compatible': bool,
                        'errors': List[str],
                        'warnings': List[str],
                    }
                },
                'summary': {
                    'total': int,
                    'compatible': int,
                    'incompatible': int,
                }
            }
        """
        report = {
            'valid': True,
            'encoders': {},
            'summary': {
                'total': len(self.requirements),
                'compatible': 0,
                'incompatible': 0,
            }
        }
        
        for name, req in self.requirements.items():
            compatible, errors, warnings = req.check_compatibility(contract)
            
            report['encoders'][name] = {
                'compatible': compatible,
                'errors': errors,
                'warnings': warnings,
            }
            
            if compatible:
                report['summary']['compatible'] += 1
            else:
                report['summary']['incompatible'] += 1
                report['valid'] = False
        
        return report['valid'], report
    
    def print_report(self, report: Dict[str, Any]):
        """Pretty print validation report."""
        print("=" * 70)
        print("ENCODER REQUIREMENTS VALIDATION")
        print("=" * 70)
        
        summary = report['summary']
        print(f"Total encoders: {summary['total']}")
        print(f"Compatible: {summary['compatible']} ✓")
        print(f"Incompatible: {summary['incompatible']} ✗")
        print()
        
        for name, result in report['encoders'].items():
            status = "✓" if result['compatible'] else "✗"
            print(f"{status} {name:20s}", end="")
            
            if result['errors']:
                print(" [INCOMPATIBLE]")
                for error in result['errors']:
                    print(f"    ✗ {error}")
            elif result['warnings']:
                print(" [DEGRADED]")
                for warning in result['warnings']:
                    print(f"    ⚠ {warning}")
            else:
                print(" [OK]")
        
        print("=" * 70)