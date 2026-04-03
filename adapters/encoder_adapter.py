"""
Encoder Adapter System
======================
Automatically configure and adapt encoders based on dataset capabilities.

This is the "intelligent middleware" that:
1. Analyzes dataset contract
2. Checks encoder requirements
3. Selects optimal encoders
4. Handles runtime adaptations (padding, fallbacks, etc.)

"""

from typing import Dict, Any, Optional, Literal, Tuple, List
import torch
import torch.nn as nn
import warnings
from dataclasses import dataclass, field

from datasets.navsim.navsim_utilize.contract import DataContract, FeatureType

from encode.requirements import (
    EncoderRequirements,
    StandardRequirements,
    RequirementValidator,     
)

from datasets.navsim.navsim_utilize.data import BaseNavsimDataset, EnhancedNavsimDataset, PhaseNavsimDataset
#### Encoder configuration

@dataclass
class EncoderConfig:
    name: str                           # Encoder name (e.g., 'lidar', 'camera')
    encoder_type: str                   # Type (e.g., 'PointPillars', 'LidarBEV')
    input_key: str                      # Key in batch dict
    requirements: EncoderRequirements   # What this encoder needs

    # Adaptation flags
    needs_padding: bool = False         # Need to pad dimensions?
    needs_fallback: bool = False        # Using fallback feature?
    fallback_key: Optional[str] = None  # Fallback input key
    
    # Build parameters
    build_params: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        lines = [f"EncoderConfig(name='{self.name}', type='{self.encoder_type}')"]
        lines.append(f"  Input: {self.input_key}")
        if self.needs_fallback:
            lines.append(f"  Fallback: {self.fallback_key}")
        if self.needs_padding:
            lines.append(f"  Adaptation: Dimension padding required")
        return "\n".join(lines)

@dataclass
class AdapterConfiguration:
    """Complete adapter configuration."""
    mode: str                           # 'auto', 'minimal', 'full'
    encoders: Dict[str, EncoderConfig]  # Configured encoders
    adaptations: List[str]              # List of adaptations applied
    warnings: List[str]                 # Warnings during config
    physical_limits: Dict[str, Any]     # Physical constrain

    def summary(self) -> str:
        """summary"""
        lines = [
            "=" * 70,
            f"ADAPTER CONFIGURATION - {self.mode.upper()} MODE",
            "=" * 70,
            f"Configured Encoders: {len(self.encoders)}",
        ]

        for name, config in self.encoders.items():
            lines.append(f" {name:12s}: {config.encoder_type:15s} <- {config.input_key}")
            if config.needs_fallback:
                lines.append(f"    Fallback: {config.fallback_key}")
            
        if self.adaptations:
            lines.append("\n runtiume adaptations")
            for adapt in self.adaptations:
                lines.append(f" {adapt}")
            
        if self.warnings:
            lines.append("\n Warnings:")
            for warn in self.warnings:
                lines.append(f" {warn}")

        lines.append(f"\n Physical Limits: ")
        lines.append(f" max batch size: {self.physical_limits.get('max_batch_size', 'N/A')}")
        lines.append(f"  Memory/sample: {self.physical_limits.get('memory_mb', 'N/A')} MB")

        lines.append("=" * 70)
        return "\n".join(lines)
    
######## adapter ################

class EncoderAdapter:
    """
    Intelligent adapter between datasets and encoders.
    
    Usage:
        >>> dataset = NavsimDataset("mini")
        >>> adapter = EncoderAdapter(dataset, mode="auto")
        >>> encoders = adapter.build_encoders(hidden_size=768)
        >>> 
        >>> for batch in dataloader:
        >>>     adapted = adapter.adapt_batch(batch)
        >>>     features = encoders['lidar'](adapted['lidar'])
    """
    
    def __init__(
            self,
            dataset: BaseNavsimDataset,
            mode: Literal["auto", "minimal", "efficient", "full"] = "auto",
            allow_degradation: bool = True,
            prefer_quality: bool = True, # prefer quality over speed
    ):
        
        """
        Initialize adapter.
        
        Args:
            dataset: Dataset to adapt
            mode: Configuration mode
                - 'auto': Automatically select best mode
                - 'minimal': Use minimum required encoders
                - 'efficient': Balance quality and speed
                - 'full': Use all available features
            allow_degradation: Allow fallbacks and adaptations
            prefer_quality: When choosing between options, prefer quality
        """
        self.dataset = dataset
        self.contract = dataset.get_contract()
        self.mode = mode if mode != "auto" else self._auto_select_mode()
        self.allow_degradation = allow_degradation
        self.prefer_quality = prefer_quality

        # compute configuration
        self.config = self._compute_configuration()

        # valid
        self._validate_configuration()


    def _auto_select_mode(self) ->  str:
        """
        Automatically select best mode based on dataset capabilities.
        
        Scoring:
        - Raw sensors: +2 points each
        - Processed features: +1 point each
        - Physical constraints: -1 per constraint
        
        Score >= 9: full
        Score >= 5: efficient
        Score < 5: minimal
        """

        score = 0

        # raw sensors (high value)
        if self.contract.has(FeatureType.LIDAR_POINTS):
            score += 2

        if self.contract.has(FeatureType.CAMERA_IMAGES):
            score += 2
            if self.contract.num_cameras >= 8:
                score += 1


        #processed features (medium value)
        if self.contract.has(FeatureType.LIDAR_POINTS):
            score += 1

        if self.contract.has(FeatureType.BEV_LABELS):
            score += 1
            if self.contract.bev_channels >= 12:
                score += 1

        if self.contract.has(FeatureType.VECTOR_MAP):
            score += 1

        
        # agent features
        if self.contract.has_acceleration:
            score += 1

        if self.contract.has_nearby_agents:
            score += 2

        
        # physical constraints (penalties)
        if self.contract.max_batch_size < 8:
            score -= 1
        if self.contract.memory_footprint_mb > 100:
            score -= 1

        # select mode
        if score >= 9:
            return "full"
        
        if score >= 5:
            return "efficient"
        
        else:
            return "minimal"
        
    def _compute_configuration(self) -> AdapterConfiguration:
        """
        Compute optimal encoder configuration.
        
        Analyzes dataset contract and mode to determine which encoders to use,
        what adaptations are needed, and any warnings to surface.
        """
        encoders = {}
        adaptations = []
        warnings_list = []

        
        # LIDAR ENCODER SELECTION
        
        lidar_config = self._configure_lidar_encoder()
        if lidar_config:
            encoders['lidar'] = lidar_config['encoder']
            adaptations.extend(lidar_config.get('adaptations', []))
            warnings_list.extend(lidar_config.get('warnings', []))

        
        # CAMERA ENCODER SELECTION
        
        camera_config = self._configure_camera_encoder()
        if camera_config:
            encoders['camera'] = camera_config['encoder']
            warnings_list.extend(camera_config.get('warnings', []))

        
        # BEV LABEL ENCODER
        
        if self.contract.has(FeatureType.BEV_LABELS):
            encoders['bev'] = EncoderConfig(
                name='bev',
                encoder_type='BEV',
                input_key='labels',
                requirements=StandardRequirements.BEV_SEMANTIC,
                build_params={
                    'num_channels': self.contract.bev_channels,
                }
            )

        
        # AGENT ENCODER SELECTION
        
        agent_config = self._configure_agent_encoder()
        encoders['agent'] = agent_config['encoder']
        if agent_config.get('needs_padding'):
            adaptations.append(
                "Agent: Padding 5D states to 7D (zero acceleration)"
            )

        
        # OPTIONAL ENCODERS
        
        
        # Vector map (full mode only)
        if self.contract.has(FeatureType.VECTOR_MAP) and self.mode == 'full':
            encoders['vector_map'] = EncoderConfig(
                name='vector_map',
                encoder_type='VectorMap',
                input_key='vector_map',
                requirements=StandardRequirements.VECTOR_MAP,
            )
        
        # Goal/route (full and efficient modes)
        if self.contract.has(FeatureType.ROUTE) and self.mode in ['full', 'efficient']:
            encoders['goal'] = EncoderConfig(
                name='goal',
                encoder_type='GoalIntent',
                input_key='route_info',
                requirements=StandardRequirements.GOAL_INTENT,
            )

        
        # BUILD FINAL CONFIGURATION
        
        return AdapterConfiguration(
            mode=self.mode,
            encoders=encoders,
            adaptations=adaptations,
            warnings=warnings_list,
            physical_limits={
                'max_batch_size': self.contract.max_batch_size,
                'memory_mb': self.contract.memory_footprint_mb,
            }
        )


    def _configure_lidar_encoder(self) -> Optional[Dict[str, Any]]:
        """
        Configure LiDAR encoder based on available data and mode.
        
        Priority:
        1. Raw point cloud -> PointPillars (full/efficient modes)
        2. Preprocessed BEV -> LidarBEV encoder (ALL modes)
        3. None if no LiDAR data available
        
        Returns:
            Dict with 'encoder', 'warnings', 'adaptations' or None
        """
        has_points = self.contract.has(FeatureType.LIDAR_POINTS)
        has_bev = self.contract.has(FeatureType.LIDAR_BEV)
        
        if not has_points and not has_bev:
            return None
        
        warnings = []
        
        # Prefer raw points in full/efficient modes
        if has_points and self.mode in ['full', 'efficient']:
            encoder = EncoderConfig(
                name='lidar',
                encoder_type='PointPillars',
                input_key='lidar_original',
                requirements=StandardRequirements.POINTPILLARS,
                build_params={
                    'voxel_size': (0.25, 0.25, 4.0),
                    'point_cloud_range': (-50, -50, -2, 50, 50, 2),
                    'max_points_per_voxel': 32,
                    'max_voxels': 16000,
                }
            )
            return {'encoder': encoder, 'warnings': warnings}
        
        # Use BEV if available (works in ALL modes including minimal)
        if has_bev:
            encoder = EncoderConfig(
                name='lidar',
                encoder_type='LidarBEV',
                input_key='lidar_bev',
                requirements=StandardRequirements.LIDAR_BEV,
                needs_fallback=has_points,  # True if we're falling back from points
            )
            
            # Add appropriate warnings based on mode
            if has_points:
                if self.mode == 'minimal':
                    warnings.append(
                        "LiDAR: Raw points available but using BEV in minimal mode"
                    )
                elif self.mode in ['full', 'efficient']:
                    warnings.append(
                        "LiDAR: Using BEV encoder (PointPillars available but not selected)"
                    )
            
            return {'encoder': encoder, 'warnings': warnings}
        
        # Should never reach here if has_points or has_bev is True
        return None


    def _configure_camera_encoder(self) -> Optional[Dict[str, Any]]:
        """
        Configure camera encoder based on available data and mode.
        
        Returns:
            Dict with 'encoder', 'warnings' or None
        """
        has_images = self.contract.has(FeatureType.CAMERA_IMAGES)
        has_camera_bev = self.contract.has(FeatureType.CAMERA_BEV)
        num_cameras = self.contract.num_cameras
        
        warnings = []
        
        # Real camera images
        if has_images and num_cameras > 0:
            if self.mode in ['full', 'efficient']:
                encoder = EncoderConfig(
                    name='camera',
                    encoder_type='MultiCamera',
                    input_key='camera_images',
                    requirements=StandardRequirements.MULTI_CAMERA,
                    build_params={
                        'num_cameras': num_cameras,
                        'use_pretrained': True,
                    }
                )
                return {'encoder': encoder, 'warnings': warnings}
            
            elif self.mode == 'minimal':
                warnings.append(
                    "Camera: Available but skipped in minimal mode"
                )
                return None
        
        # Only placeholder BEV available
        if has_camera_bev:
            warnings.append(
                "Camera: Only placeholder BEV available (no real encoder)"
            )
        
        return None


    def _configure_agent_encoder(self) -> Dict[str, Any]:
        """
        Configure agent encoder with automatic dimension adaptation.
        
        Handles:
        - 5D states -> pad to 7D if in full/efficient mode
        - 7D states -> use directly
        - Otherwise -> use basic 5D encoder
        
        Returns:
            Dict with 'encoder', 'needs_padding'
        """
        agent_dim = self.contract.agent_state_dim
        needs_padding = False
        
        # Determine target dimension and requirements
        if agent_dim == 7:
            # Already have full 7D states
            target_dim = 7
            req = StandardRequirements.AGENT_FULL
            
        elif agent_dim == 5 and self.mode in ['full', 'efficient']:
            # Upgrade 5D to 7D by padding acceleration
            target_dim = 7
            req = StandardRequirements.AGENT_FULL
            needs_padding = True
            
        else:
            # Use basic 5D mode (minimal mode or already 5D)
            target_dim = 5
            req = StandardRequirements.AGENT_BASIC
        
        encoder = EncoderConfig(
            name='agent',
            encoder_type='Agent',
            input_key='agent_states',
            requirements=req,
            needs_padding=needs_padding,
            build_params={
                'state_dim': target_dim,
                'history_length': self.contract.history_length,
            }
        )
        
        return {
            'encoder': encoder,
            'needs_padding': needs_padding,
        }
    def _validate_configuration(self):
        """Validate that configuration meets requirements."""
        requirements = {
            name: config.requirements 
            for name, config in self.config.encoders.items()
        }

        validator = RequirementValidator(requirements=requirements)
        is_valid, report = validator.validate(self.contract)
        
        # FIXED: Only raise if invalid AND degradation NOT allowed
        if not is_valid and not self.allow_degradation:
            validator.print_report(report)
            raise ValueError(
                "Dataset contract does not satisfy encoder requirements. "
                "Set allow_degradation=True to use fallbacks."
            )
        
        # If degradation is allowed, just print warnings
        if not is_valid and self.allow_degradation:
            print("\n  Warning: Using degraded configuration due to missing features")
            validator.print_report(report)
        
        # Add any additional warnings from validation
        for encoder_name, result in report['encoders'].items():
            if result['warnings']:
                self.config.warnings.extend(result['warnings'])

    def build_encoders(
            self,
            hidden_size: int = 768,
            device: str = 'cuda'
    ) -> Dict[str, nn.Module]:
        """
        Build actual PyTorch encoder modules.
        
        Args:
            hidden_size: Hidden dimension for all encoders
            device: Device to place encoders on
            
        Returns:
            Dict mapping encoder names to PyTorch modules
        """

        #import encoder modules
        try:
            from encode.modality_encoder import (
                
                LidarEmbedding,
                MultiCameraEncoder,
                BEVEncoder
            )
            # consider action, goal, pedestrian and lacked model, whether we import more here
            from encode.modality_encoder import AgentEncoder
        
        except ImportError as e:
            raise ImportError(
                f"Failed to import encoder modules: {e}\n"
                f"Make sure your encoder modules are properly structured."
            )
        
        encoders = {}

        for name, config in self.config.encoders.items():
            enc_type = config.encoder_type
            params = config.build_params.copy()
            params['hidden_size'] = hidden_size

            # build encoder based on type
            # if enc_type == 'PointPillars':
            #     encoders[name] = (**params).to(device)

            if enc_type == 'LidarBEV':
                encoders[name] = LidarEmbedding(
                    in_channels= 2,
                    hidden_size= hidden_size,
                    patch_size= 4
                ).to(device)

            elif enc_type == 'MultiCamera':
                encoders[name] = MultiCameraEncoder(**params).to(device)

            elif enc_type == 'Agent':
                encoders[name] = AgentEncoder(**params).to(device)


            # elif enc_type == 'VectorMap':
            #     encoders[name] = VectorMapEncoder(hidden_size=hidden_size).to(device)
            
            # elif enc_type == 'GoalIntent':
            #     encoder

            else:
                warnings.warn(f" Unknown encoder type: {enc_type}, skipping")

        
        return encoders
    
    def adapt_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt batch data to match encoder expectations.
        
        This is where RUNTIME ADAPTATION happens:
        - Padding dimensions
        - Handling missing features
        - Type conversions
        
        Args:
            batch: Raw batch from dataloader
            
        Returns:
            Adapted batch ready for encoders
        """

        adapted = {}
    
        for name, config in self.config.encoders.items():
            input_key = config.input_key

            if input_key not in batch:
                if config.fallback_key and config.fallback_key in batch:
                    data = batch[config.fallback_key]
                else:
                    warnings.warn(f"Missing out '{input_key}' for encoder '{name}'")
                    continue
            else:
                data = batch[input_key]

            if name == 'agent' and config.needs_padding:
                data = self._pad_agent_states(data)

            adapted[name] = data

        # ── Passthrough: copy all keys not consumed by encoders ──────────
        # gt_trajectory, agent_history, multi_agent_*, context features, meta, etc.
        # are produced by the dataset but never declared as encoder inputs,
        # so they would otherwise be silently dropped above.
        ENCODER_INPUT_KEYS = {cfg.input_key for cfg in self.config.encoders.values()}
        PASSTHROUGH_KEYS = [
            'gt_trajectory',
            'agent_history',
            'multi_agent_states',
            'multi_agent_history',
            'intersection_features',
            'goal_features',
            'traffic_control_features',
            'pedestrian_features',
            'token',
        ]
        for key in PASSTHROUGH_KEYS:
            if key in batch and key not in adapted:
                adapted[key] = batch[key]

        return adapted
        

    def _pad_agent_states(self, states: torch.Tensor) -> torch.Tensor:
        """
        Pad agent states from 5D to 7D by adding zero acceleration.
        
        Input:  (B, N, 5) - [x, y, vx, vy, heading]
        Output: (B, N, 7) - [x, y, vx, vy, ax=0, ay=0, heading]
        """
        if states.shape[-1] == 7:
            return states  # Already 7D
        
        if states.shape[-1] != 5:
            raise ValueError(
                f"Expected 5D or 7D states, got {states.shape[-1]}D"
            )
        
        B, N = states.shape[:2]
        zeros = torch.zeros(B, N, 2, device=states.device, dtype=states.dtype)
        
        # Reconstruct: [x, y, vx, vy, ax=0, ay=0, heading]
        padded = torch.cat([
            states[..., :4],     # x, y, vx, vy
            zeros,               # ax=0, ay=0
            states[..., 4:5]     # heading
        ], dim=-1)
        
        return padded
    
    def print_summary(self):
        """Print configuration summary."""
        print(self.config.summary())
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size for this configuration."""
        return self.config.physical_limits['max_batch_size']
    
    def __repr__(self) -> str:
        return (
            f"EncoderAdapter(\n"
            f"  dataset={self.dataset.__class__.__name__},\n"
            f"  mode={self.mode},\n"
            f"  encoders={list(self.config.encoders.keys())}\n"
            f")"
        )

# Helper Functions


def create_adapter(
    dataset: BaseNavsimDataset,
    mode: str = "auto",
    **kwargs
) -> EncoderAdapter:
    """
    Convenience function to create adapter.
    
    Args:
        dataset: Dataset to adapt
        mode: Configuration mode
        **kwargs: Additional arguments to EncoderAdapter
    
    Returns:
        Configured adapter
    """
    adapter = EncoderAdapter(dataset, mode=mode, **kwargs)
    adapter.print_summary()
    return adapter

def quick_build(
    dataset: BaseNavsimDataset,
    hidden_size: int = 768,
    mode: str = "auto",
    device: str = "cuda",
) -> Tuple[EncoderAdapter, Dict[str, nn.Module]]:
    """
    Quick build: Create adapter + encoders in one call.
    
    Args:
        dataset: Dataset to adapt
        hidden_size: Hidden dimension
        mode: Configuration mode
        device: Device for encoders
    
    Returns:
        (adapter, encoders_dict)
    """
    adapter = create_adapter(dataset, mode=mode)
    encoders = adapter.build_encoders(hidden_size=hidden_size, device=device)
    return adapter, encoders
