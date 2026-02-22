"""
Improved Multi-Modal Encoder with Cross-Attention Gating
Replaces the naive concatenation/sequential fusion in TransDiffuserDiT

Key improvements:
1. Learned gating mechanism to select relevant modalities
2. Symmetric cross-attention fusion (no order dependency)
3. Fixed output token count (no memory explosion)
4. Handles missing modalities gracefully
5. Supports both parallel and hierarchical modes properly
"""
# task work
# please connect this hierachy encoder to modality encoder, this help model able to handle and select modality.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ModalityGateInfo:
    """Information about gating decisions"""
    gate_weights: torch.Tensor  # [B, num_modalities]
    confidence: torch.Tensor  # [B, 1]
    gate_logits: torch.Tensor  # [B, num_modalities]
    modality_names: List[str]
    num_active: float  # Average number of active modalities


class ModalityGateNetwork(nn.Module):
    """
    Learns which modalities are relevant for the current scene.
    
    Key features:
    - Analyzes ALL modalities to make informed decisions
    - Outputs gate weights indicating modality importance
    - Supports soft/hard/topk gating strategies
    - Estimates confidence in gating decisions
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_modalities: int,
        gate_type: str = 'soft',
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.gate_type = gate_type
        self.temperature = temperature
        self.num_modalities = num_modalities
        
        # Context analyzer - processes concatenated modality summaries
        self.context_analyzer = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Per-modality importance scorer
        self.modality_scorer = nn.Linear(hidden_size, num_modalities)
        
        # Confidence estimator (how confident are we about this decision?)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Temperature learner (adaptive temperature for gumbel-softmax)
        self.temperature_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensures positive temperature
        )
        
    def forward(
        self, 
        modality_summaries: Dict[str, torch.Tensor],
        modality_features: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Tuple[torch.Tensor, ModalityGateInfo]:
        """
        Compute gate weights based on scene context.
        
        Args:
            modality_summaries: {name: [B, D]} - Global summary per modality
            modality_features: {name: [B, T, D]} - Full features (for masking)
            training: Whether in training mode
            
        Returns:
            gate_weights: [B, num_modalities] - Importance weights
            gate_info: Additional information for analysis
        """
        device = list(modality_summaries.values())[0].device
        
        # Order modalities consistently
        modality_names = list(modality_summaries.keys())
        
        # Concatenate all summaries (pad missing ones with zeros)
        summary_list = []
        for name in modality_names:
            if name in modality_summaries:
                summary_list.append(modality_summaries[name])
            else:
                # Missing modality: use zero vector
                B = list(modality_summaries.values())[0].shape[0]
                summary_list.append(torch.zeros(B, modality_summaries[modality_names[0]].shape[1], device=device))
        
        context = torch.cat(summary_list, dim=-1)  # [B, D * num_modalities]
        
        # Analyze global context
        context_features = self.context_analyzer(context)  # [B, D]
        
        # Compute gate logits
        gate_logits = self.modality_scorer(context_features)  # [B, num_modalities]
        
        # Compute confidence
        confidence = self.confidence_head(context_features)  # [B, 1]
        
        # Adaptive temperature (for gumbel-softmax)
        if self.gate_type == 'hard' and training:
            adaptive_temp = self.temperature_predictor(context_features).squeeze(-1)  # [B]
        else:
            adaptive_temp = self.temperature
        
        # Apply gating strategy
        if self.gate_type == 'soft':
            # Soft gating: weighted combination (always differentiable)
            gate_weights = torch.softmax(gate_logits / self.temperature, dim=-1)
            
        elif self.gate_type == 'hard':
            # Hard selection: use Gumbel-Softmax
            if training:
                gate_weights = F.gumbel_softmax(
                    gate_logits, 
                    tau=adaptive_temp.mean(),  # Use predicted temperature
                    hard=True
                )
            else:
                # During inference: select argmax
                gate_weights = torch.zeros_like(gate_logits)
                gate_weights.scatter_(1, gate_logits.argmax(dim=1, keepdim=True), 1.0)
        
        elif self.gate_type == 'topk':
            # Select top-k modalities
            k = max(1, self.num_modalities // 2)
            _, top_indices = torch.topk(gate_logits, k=k, dim=-1)
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(1, top_indices, 1.0 / k)
        
        elif self.gate_type == 'learned':
            # Sigmoid gating: can activate multiple modalities independently
            gate_weights = torch.sigmoid(gate_logits)
            # Normalize to sum to 1
            gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        
        # Mask out missing modalities
        for i, name in enumerate(modality_names):
            if name not in modality_features:
                gate_weights[:, i] = 0
        
        # Renormalize after masking
        gate_weights = gate_weights / (gate_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute number of active modalities
        num_active = (gate_weights > 0.1).sum(dim=1).float().mean().item()
        
        # Package information
        gate_info = ModalityGateInfo(
            gate_weights=gate_weights,
            confidence=confidence,
            gate_logits=gate_logits,
            modality_names=modality_names,
            num_active=num_active
        )
        
        return gate_weights, gate_info


class ParallelCrossAttentionFusion(nn.Module):
    """
    Parallel fusion: All modalities attend to each other simultaneously.
    
    Benefits:
    - No order dependency
    - Symmetric treatment of all modalities
    - Efficient parallel computation
    - Each modality sees all others equally
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        dropout: float,
        output_tokens: int = 16  # Fixed output size per modality
    ):
        super().__init__()
        self.output_tokens = output_tokens
        
        # Self-attention within each modality
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention between modalities
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # FFN for refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Adaptive pooling to fixed length (solves variable token problem!)
        self.pooling = nn.AdaptiveAvgPool1d(output_tokens)
        
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor],
        gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse modalities with parallel cross-attention.
        
        Args:
            modality_features: {name: [B, T_m, D]} - Variable lengths
            gate_weights: [B, num_modalities] - Importance weights
            
        Returns:
            fused: [B, T_fixed, D] - Fixed token count!
        """
        modality_names = list(modality_features.keys())
        B = list(modality_features.values())[0].shape[0]
        D = list(modality_features.values())[0].shape[2]
        
        # Step 1: Self-attention within each modality
        enhanced_features = {}
        for name, features in modality_features.items():
            attn_out, _ = self.self_attn(features, features, features)
            features = self.norm1(features + attn_out)
            enhanced_features[name] = features
        
        # Step 2: Cross-attention - each modality attends to all others
        fused_features = {}
        for i, (name, query_features) in enumerate(enhanced_features.items()):
            # Build key-value from all other modalities (weighted by gate)
            kv_list = []
            for j, (other_name, other_features) in enumerate(enhanced_features.items()):
                if other_name != name:
                    # Apply gate weight
                    weight = gate_weights[:, j:j+1, None]  # [B, 1, 1]
                    kv_list.append(other_features * weight)
            
            if len(kv_list) > 0:
                # Concatenate other modalities
                kv = torch.cat(kv_list, dim=1)  # [B, T_others, D]
                
                # Cross-attention: query attends to others
                cross_out, _ = self.cross_attn(query_features, kv, kv)
                fused = self.norm2(query_features + cross_out)
            else:
                # Only one modality present
                fused = query_features
            
            # FFN refinement
            fused = fused + self.ffn(self.norm3(fused))
            fused_features[name] = fused
        
        # Step 3: Pool to fixed length (KEY IMPROVEMENT!)
        pooled_list = []
        for name, features in fused_features.items():
            # [B, T, D] → [B, D, T] for pooling
            features = features.transpose(1, 2)
            pooled = self.pooling(features)  # [B, D, output_tokens]
            pooled = pooled.transpose(1, 2)  # [B, output_tokens, D]
            pooled_list.append(pooled)
        
        # Step 4: Concatenate all modalities (now same length!)
        output = torch.cat(pooled_list, dim=1)  # [B, output_tokens*num_modalities, D]
        
        return output


class HierarchicalCrossAttentionFusion(nn.Module):
    """
    Hierarchical fusion with symmetric cross-attention.
    
    Unlike naive sequential fusion:
    - All modalities eventually see all others
    - No order dependency
    - Uses aggregated context instead of sequential accumulation
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        dropout: float,
        num_modalities: int,
        output_tokens: int = 16
    ):
        super().__init__()
        self.output_tokens = output_tokens
        
        # Cross-attention layers (one per modality)
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_modalities)
        ])
        
        # FFN for each modality
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout)
            )
            for _ in range(num_modalities)
        ])
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Pooling to fixed length
        self.pooling = nn.AdaptiveAvgPool1d(output_tokens)
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        gate_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Hierarchical fusion with symmetric treatment.
        """
        modality_names = list(modality_features.keys())
        
        # Step 1: Build aggregated context from all modalities
        aggregated_context = []
        for i, name in enumerate(modality_names):
            weight = gate_weights[:, i:i+1, None]  # [B, 1, 1]
            aggregated_context.append(modality_features[name] * weight)
        
        # Global context: weighted combination of all modalities
        context = torch.cat(aggregated_context, dim=1)  # [B, T_total, D]
        
        # Step 2: Each modality attends to global context
        enhanced_features = {}
        for i, (name, features) in enumerate(modality_features.items()):
            # Cross-attention to aggregated context
            attn_out, _ = self.cross_attentions[i](features, context, context)
            enhanced = self.norms[i](features + attn_out)
            
            # FFN
            enhanced = enhanced + self.ffns[i](enhanced)
            
            enhanced_features[name] = enhanced
        
        # Step 3: Pool to fixed length
        pooled_list = []
        for name, features in enhanced_features.items():
            features = features.transpose(1, 2)  # [B, D, T]
            pooled = self.pooling(features)  # [B, D, output_tokens]
            pooled = pooled.transpose(1, 2)  # [B, output_tokens, D]
            pooled_list.append(pooled)
        
        # Step 4: Concatenate and fuse
        output = torch.cat(pooled_list, dim=1)  # [B, output_tokens*num_modalities, D]
        output = self.final_fusion(output)
        
        return output


class ModalityEncoder(nn.Module):
    """
    Complete improved modality encoder.
    
    Replaces the naive encode_modalities() in TransDiffuserDiT.
    
    Key features:
    1. Learned gating for modality selection
    2. Proper cross-attention fusion (parallel or hierarchical)
    3. Fixed output token count
    4. Handles missing modalities
    5. Provides interpretable gate weights
    """
    
    def __init__(
        self,
        modality_embedders: nn.ModuleDict,
        modality_config: Dict[str, int],
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        parallel: bool = True,
        use_gating: bool = True,
        gate_type: str = 'soft',
        output_tokens_per_modality: int = 16
    ):
        super().__init__()
        self.modality_embedders = modality_embedders
        self.modality_config = modality_config
        self.hidden_size = hidden_size
        self.parallel = parallel
        self.use_gating = use_gating
        self.output_tokens_per_modality = output_tokens_per_modality
        
        num_modalities = len(modality_config)
        
        # Modality-specific learnable type embeddings
        self.modality_type_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            for name in modality_config.keys()
        })
        
        # Gating network
        if use_gating:
            self.gate_network = ModalityGateNetwork(
                hidden_size=hidden_size,
                num_modalities=num_modalities,
                gate_type=gate_type,
                dropout=dropout
            )
        
        # Fusion layer
        if parallel:
            self.fusion_layer = ParallelCrossAttentionFusion(
                hidden_size, num_heads, dropout, output_tokens_per_modality
            )
        else:
            self.fusion_layer = HierarchicalCrossAttentionFusion(
                hidden_size, num_heads, dropout, num_modalities, output_tokens_per_modality
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def encode_modalities(
        self, 
        x: Dict[str, torch.Tensor],
        return_gate_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[ModalityGateInfo]]:
        """
        IMPROVED encode_modalities - replaces naive version in TransDiffuserDiT.
        
        Args:
            x: Dict of modality tensors {name: [B, C, H, W]}
            return_gate_info: Whether to return gating information
            
        Returns:
            encoded: [B, T_fixed, D] - Fixed token count!
            gate_info: Optional gating information for analysis
        """
        B = list(x.values())[0].shape[0]
        device = list(x.values())[0].device
        
        # Step 1: Encode each modality independently
        modality_features = {}
        modality_summaries = {}
        
        for modality_name in self.modality_config.keys():
            if modality_name in x and modality_name in self.modality_embedders:
                # Encode using existing embedder
                features = self.modality_embedders[modality_name](x[modality_name])  # [B, T_m, D]
                
                # Add modality-specific type embedding
                features = features + self.modality_type_embeddings[modality_name]
                
                modality_features[modality_name] = features
                
                # Global summary for gating
                modality_summaries[modality_name] = features.mean(dim=1)  # [B, D]
            else:
                # Handle missing modality
                modality_summaries[modality_name] = torch.zeros(B, self.hidden_size, device=device)
        
        # Step 2: Compute gate weights
        if self.use_gating and len(modality_features) > 1:
            gate_weights, gate_info = self.gate_network(
                modality_summaries, 
                modality_features,
                self.training
            )
        else:
            # No gating: equal weights
            num_present = len(modality_features)
            gate_weights = torch.ones(B, len(self.modality_config), device=device) / num_present
            gate_info = None
        
        # Step 3: Fusion with cross-attention
        fused_features = self.fusion_layer(modality_features, gate_weights)
        
        # Step 4: Final projection
        encoded = self.output_projection(fused_features)  # [B, T_fixed, D]
        
        if return_gate_info:
            return encoded, gate_info
        else:
            return encoded, None
    
    def get_output_token_count(self) -> int:
        """Get the fixed output token count."""
        num_modalities = len(self.modality_config)
        return self.output_tokens_per_modality * num_modalities


class ModalityEncoderAdapter(nn.Module):
    """
    Wraps  ModalityEncoder to work with TransDiffuserDit

    What it does:
    - Extract scene modalities from adapted_batch
    - Call ModalityEncoder with proper format
    - Return scene features compatibale with Agent Encoder
    """
    
    def __init__(
            self,
            modality_embedders: nn.ModuleDict,
            modality_config: Dict[str, int],
            hidden_size: int = 768,
            num_heads: int = 12,
            use_gating: bool = True,
            gate_type: str = 'soft',
            output_token_per_modality: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # core modality encoder
        self.modality_encoder = ModalityEncoder(
            modality_embedders=modality_embedders,
            modality_config= modality_config,
            hidden_size= hidden_size,
            num_heads= num_heads,
            dropout= 0.1,
            parallel=True,
            use_gating=use_gating,
            gate_type=gate_type,
            output_tokens_per_modality=output_token_per_modality
        )

        # mapping from adapted_batch keys to modality names
        # customize this based on your dataset
        self.batch_key_mapping = {
            'lidar_bev' : 'lidar',
            'camera_bev': 'camera',
            'bev_labels': 'bev'
        }

    def forward(
            self,
            adapted_batch: Dict[str, torch.Tensor],
            return_gate_info: bool = False
    )-> Tuple[torch.tensor, Optional[ModalityGateInfo]]:
        """
        Encode scene modalities from adapted batch.
        
        Args:
            adapted_batch: Output from adapter.adapt_batch()
                Expected keys: 'lidar_bev', 'camera_bev', 'bev_labels'
            return_gate_info: Whether to return gating decisions
            
        Returns:
            scene_features: [B, T_scene, D] - Scene context tokens
            gate_info: Optional gating information
        
        """

        # extract scene modalities
        scene_modalities = self._extract_scene_modalities(adapted_batch)

        # encode with gating
        scene_features, gate_info = self.modality_encoder.encode_modalities(
            scene_modalities, 
            return_gate_info=return_gate_info
        )

        return scene_features, gate_info
    

    def _extract_scene_modalities(
            self,
            adapted_batch: Dict[str, torch.Tensor]
    )-> Dict[str, torch.Tensor]:
        """
        Extract scene modalities from adapted batch.
        
        Maps dataset-specific keys to modality encoder expected format.
        """
        scene_modalities = {}

        for batch_key, modality_name in self.batch_key_mapping.items():
            if batch_key in adapted_batch:
                scene_modalities[modality_name] = adapted_batch[batch_key]

            return scene_modalities
        
    def get_output_token_count(self) -> int:
        """Fixed output token count from modality encoder."""
        return self.modality_encoder.get_output_token_count()      
# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_improved_modality_encoder(
    modality_embedders: nn.ModuleDict,
    modality_config: Dict[str, int],
    hidden_size: int = 768,
    num_heads: int = 12,
    parallel: bool = True,
    use_gating: bool = True,
    gate_type: str = 'soft',
    output_tokens_per_modality: int = 16
) -> ModalityEncoder:
    """
    Factory function to create improved modality encoder.
    
    Use this to replace the encode_modalities logic in TransDiffuserDiT.
    
    Args:
        modality_embedders: Existing modality embedders from TransDiffuserDiT
        modality_config: Modality configuration dict
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        parallel: Use parallel or hierarchical fusion
        use_gating: Enable learned gating
        gate_type: 'soft', 'hard', 'topk', or 'learned'
        output_tokens_per_modality: Fixed tokens per modality
        
    Returns:
        Improved modality encoder
    """
    return ModalityEncoder(
        modality_embedders=modality_embedders,
        modality_config=modality_config,
        hidden_size=hidden_size,
        num_heads=num_heads,
        dropout=0.1,
        parallel=parallel,
        use_gating=use_gating,
        gate_type=gate_type,
        output_tokens_per_modality=output_tokens_per_modality
    )
