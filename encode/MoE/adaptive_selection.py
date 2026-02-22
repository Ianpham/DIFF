"""
Phase 2: Adaptive Modality Selection

This builds on Phase 1's confidence weighting to conditionally execute modality encoders.

Key Idea: Don't compute encoders that won't be used!
- Highway scene: Skip pedestrian, intersection encoders (20-40% compute reduction)
- Urban scene: Use all encoders
- Parking: Skip intersection encoder

Inspired by:
1. AdaSlot (CVPR 2023) - Adaptive slot allocation
2. Expert Choice Routing (Google, 2022) - Experts select tokens
3. Soft MoE (Google DeepMind, 2024) - Differentiable routing

Implementation Strategy:
1. Lightweight router predicts which modalities to use
2. Gumbel-Softmax for differentiable selection during training
3. Hard selection during inference for actual compute savings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# Core Components
# ============================================================================

class ModalityRouter(nn.Module):
    """
    Lightweight router that predicts which modality encoders to execute.
    
    Input: Scene context (agent states, map features, history)
    Output: Binary selection mask [B, K] indicating which modalities to use
    
    Design principles:
    - Very lightweight (< 1% of total compute)
    - Fast to execute before encoders
    - Trained end-to-end with main task
    """
    
    def __init__(
        self,
        context_dim: int,
        num_modalities: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        min_modalities: int = 2,  # Always use at least this many
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature
        self.min_modalities = min_modalities
        
        # Lightweight routing network
        self.router = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_modalities)
        )
        
        # Per-modality prior (learned importance)
        self.modality_prior = nn.Parameter(torch.zeros(num_modalities))
    
    def forward(
        self,
        context: torch.Tensor,
        training: bool = True,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [B, context_dim] - Scene context features
            training: If True, use Gumbel-Softmax (differentiable)
                     If False, use hard selection (actual compute savings)
            
        Returns:
            selection_mask: [B, K] - Binary mask (1 = use, 0 = skip)
            selection_probs: [B, K] - Selection probabilities
        """
        B = context.size(0)
        
        # Compute routing logits
        logits = self.router(context)  # [B, K]
        
        # Add learned prior (some modalities generally more useful)
        logits = logits + self.modality_prior.unsqueeze(0)
        
        # Get selection probabilities
        probs = torch.sigmoid(logits)  # [B, K] in [0, 1]
        
        if training:
            # Gumbel-Softmax trick for differentiable sampling
            # We use Gumbel noise + sigmoid for independent binary decisions
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            noisy_logits = (logits + gumbel_noise) / self.temperature
            selection_mask = torch.sigmoid(noisy_logits)
            
            # Ensure minimum number of modalities (differentiable)
            # Select top-k + soft others
            top_k_vals, top_k_idx = torch.topk(probs, self.min_modalities, dim=1)
            min_mask = torch.zeros_like(probs).scatter(1, top_k_idx, 1.0)
            selection_mask = torch.maximum(selection_mask, min_mask)
            
        else:
            # Hard selection for inference (actual compute savings)
            # Select modalities with prob > 0.5
            selection_mask = (probs > 0.5).float()
            
            # Ensure minimum number
            num_selected = selection_mask.sum(dim=1, keepdim=True)
            need_more = (num_selected < self.min_modalities).float()
            
            if need_more.sum() > 0:
                # For samples that selected too few, take top-k
                for i in range(B):
                    if num_selected[i] < self.min_modalities:
                        _, top_idx = torch.topk(probs[i], self.min_modalities)
                        selection_mask[i, :] = 0
                        selection_mask[i, top_idx] = 1
        
        if return_logits:
            return selection_mask, probs, logits
        return selection_mask, probs


class AdaptiveModalityEncoder(nn.Module):
    """
    Wrapper around modality encoders that conditionally executes them.
    
    This is the key component that provides actual compute savings.
    """
    
    def __init__(self, encoder: nn.Module, modality_name: str):
        super().__init__()
        self.encoder = encoder
        self.modality_name = modality_name
        self._last_selection_rate = 1.0
    
    def forward(
        self,
        input_data: torch.Tensor,
        selection_mask: torch.Tensor,
        modality_idx: int
    ) -> torch.Tensor:
        """
        Args:
            input_data: [B, ...] - Input for this modality
            selection_mask: [B, K] - Binary selection mask
            modality_idx: Index of this modality in the mask
            
        Returns:
            output: [B, hidden_dim] - Encoder output (zeros if not selected)
        """
        B = input_data.size(0)
        
        # Get selection for this modality
        selected = selection_mask[:, modality_idx]  # [B]
        num_selected = selected.sum().item()
        
        # Track selection rate for logging
        self._last_selection_rate = num_selected / B
        
        if num_selected == 0:
            # Nobody selected this modality - return zeros (skip encoder)
            # This is where we save compute!
            return torch.zeros(B, self.encoder.output_dim, 
                             device=input_data.device, dtype=input_data.dtype)
        
        if num_selected == B:
            # Everyone selected - just run encoder normally
            return self.encoder(input_data)
        
        # Partial selection - only encode selected samples
        # This is the tricky part for batched execution
        selected_idx = torch.where(selected > 0.5)[0]
        selected_inputs = input_data[selected_idx]
        
        # Encode only selected samples
        selected_outputs = self.encoder(selected_inputs)
        
        # Scatter back to full batch
        full_output = torch.zeros(B, selected_outputs.size(-1),
                                  device=input_data.device, dtype=input_data.dtype)
        full_output[selected_idx] = selected_outputs
        
        return full_output
    
    @property
    def selection_rate(self) -> float:
        """Get the selection rate from last forward pass."""
        return self._last_selection_rate


# ============================================================================
# Main Adaptive Selection System
# ============================================================================

class AdaptiveModalitySelectionSystem(nn.Module):
    """
    Complete adaptive modality selection system.
    
    Combines:
    1. Lightweight router (predicts which modalities to use)
    2. Conditional encoder execution (actual compute savings)
    3. Confidence-weighted fusion (from Phase 1)
    
    Usage:
        system = AdaptiveModalitySelectionSystem(...)
        
        # Training
        outputs, info = system(
            context=scene_context,
            modality_inputs=modality_data,
            training=True
        )
        
        # Inference (with compute savings)
        outputs, info = system(
            context=scene_context,
            modality_inputs=modality_data,
            training=False
        )
    """
    
    def __init__(
        self,
        context_dim: int,
        num_modalities: int,
        hidden_dim: int,
        modality_encoders: Dict[str, nn.Module],
        use_confidence_weighting: bool = True,
        router_hidden_dim: int = 64,
        temperature: float = 1.0,
        min_modalities: int = 2
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.modality_names = list(modality_encoders.keys())
        self.use_confidence_weighting = use_confidence_weighting
        
        # Router for selecting modalities
        self.router = ModalityRouter(
            context_dim=context_dim,
            num_modalities=num_modalities,
            hidden_dim=router_hidden_dim,
            temperature=temperature,
            min_modalities=min_modalities
        )
        
        # Wrap encoders for conditional execution
        self.encoders = nn.ModuleDict()
        for name, encoder in modality_encoders.items():
            self.encoders[name] = AdaptiveModalityEncoder(encoder, name)
        
        # Confidence weighting (from Phase 1)
        if use_confidence_weighting:
            from confidence_weighting import ConfidenceWeightedSlotAttention
            self.fusion = ConfidenceWeightedSlotAttention(
                num_slots=3,  # Adjust as needed
                hidden_dim=hidden_dim,
                num_modalities=num_modalities,
                use_confidence=True
            )
        else:
            # Simple weighted averaging
            self.fusion_weights = nn.Parameter(torch.ones(num_modalities))
    
    def forward(
        self,
        context: torch.Tensor,
        modality_inputs: Dict[str, torch.Tensor],
        training: bool = True,
        return_detailed_info: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            context: [B, context_dim] - Scene context for routing
            modality_inputs: Dict[modality_name -> [B, ...]] - Inputs for each modality
            training: If True, use soft selection; if False, use hard selection
            
        Returns:
            fused_output: [B, num_slots, hidden_dim] or [B, hidden_dim]
            info_dict: Dictionary with selection masks, rates, etc.
        """
        B = context.size(0)
        
        # Step 1: Route - decide which modalities to use
        selection_mask, selection_probs = self.router(
            context, 
            training=training
        )
        
        # Step 2: Conditionally encode modalities
        modality_encodings = []
        selection_rates = {}
        
        for i, name in enumerate(self.modality_names):
            encoder_input = modality_inputs[name]
            
            # Encode (or skip if not selected)
            encoding = self.encoders[name](
                encoder_input,
                selection_mask,
                modality_idx=i
            )
            
            modality_encodings.append(encoding)
            selection_rates[name] = self.encoders[name].selection_rate
        
        # Stack encodings: [B, K, hidden_dim]
        modality_encodings = torch.stack(modality_encodings, dim=1)
        
        # Step 3: Mask out unselected modalities
        # selection_mask: [B, K] -> [B, K, 1] for broadcasting
        mask_3d = selection_mask.unsqueeze(-1)
        modality_encodings = modality_encodings * mask_3d
        
        # Step 4: Fuse with confidence weighting
        if self.use_confidence_weighting:
            fused_output, fusion_info = self.fusion(
                modality_encodings,
                return_attention=True,
                return_confidence=True
            )
        else:
            # Simple weighted average
            weights = F.softmax(self.fusion_weights, dim=0)
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
            fused_output = (modality_encodings * weights).sum(dim=1)
            fusion_info = {}
        
        # Prepare info dictionary
        info = {
            'selection_mask': selection_mask.detach(),
            'selection_probs': selection_probs.detach(),
            'selection_rates': selection_rates,
            'avg_modalities_used': selection_mask.sum(dim=1).mean().item(),
            **fusion_info
        }
        
        if return_detailed_info:
            info['modality_encodings'] = modality_encodings.detach()
        
        return fused_output, info


# ============================================================================
# Training Utilities
# ============================================================================

def compute_routing_loss(
    selection_probs: torch.Tensor,
    target_sparsity: float = 0.6,  # Target: use 60% of modalities on average
    sparsity_weight: float = 1.0
) -> torch.Tensor:
    """
    Encourage router to be sparse (not use all modalities always).
    
    Args:
        selection_probs: [B, K] - Selection probabilities
        target_sparsity: Target fraction of modalities to use
        
    Returns:
        loss: Scalar sparsity loss
    """
    # Compute average number of modalities selected
    K = selection_probs.size(1)
    avg_selected = selection_probs.sum(dim=1).mean()  # Scalar
    target_selected = target_sparsity * K
    
    # L2 loss to target sparsity
    sparsity_loss = F.mse_loss(
        avg_selected,
        torch.tensor(target_selected, device=selection_probs.device)
    )
    
    return sparsity_weight * sparsity_loss


def compute_load_balance_loss(
    selection_probs: torch.Tensor,
    min_usage_threshold: float = 0.1
) -> torch.Tensor:
    """
    Ensure all modalities get used at least some minimum fraction of time.
    
    Prevents router from completely ignoring some modalities.
    
    Args:
        selection_probs: [B, K]
        min_usage_threshold: Minimum average selection probability per modality
        
    Returns:
        loss: Scalar load balance loss
    """
    # Average selection probability per modality
    modality_usage = selection_probs.mean(dim=0)  # [K]
    
    # Penalize modalities used less than threshold
    under_threshold = F.relu(min_usage_threshold - modality_usage)
    
    return under_threshold.sum()


def compute_task_dependent_routing_loss(
    selection_mask: torch.Tensor,
    prediction_errors: torch.Tensor,
    modality_names: List[str]
) -> torch.Tensor:
    """
    Optional: Supervise routing with task performance.
    
    Idea: If prediction error is high, penalize not using key modalities.
    
    Args:
        selection_mask: [B, K] - Which modalities were selected
        prediction_errors: [B] - Per-sample prediction error (ADE/FDE)
        modality_names: List of modality names
        
    Returns:
        loss: Scalar task-aware routing loss
    """
    # For high-error samples, encourage using more modalities
    high_error_mask = (prediction_errors > prediction_errors.median()).float()
    
    # Compute modality coverage for high-error samples
    coverage = (selection_mask * high_error_mask.unsqueeze(1)).sum(dim=1)
    
    # Encourage high coverage for difficult samples
    target_coverage = selection_mask.size(1) * 0.8  # Use 80% of modalities
    coverage_loss = F.relu(target_coverage - coverage).mean()
    
    return coverage_loss


# ============================================================================
# Evaluation & Analysis
# ============================================================================

class AdaptiveSelectionAnalyzer:
    """
    Analyze adaptive selection patterns to understand router behavior.
    """
    
    @staticmethod
    def compute_selection_statistics(
        selection_masks: torch.Tensor,
        scene_types: List[str],
        modality_names: List[str]
    ) -> Dict:
        """
        Compute selection statistics by scene type.
        
        Args:
            selection_masks: [N, K] - Selection masks for N samples
            scene_types: List[str] - Scene type for each sample
            modality_names: List[str] - Names of modalities
            
        Returns:
            stats: Dictionary with selection rates per scene type
        """
        unique_scenes = list(set(scene_types))
        stats = {scene: {} for scene in unique_scenes}
        
        for scene in unique_scenes:
            # Get masks for this scene type
            scene_idx = [i for i, s in enumerate(scene_types) if s == scene]
            scene_masks = selection_masks[scene_idx]  # [N_scene, K]
            
            # Compute selection rate per modality
            for i, name in enumerate(modality_names):
                selection_rate = scene_masks[:, i].mean().item()
                stats[scene][name] = selection_rate
        
        return stats
    
    @staticmethod
    def print_selection_statistics(stats: Dict):
        """Pretty print selection statistics."""
        print("\n" + "="*70)
        print("Adaptive Modality Selection Statistics")
        print("="*70)
        
        for scene_type, modality_stats in stats.items():
            print(f"\n{scene_type}:")
            print("-" * 50)
            for modality, rate in modality_stats.items():
                bar = "█" * int(rate * 40)
                print(f"  {modality:20s} [{rate:5.1%}] {bar}")
    
    @staticmethod
    def compute_compute_savings(
        selection_rates: Dict[str, float],
        encoder_flops: Dict[str, float]
    ) -> Dict:
        """
        Estimate compute savings from adaptive selection.
        
        Args:
            selection_rates: Dict[modality -> average selection rate]
            encoder_flops: Dict[modality -> FLOPs per forward pass]
            
        Returns:
            savings: Dictionary with compute statistics
        """
        total_flops_baseline = sum(encoder_flops.values())
        total_flops_adaptive = sum(
            rate * flops 
            for rate, flops in zip(selection_rates.values(), encoder_flops.values())
        )
        
        savings_pct = (1 - total_flops_adaptive / total_flops_baseline) * 100
        
        return {
            'baseline_flops': total_flops_baseline,
            'adaptive_flops': total_flops_adaptive,
            'savings_percent': savings_pct,
            'per_modality_savings': {
                name: (1 - rate) * 100
                for name, rate in selection_rates.items()
            }
        }


# # ============================================================================
# # Integration Example
# # ============================================================================

# class YourTrajectoryModel(nn.Module):
#     """
#     Example integration of adaptive selection into trajectory prediction model.
#     """
    
#     def __init__(self, config):
#         super().__init__()
        
#         # Your existing encoders
#         self.behavior_encoder = ...  # Your BehaviorEncoder
#         self.pedestrian_encoder = ...  # Your PedestrianEncoder
#         self.intersection_encoder = ...  # Your IntersectionEncoder
#         self.goal_encoder = ...  # Your GoalEncoder
#         self.traffic_encoder = ...  # Your TrafficEncoder
        
#         # Context encoder (for routing decisions)
#         self.context_encoder = nn.Sequential(
#             nn.Linear(config.agent_dim + config.map_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64)
#         )
        
#         # Adaptive selection system
#         self.adaptive_system = AdaptiveModalitySelectionSystem(
#             context_dim=64,
#             num_modalities=5,
#             hidden_dim=config.hidden_dim,
#             modality_encoders={
#                 'behavior': self.behavior_encoder,
#                 'pedestrian': self.pedestrian_encoder,
#                 'intersection': self.intersection_encoder,
#                 'goal': self.goal_encoder,
#                 'traffic': self.traffic_encoder
#             },
#             use_confidence_weighting=True,
#             min_modalities=2  # Always use at least 2 modalities
#         )
        
#         # Decoder (unchanged)
#         self.trajectory_decoder = ...
    
#     def forward(self, batch, training=True):
#         # Encode context for routing
#         context = torch.cat([
#             batch['agent_features'],
#             batch['map_features']
#         ], dim=-1)
#         context = self.context_encoder(context)  # [B, 64]
        
#         # Prepare modality inputs
#         modality_inputs = {
#             'behavior': batch['behavior_features'],
#             'pedestrian': batch['pedestrian_features'],
#             'intersection': batch['intersection_features'],
#             'goal': batch['goal_features'],
#             'traffic': batch['traffic_features']
#         }
        
#         # Adaptive selection + fusion
#         fused_output, info = self.adaptive_system(
#             context=context,
#             modality_inputs=modality_inputs,
#             training=training
#         )
        
#         # Decode trajectories
#         predictions = self.trajectory_decoder(fused_output)
        
#         return predictions, info


# if __name__ == "__main__":
#     print("Phase 2: Adaptive Modality Selection")
#     print("=" * 70)
#     print("\nKey Benefits:")
#     print("1. 20-40% compute reduction on simple scenes")
#     print("2. Automatic adaptation to scene complexity")
#     print("3. Interpretable routing decisions")
#     print("\nSee phase2_integration.py for full integration guide")