"""
Phase 4: Advanced Optimization

This phase pushes the system to its limits with cutting-edge techniques from 
the latest MoE and neural architecture research.

Key features:
1. Dynamic depth/width adaption (conditional compute)
2. Expert choice routing (modalities select which samples to process)
3. Cross-modality attention (learn modality relationships)
4. Memory-efficient training (gradient checkpointing, mixed precision)
5. Multi-task learning (joint prediction objectives)
6. Online adaption (coninuous learning during deployment)

Expected Benefits:
- Additional 5-10% compute savings (total: 70-75% reduction)
- +1-2% additional accuracy (total: +10-12% over baseline)
- Faster training convergence (2-3x speedup)
- Better generalization to unseen scenarios

Inspired by:
1. Soft MoE (Google DeepMind, 2024) - Differentiable expert assignment
2. V-MoE (Google, 2022) - Vision-specific MoE optimizations
3. DeepSeek-V3 (2024) - Multi-head latent attention
4. Mixtral 8x7B (2024) - Efficient sparse MoE
5. Meta's MOAT (2023) - Multi-task adaptive routing
"""
# mean that the agent conder still stay the same( better for us to work in diffusion model, so agent encoder stand as a router for final product as well
# mean that we also use agent_config to set up that configuration that we need)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

# dynamic depth adaptions

class AdaptiveDepthEncoder(nn.Module):
    """
    Encoder with dynamic depth - easy scenes use fewer layers.
    
    Key idea: Not all inputs need full network depth.
    Highway scenes might only need 2 layers, urban scenes need all 4.
    
    Saves compute by early exiting when confidence is high.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            max_depth: int = 4,
            min_depth: int = 2,
            confidence_threshold: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.max_depth = max_depth
        self.min_depth = min_depth

        self.confidence_threshod = confidence_threshold

        # encoder layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for i in range(max_depth)
        ])

        # confidence estimator per layer (predicts if we can early exist)
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
            for _ in range(max_depth)
        ])

        # output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            training: bool = True,            
    )-> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, input_dim]
            training: If True, use all layers; if False, allow early exit
            
        Returns:
            output: [B, hidden_dim]
            info: Dict with depth used, confidence scores
        """
        B = x.size(0)

        # track which samples have exited
        active_mask = torch.ones(B, device = x.device, dtype = torch.bool)
        depth_used = torch.zeros(B, device = x.device, dtype = torch.long)

        confidences = []

        h = x 
        for layer_idx, (layer, conf_head) in enumerate(zip(self.layers, self.confidence_heads)):
            # process layer
            h = layer(h)

            depth_used[active_mask] = layer_idx + 1

            # check confidence
            confidence = conf_head(h) # [B, 1]
            confidences.append(confidence)

            # early exit decision (only during inference)
            if not training and layer_idx >= self.min_depth -1 :
                can_exit = (confidence.squeeze(-1) > self.confidence_threshod) & active_mask
                active_mask = active_mask & ~can_exit

                # if all exited, top
                if not active_mask.any():
                    break

        
        output = self.output_proj(h)

        info = {
            'depth_used': depth_used.float().mean().item(),
            'confidence_per_layer': torch.stack(confidences, dim = 1).detach(),
            'avg_confidence': torch.stack(confidences, dim = 1).mean().item()
        }

        return output, info
    
# 2. Expert Choice Routing (Modalities Choose Samples)

class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice Routing: Each modality chooses which samples to process.
    
    Different from standard routing where samples choose modalities!
    
    Benefits:
    - Automatic load balancing (each expert gets fixed capacity)
    - Natural specialization (experts pick what they're good at)
    - Better scaling to many modalities
    
    Inspired by Google's "Mixture-of-Experts with Expert Choice Routing" (2022)
    """

    def __init__(
            self,
            context_dim : int,
            num_modalities: int,
            expert_capacity: int, # how many samples each modality processes
            hidden_dim: int = 64
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.expert_capacity = expert_capacity

        # Each modality has a scoring network
        # Score(modality, sample) = how much modality wants to process this sample
        self.modality_scores = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_modalities)
        ])

    def forward(
            self,
            context: torch.Tensor,
            training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            context: [B, context_dim] - Scene context
            
        Returns:
            assignment_matrix: [B, K] - Which modalities process which samples
            scores: [B, K] - Assignment scores
            info: Dict with load statistics
        """

        B = context.size(0)
        K = self.num_modalities

        # each modality scores all samples
        scores = []
        for scorer in self.modality_scores:
            score = scorer(context) # [B, 1]
            scores.append(score)

        scores = torch.cat(scores, dim = -1) # [B, K]

        # expert choice: each modality selects top-capacity samples
        assignment = torch.zeros_like(scores)
        for k in range(K):
            # This modality's scores for all samples
            modality_scores = scores[:, k]  # [B]
            
            # Select top capacity samples
            if training:
                # Soft selection (differentiable)
                top_vals, top_idx = torch.topk(modality_scores, min(self.expert_capacity, B))
                soft_assignment = torch.zeros_like(modality_scores)
                soft_assignment[top_idx] = torch.softmax(top_vals, dim=0)
                assignment[:, k] = soft_assignment
            else:
                # Hard selection
                top_vals, top_idx = torch.topk(modality_scores, min(self.expert_capacity, B))
                assignment[top_idx, k] = 1.0

        # Info
        info = {
            'samples_per_modality': assignment.sum(dim=0).detach(),
            'modalities_per_sample': assignment.sum(dim=1).mean().item(),
            'load_balance_variance': assignment.sum(dim=0).var().item()
        }
        
        return assignment, scores, info
    
# cross modality attention (learn modality realtionships)

class CrossModalityAttention(nn.Module):
    """
    Learn relationships between modalities.
    
    Key insight: Some modalities are complementary:
    - Intersection + traffic control (both capture rules)
    - Pedestrian + goal (both capture intent)
    - Behavior + maneuver (both capture dynamics)
    
    Instead of treating modalities independently, learn to attend
    across modalities to enhance representations.
    """

    def __init__(
            self, 
            hidden_dim: int,
            num_modalities: int, 
            num_heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        self.head_dim = hidden_dim // self.num_heads

        # Multi-head cross-modality attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # learnable modality relationship prior
        # relationship_matrix[i, j] = how much modality i should attend to j
        self.relationship_prior = nn.Parameter(
            torch.eye(num_modalities) + 0.1 * torch.randn(num_modalities, num_modalities)
        )

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            modality_encodings: torch.Tensor,
            selection_mask: Optional[torch.Tensor] = None
    )-> torch.Tensor:
        """
        Args:
            modality_encodings: [B, K, hidden_dim]
            seletion_mask : [B, K] - which modalities are active

        Returns:
            enhanced_encodings: [B, K, hidden_dim]
        """
        B, K, D = modality_encodings.shape

        # multi-head attention
        q = self.q_proj(modality_encodings).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(modality_encodings).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(modality_encodings).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, K, K]

        # add learned realtionship prior
        relationship_bias = self.relationship_prior.unsqueeze(0).unsqueeze(0) # [1, 1, K, K]
        attn_scores = attn_scores + relationship_bias

        # mask out inactive modalities
        if selection_mask is not None:
            # create attention mask : [B, 1, K, K]
            mask_2d = selection_mask.unsqueeze(1) * selection_mask.unsqueeze(2)
            mask_2d = mask_2d.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask_2d.bool(), float('inf'))

        # attention weights
        attn_weights = F.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)

        # apply attention
        attn_output = torch.matmul(attn_weights, v) # [B, H, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, K, D)

        # output projection
        output = self.out_proj(attn_output)

        # residual connection
        enhanced = self.layer_norm(modality_encodings + output)

        return enhanced