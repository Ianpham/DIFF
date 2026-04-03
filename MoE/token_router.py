
"""
DDPM/moe/token_router.py
 
Three-group MoE token routing for trajectory-conditioned diffusion.
 
Architecture (from design plan v3):
    Group A — Raw Sensory tokens   (Dynamic Gaussians, LiDAR, Camera BEV)
               Expert-choice routing, spatially-anchored.
               Hard-routed by spatial position + coarse density.
 
    Group B — Interaction tokens   (Agent states, ego state, history)
               Scene-conditioned soft routing, cross-group informed.
               Gate conditioned on Group A output + Group C output + t_embed.
               Gate temperature τ(t) modulates sharpness by diffusion timestep.
 
    Group C — Map / Context tokens (Phantom Gaussians, vector map, traffic lights)
               Deterministic-first structural routing, soft fallback.
               Gate almost-deterministic: token type drives routing,
               with a light learned soft residual.
 
Information flow is a DIRECTED graph (enforced by attention mask):
    A → C:  raw sensors confirm/refine map structure
    A → B:  raw sensors confirm agent velocities
    C → B:  map context conditions interaction routing
    B ↛ A:  interaction does NOT feed back to sensory  (no gradient leak)
    C ↛ A:  map does NOT feed back to sensory
 
Key design properties:
    - Group-local LayerNorm: normalization statistics per group, not global.
    - Attention mask:        causal mask that enforces A->C->B directionality.
    - Stop-gradient:         Group C output detached when used as Group B gate input.
    - Orthogonal group identity embeddings regularized to be distinguishable.
    - Two-layer capacity mechanism:
        Layer 1: Token-level hard cap (non-differentiable, zero gradient).
        Layer 2: Batch-level router bias scalars (gradient to bias only, not to W_g).
      Replaces load-balance auxiliary loss entirely (Part 0.2 / 4.2).
    - DyDiT timestep-conditioned skip: Group C skips at low t, Group B at high t.
    - Gate temperature tau(t): sharp at low-t, soft at high-t (Part 2.8).
    - Shared expert weight w_shared(t): timestep-scheduled (Part 2.9).
    - Warm-up cross-attention layer before main stack (Part 3.3).
    - Learned Group B skip predictor (Part 4.5).
    - Group C two-level static cache (Part 4.5).
 
This file provides:
    GroupTokenizer              -- partitions raw tokens into A/B/C, adds group identity emb
    DirectedAttentionMask       -- builds the causal attention mask for the three groups
    GroupLocalLayerNorm         -- per-group normalization
    GroupARouter                -- expert-choice spatial router for Group A
    GroupCRouter                -- structural deterministic router for Group C
    EgoCentricCrossAttention    -- Stage 1: each agent attends to ego only (Part 2.3)
    EgoProximityAgentAttention  -- Stage 2: distance-biased agent-agent attention (Part 2.4)
    MapContextReweighting       -- Stage 3: per-agent scalar gate for map context (Part 2.6)
    IntentionPredictionHead     -- Pre-gate intention logits (Part 2.7)
    GroupBRouter                -- scene-conditioned soft router for Group B (cross-group gate)
    TokenLevelHardCap           -- non-differentiable post-softmax capping (Part 4.2 Layer 1)
    WarmUpCrossAttention        -- lighter cross-attention before main stack (Part 3.3)
    LearnedSkipPredictor        -- learned per-token skip for Group B (Part 4.5)
    GroupCTwoLevelCache         -- two-level static cache for Group C (Part 4.5)
    DyDiTSkipScheduler          -- per-group skip decision conditioned on diffusion timestep t
    ModalityMoERouter           -- top-level orchestrator that wires all of the above
"""
 

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from MoE.intention_heads import IntentionHeads

# constanst/ group identifiers

GROUP_A  = 0    #raw sensory (Dynamic Gaussians, BEV, LIDAR summaries)
GROUP_B  = 1    # Interaction(agent / ego states, history)
GROUP_C  = 2    # Map/context (phantom Gaussians, vector map, traffic lights)

# Token-type codes within GROUP C - used by structural router.
# These are set at tokenization time from the data pipeline metadata.

TTYPE_PHANTOM       = 0 # phantom / occluded Gaussians.
TTYPE_VECTORMAP     = 1 # vector map lane tokens.
TTYPE_TRAFFIC_LT    = 2 # traffic light tokens
TTYPE_INTERSECTION  = 3 # intersection topoloby tokens
TTYPE_BEV_STRUCT    = 4 # structural BEV features (no raw sensor)
TTYPE_UNKNOWN       = 5 # soft-fallback for ambigous tokens

NUM_C_TYPES = 6


# Config dataclasses

@dataclass
class MoEConfig:
    """All hyper-parameters for the three-group MoE router."""
 
    # Token counts per group
    num_tokens_A: int = 64
    num_tokens_B: int = 64
    num_tokens_C: int = 128
 
    # Feature dimension (shared across groups)
    embed_dim: int = 256
 
    # Number of experts per group
    num_experts_A: int = 4
    num_experts_B: int = 4
    num_experts_C: int = 6      # >= NUM_C_TYPES
 
    # Top-k experts per token
    top_k_A: int = 2
    top_k_B: int = 2
    top_k_C: int = 1
 
    # Capacity factor per group
    capacity_factor_A: float = 1.5
    capacity_factor_B: float = 1.5
    capacity_factor_C: float = 2.0
 
    # Routing floor: epsilon = min(routing_floor_base, 0.15 / k)  (Part 1.4)
    routing_floor_base: float = 0.05
 
    # Token-level hard cap (Part 4.2, Layer 1)
    cap_low:  float = 0.50
    cap_high: float = 0.60
 
    # DyDiT skip thresholds
    t_skip_C: float = 0.2
    t_skip_B: float = 0.7
 
    # Group A cache interval
    cache_interval_A: int = 5
 
    # Orthogonal group identity regularization weight
    ortho_reg_weight: float = 1e-3
 
    # Stop-gradient on Group C -> Group B gate path
    stopgrad_C_to_B: bool = True
 
    # Maximum diffusion timesteps
    T_max: int = 1000
 
    # Group C structural router temperature
    struct_router_temp: float = 0.1

    # Group C soft residual weight for KNOWN types (Part 1.2)
    # Controls how much the learned secondary gate influences known-type routing.
    # 0.0 = pure deterministic for known types, 1.0 = fully learned.
    # UNKNOWN tokens always use full soft gate regardless of this value.
    struct_soft_residual_weight: float = 0.05  # small but non-zero per plan

 
    # Expert FFN hidden dim multiplier
    expert_ff_mult: int = 4
 
    # Number of attention heads for cross-group gate attention
    gate_num_heads: int = 4
 
    # Gate temperature schedule (Part 2.8)
    tau_min: float = 0.5
    tau_max: float = 2.0
 
    # Shared expert weight schedule (Part 2.9)
    shared_weight_min: float = 0.10
    shared_weight_max: float = 0.20
 
    # Batch-level router bias penalty coefficient (Part 4.2 Layer 2)
    bias_penalty_coeff: float = 0.01
 
    # Stage 2 agent-agent attention (Part 2.4)
    agent_attn_num_heads: int = 4
    agent_attn_K_default: int = 4
    agent_attn_K_max: int = 6
    agent_proximity_threshold: float = 20.0
 
    # Intention prediction (Part 2.7)
    vehicle_intention_classes: int = 6
    pedestrian_intention_classes: int = 2
 
    # Warm-up cross-attention (Part 3.3)
    warmup_num_heads: int = 2
 
    # Learned skip predictor (Part 4.5)
    skip_predictor_hidden: int = 128
    skip_min_token_fraction: float = 0.20
 
    # Group C two-level cache (Part 4.5)
    t_cache_high_default: float = 0.6
    t_cache_low: float = 0.15
 
    @property
    def routing_floor(self) -> float:
        """Budget-aware routing floor: epsilon = min(base, 0.15 / k) (Part 1.4)."""
        k = self.num_experts_B
        return min(self.routing_floor_base, 0.15 / max(k, 1))
    
# group identity embeddings

class GroupIdentityEmbedding(nn.Module):
    """
    Learnable per-group identity vectors with orthogonality regularizations.
    
    Each group gets a single learned vector that is broadcast-added to all
    tokens in that group before processing.  Orthogonality regularization
    keeps the three vectors distinguishable throughout training.

    Shape: group_id_vecs  (3, embed_dim)  
    """

    def __init__(self, embed_dim: int, num_groups: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # initialize with random orthonormal vectors so the model starts
        # with perfect separation (avoids cold-start confusion)

        vecs = torch.randn(num_groups,embed_dim)
        vecs = torch.linalg.qr(vecs.T).Q.T[:num_groups]     # (3, D), orthonormal rows
        self.group_id_vecs = nn.Parameter(vecs)


    def forward(self, tokens: torch.Tensor, group_id: int) -> torch.Tensor:
        """
        Args:
            tokens:   (B, N, D)
            group_id: one of GROUP_A, GROUP_B, GROUP_C
        Returns:
            tokens + group_identity_vector  (B, N, D)        
        """
        vec = self.group_id_vecs[group_id]      #(D,)
        return tokens + vec.unsqueeze(0).unsqueeze(0)
    
    def orthogonality_loss(self) -> torch.Tensor:
        """
        Penalise off-diagonal entries of Gram matrix.
        Loss = || G - I ||_F^2  where  G = V V^T / ||V||^2
        Encourages the three identity vectors to remain orthogonal.
        """
        V = F.normalize(self.group_id_vecs, dim = -1)   # (3, D) unit rows
        G = V @ V.T
        I = torch.eye(self.num_groups, device = V.device)

        return ((G - I) ** 2).sum()
    

# group local Layer Norm
class GroupLocalLayerNorm(nn.Module):
    """
    Applies LayerNorm independently within each group's token slice.

    Prevents normalization statistics from leaking between groups (path 2
    contamination described)

    Args:
        embed_dim: feature dimension D.
        num_groups: number of groups (default 3).

    """

    def __init__(
            self,
            embed_dim: int,
            num_groups: int = 3,
    ):
        super().__init__()

        # one independent LayerNorm per group
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_groups)
        ])

    def forward(
            self,
            tokens_A: torch.Tensor,
            tokens_B: torch.Tensor,
            tokens_C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args / Returns: each tensor (B, N_g, D), normalized independently.
        """
        return (
            self.norms[GROUP_A](tokens_A),
            self.norms[GROUP_B](tokens_B),
            self.norms[GROUP_C](tokens_C)
        )
    

# Directed attention mask

class DirectedAttentionMask:
    """
    Builds a boolean attentuon mask that enforces A -> C -> B directionality
    
    Tokens are concatenated in the order [A | C | B] before attentions.
    The mask allows:
        - A attends to A       (self)
        - C attends to A, C    (A→C direction: C reads raw evidence)
        - B attends to A, C, B (C→B direction: B reads map + raw context)   
    
    Blocks:
        - A attending to C or B (no upward feedback)
        - C attending to B      (no upward feedback)

    This is a STATIC mask (same for all batches / timesteps) since the group
    sizes are fixed. It is registered as a buffer in ModalityMoERouter
    
    Args:
        num_tokens_A: N_A
        num_tokens_C: N_C
        num_tokens_B: N_B

    Usage:
        mask = DirectedAttentionMask(64, 128, 64)
        bool_mask = mask.build()      # (N_total, N_total)  True = attend
        float_mask = mask.build_additive()  # 0.0 or -inf, for additive masking

    """
    def __init__(
            self,
            num_tokens_A: int,
            num_tokens_C: int,
            num_tokens_B: int, 
    ):
        super().__init__()
        self.N_A = num_tokens_A
        self.N_B = num_tokens_B
        self.N_C = num_tokens_C

        self.N = num_tokens_A + num_tokens_B + num_tokens_C

    def build(self) -> torch.BoolTensor:
        """Return (N, N) boolean mask. True = this (query, key) pair is allowed"""
        N_A, N_C, N_B = self.N_A, self.N_C, self.N_B

        N = self.N
        mask = torch.zeros(N, N, dtype= torch.bool)

        # slices (tokens are laid out as A first, then C, then B)
        sA = slice(0, N_A)
        sC = slice(N_A,N_A + N_C)
        sB = slice(N_A + N_C, N)

        # A → A  (self-attention within Group A)
        mask[sA, sA] = True
        # C → A  (Group C queries attend to Group A keys)
        mask[sC, sA] = True
        # C → C  (self-attention within Group C)
        mask[sC, sC] = True
        # B → A  (Group B queries attend to Group A keys)
        mask[sB, sA] = True
        # B → C  (Group B queries attend to Group C keys)
        mask[sB, sC] = True
        # B → B  (self-attention within Group B)
        mask[sB, sB] = True

        return mask  # shape (N, N), True = attend
    
    def build_additive(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Returns (N, N) float tensor: 0.0 where allowed, -inf where blocked.
        Compatible with PyTorch's MultiheadAttention attn_mask parameter.
        """
        bool_mask = self.build().to(device)
        additive = torch.zeros(self.N, self.N, device=device)
        additive[~bool_mask] = float("-inf")
        return additive
    
# token level hard cap
class TokenLevelHardCap:
    """Non-differentiable post-softmax capping with redistribution.
 
    For each token, if any expert receives routing weight > cap(t),
    clip to cap(t) and redistribute excess proportionally to other
    experts (above their floor).
 
    ZERO gradient flows through this operation — implemented via
    torch.no_grad() context. The gate parameters receive no signal
    from this clipping.
 
    Groups A and C: use this only.
    Group B: uses this + batch-level bias adjustment (Layer 2).
 
    Args:
        cap_low:  cap at t=0 (tight — need multiple experts at low noise)
        cap_high: cap at T_max (loose — some concentration acceptable)
        T_max:    maximum diffusion timestep
    """

    def __init__(self, cap_low: float = 0.50, cap_high: float = 0.60, T_max: int = 1000):
        self.cap_low = cap_low
        self.cap_high = cap_high
        self.T_max = T_max

    def cap_value(self, t: torch.Tensor) -> torch.Tensor:
        """compute cap(t) per sample. Return (B, ) value"""
        t_norm = t.float() / self.T_max
        return self.cap_low + (self.cap_high + self.cap_low) * t_norm
    
    @torch.no_grad()
    def apply(
        self,
        probs: torch.Tensor,            # (B, N, E) - post-softmax routing probs
        t: torch.Tensor,                # (B,)  - diffusion timestep
        floor: float = 0.0,             # routing floor (already applied)
    ) -> torch.Tensor:
        """clip and redistribute. return (B, N, E) - capped probs (detached)"""
        B, N, E = probs.shape
        cap = self.cap_value(t) # (B,)
        cap = cap.view(B, 1, 1).expand(B, N, E)

        capped = probs.clone()
        excess = F.relu(capped - cap) # (B, N, E) - amount over cap
        capped = capped - excess            # clip to cap

        # redistribute excess prportionally to non-capped experts
        # (those below cap and above floor)
        headroom = F.relu(cap - capped)     # how much room each expert has
        headroom_sum = headroom.sum(-1, keepdim= True).clamp(min = 1e-8)
        redistribution = excess.sum(-1, keepdim=True) * (headroom / headroom_sum)

        capped = capped + redistribution

        return capped
    
# shared routing utilities

def _apply_routing_floor(probs: torch.Tensor, floor: float) -> torch.Tensor:
    """Ensure each expert gets at least `floor` probability mass.
    Interpolates between uniform and computed distribution.
    """
    E = probs.shape[-1]
    if floor <= 0.0:
        return probs
    uniform = torch.full_like(probs, 1.0 / E)
    alpha = min(floor * E, 1.0)
    return (1 - alpha) * probs + alpha * uniform
 
 
def _topk_mask(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top-k entries per token."""
    topk_vals, topk_idx = probs.topk(k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk_idx, topk_vals)
    return mask
# place holder for timestepEmbedding, please consider that we already have 

# warm up cross attention layer (Part 3.3) 
class WarmUpCrossAttention(nn.Module):
    """Lighter cross-attention BEFORE the main transformer stack (Part 3.3).
    Gives Group B/C cross-modal representation before any routing decision.
    Directions: A->C, A->B, C->B.  Group A unchanged (no upward feedback).
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        D = cfg.embed_dim
        nh = cfg.warmup_num_heads
        self.attn_A_to_C = nn.MultiheadAttention(D, nh, batch_first=True, dropout=0.0)
        self.norm_A_to_C = nn.LayerNorm(D)
        self.attn_A_to_B = nn.MultiheadAttention(D, nh, batch_first=True, dropout=0.0)
        self.norm_A_to_B = nn.LayerNorm(D)
        self.attn_C_to_B = nn.MultiheadAttention(D, nh, batch_first=True, dropout=0.0)
        self.norm_C_to_B = nn.LayerNorm(D)
    
    def forward(self, tokens_A, tokens_B, tokens_C):
        A_det = tokens_A.detach()
        c_out, _ = self.attn_A_to_C(query = tokens_C, key = A_det, value = A_det)
        tokens_C = self.norm_A_to_C(tokens_C + c_out)
        b_out_a, _ = self.attn_A_to_B(query = tokens_B, key = A_det, value= A_det)
        tokens_B = self.norm_A_to_B(tokens_B + b_out_a)
        C_det = tokens_C.detach()
        b_out_c, _ = self.attn_C_to_B(query=tokens_B, key=C_det, value=C_det)
        tokens_B = self.norm_C_to_B(tokens_B + b_out_c)
        return tokens_A, tokens_B, tokens_C

# stage 1 ego centric cross attention (nn.Module):

class EgoCentricCrossAttention(nn.Module):
    """Stage 1: each agent attends to ego token ONLY (Part 2.3).
    Output: LayerNorm(agent_token + CrossAttn(Q=agent, KV=ego_token)).
    Cost: O(N) -- one attention op against one key.
    """
 
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        D = cfg.embed_dim
        self.cross_attn = nn.MultiheadAttention(D, cfg.agent_attn_num_heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(D)
 
    def forward(self, tokens_B: torch.Tensor, ego_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens_B.shape
        ego_indices = ego_mask.long().argmax(dim=-1)
        ego_token = tokens_B[torch.arange(B, device=tokens_B.device), ego_indices].unsqueeze(1)
        attn_out, _ = self.cross_attn(query=tokens_B, key=ego_token, value=ego_token)
        return self.norm(tokens_B + attn_out)
    
# stage 2 Ego proximity filtered agent-agent attention 
class EgoProximityAgentAttention(nn.Module):
    """Stage 2: distance-biased agent-agent attention (Part 2.4).
    Top-K neighbors by ego-proximity. Separate ego projections.
    K from deterministic lookup table (ego speed, local density). No gradient.
    Applied as RESIDUAL over Stage 1.
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        D = cfg.embed_dim
        nh = cfg.agent_attn_num_heads
        self.K_default = cfg.agent_attn_K_default
        self.K_max = cfg.agent_attn_K_max

        self.proximity_threshold = cfg.agent_proximity_threshold

        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)

        self.ego_q_proj = nn.Linear(D, D)
        self.ego_k_proj = nn.Linear(D, D)
        self.ego_v_proj = nn.Linear(D, D)

        self.distance_bias = nn.Sequential(nn.Linear(2, D//4), nn.ReLU(), nn.Linear(D//4, nh))

        self.num_heads = nh

        self.head_dim = D // nh

        self.norm = nn.LayerNorm(D)

    def _compute_K(self, ego_speed: torch.Tensor, ego_distances: torch.Tensor) -> int:
        B, N = ego_distances.shape
        close_count = (ego_distances < self.proximity_threshold).float().sum(-1)
        
        density = close_count / max(N, 1)
        avg_speed = ego_speed.mean().item()
        avg_density = density.mean().item()

        K = self.K_default
        if avg_speed > 15.0:
            K = min(K + 1, self.K_max)
        if avg_density > 0.5:
            K = min(K + 1, self.K_max)

        return K
    
    def forward(self, agent_repr_1, ego_distance, ego_mask, ego_speed):
        B, N, D = agent_repr_1.shape
        nh, hd = self.num_heads, self.head_dim
        K = min(self._compute_K(ego_speed, ego_distance), N - 1)

        if K <= 0: 
            return self.norm(agent_repr_1)
        # top-K neighbors per agent by ago-proximity
        self_mask = torch.eye(N, device = agent_repr_1.device, dtype=torch.bool).unsqueeze(0)
        dist_rank = ego_distance.unsqueeze(1).expand(B, N, N).clone()
        dist_rank[self_mask.expand(B, -1, -1)] = float("inf")
        _, topk_idx = dist_rank.topk(K, dim = -1, largest = False) # (B, N, K)

        # gather neighbors
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D) # (B, N, K, D)
        neighbors = agent_repr_1.unsqueeze(2).expand(-1, -1, N, -1)
        gathered = torch.gather(neighbors, 2, idx_exp) # (B, N, K, D)

        # projections (ego uses separate projection)
        is_ego = ego_mask.unsqueeze(-1).float()
        Q = (1 - is_ego) * self.q_proj(agent_repr_1) + is_ego * self.ego_q_proj(agent_repr_1)

        K_mat = self.k_proj(gathered)
        V_mat = self.v_proj(gathered)

        Q = Q.reshape(B, N, nh, hd).transpose(1, 2)
        K_mat = K_mat.reshape(B, N, K, nh, hd).permute(0, 3, 1, 2, 4)
        V_mat = V_mat.reshape(B, N, K, nh, hd).permute(0, 3, 1, 2, 4)

        scores = torch.einsum("bhnd, bhnkd -> bhnk", Q, K_mat) / math.sqrt(hd)

        # distance bias
        q_dist = ego_distance.unsqueeze(-1).expand(B, N, K)
        k_dist = torch.gather(ego_distance.unsqueeze(1).expand(-1, N, -1),2, topk_idx)
        dist_pair = torch.stack([q_dist, k_dist], dim = -1)
        bias = self.distance_bias(dist_pair).permute(0, 3, 1, 2)
        scores = scores * bias

        attn_weghts = F.softmax(scores, dim = -1)
        attn_out = torch.einsum("bhnk, bhnkd -> bhnd", attn_weghts, V_mat)
        attn_out = attn_out.transpose(2, 1).reshape(B, N, D)

        return self.norm(agent_repr_1 +  attn_out)
    
# stage 3: map context re-weighting
class MapContextReweighting(nn.Module):
    """Stage 3: per-agent scalar gate (Part 2.6).
    agent_repr_3 = agent_repr_2 * sigmoid(W * [agent_repr_2 | map_ctx])
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        D = cfg.embed_dim
        self.gate = nn.Sequential(nn.Linear(D*2, D), nn.ReLU(), nn.Linear(D, D))
    
    def forward(self, agent_repr_2, map_context_summary):
        gate_input = torch.cat([agent_repr_2, map_context_summary], dim=-1)
        return agent_repr_2 * torch.sigmoid(self.gate(gate_input))

# intention prediction heads, we might want to move it to encode/modality_encoder.py
# just for now, now we can replace in IntentionHeads
# class IntentionPredictionhead(nn.Module):
#     """
#     Pre-gate intention prediction (Part 2.7)
#     Vehicle: 6-class (Lat L/S/R x long A/C/D)
#     Pedestrian: SEPARATE 2-class (not-crossing/crossing)
#     Logits feed directly into gate query (part 2.8)
#     """      

#     def __init__(self, cfg: MoEConfig):
#         super().__init__()
#         D = cfg.embed_dim
#         self.vehicle_classes = cfg.vehicle_intention_classes
#         self.ped_classes = cfg.pedestrian_intention_classes
#         self.vehicle_head = nn.Sequential(
#             nn.Linear(D, D//2),
#             nn.ReLU(),
#             nn.Linear(D//2, self.vehicle_classes)
#         )

#         self.pedestrian_head = nn.Sequential(
#             nn.Linear(D, D//2),
#             nn.ReLU(),
#             nn.Linear(D//2, self.ped_classes)
#         )

#         self.output_dim = max(self.vehicle_classes, self.ped_classes)
#         self.vehicle_proj = nn.Linear(self.vehicle_classes, self.output_dim)
#         self.ped_proj = nn.Linear(self.ped_classes, self.output_dim)

#     def forward(self, agent_repr_2, is_pedestrian):
#         vehicle_logits = self.vehicle_head(agent_repr_2)
#         ped_logits = self.pedestrian_head(agent_repr_2)
#         is_ped = is_pedestrian.unsqueeze(-1).float()
#         vehicle_proj = self.vehicle_proj(vehicle_logits)
#         ped_proj = self.ped_proj(ped_logits)
#         intention_for_gate = (1 - is_ped) * vehicle_proj + is_ped * ped_proj

#         return intention_for_gate , vehicle_logits, ped_logits
    

# learned skip predictor for group B
 
class LearnedSkipPredictor(nn.Module):
    """Learned per-token skip predictor for Group B (Part 4.5).
    2-layer MLP. Input: token + bottlenecked C context + t_embed.
    Min token floor: at least skip_min_token_fraction processed.
    """
 
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        D = cfg.embed_dim
        H = cfg.skip_predictor_hidden
        self.min_fraction = cfg.skip_min_token_fraction
        self.T_max = cfg.T_max
        self.ctx_bottleneck = nn.Linear(D, H)
        self.predictor = nn.Sequential(nn.Linear(D + H + D, H), nn.ReLU(), nn.Linear(H, 1))
 
    def forward(self, tokens_B, ctx_C_summary, t_embed, t, rare_token_mask=None):
        B, N, D = tokens_B.shape
        ctx_bn = self.ctx_bottleneck(ctx_C_summary)
        t_exp = t_embed.unsqueeze(1).expand(-1, N, -1)
        skip_score = torch.sigmoid(self.predictor(torch.cat([tokens_B, ctx_bn, t_exp], dim=-1)).squeeze(-1))
        skip_mask = skip_score > 0.5
        if rare_token_mask is not None:
            skip_mask = skip_mask & ~rare_token_mask
        min_tokens = max(1, int(N * self.min_fraction))
        n_processed = (~skip_mask).sum(dim=-1)
        for b in range(B):
            if n_processed[b] < min_tokens:
                n_unskip = min_tokens - n_processed[b].item()
                skipped = skip_mask[b].nonzero(as_tuple=True)[0]
                if len(skipped) > 0:
                    scores = skip_score[b, skipped]
                    _, lowest = scores.topk(min(n_unskip, len(skipped)), largest=False)
                    skip_mask[b, skipped[lowest]] = False
        return skip_mask
 
    def training_loss(self, tokens_B, ctx_C_summary, t_embed, grad_magnitude):
        B, N, D = tokens_B.shape
        ctx_bn = self.ctx_bottleneck(ctx_C_summary)
        t_exp = t_embed.unsqueeze(1).expand(-1, N, -1)
        skip_score = self.predictor(torch.cat([tokens_B, ctx_bn, t_exp], dim=-1)).squeeze(-1)
        gm_norm = grad_magnitude / (grad_magnitude.max(dim=-1, keepdim=True).values + 1e-8)
        target = 1.0 - gm_norm
        return F.binary_cross_entropy_with_logits(skip_score, target)
    
# group C: Two level static cache

class GroupCTwoLevelCache:
    """Two-level static cache for Group C DyDiT (Part 4.5).
    t_cache_high: conditioned on scene complexity from Group A.
    t_cache_low: fixed threshold for fine-shaping window.
    Between them: RECOMPUTE normally.
    """
    def __init__(self, cfg: MoEConfig):
        self.t_cache_high_default = cfg.t_cache_high_default
        self.t_cache_low = cfg.t_cache_low
        self.T_max = cfg.T_max
        self._cache_high: Optional[torch.Tensor] = None
        self._cache_low: Optional[torch.Tensor] = None
    
    def compute_t_cache_high(self, scene_complexity):
        base = self.t_cache_high_default
        return base - 0.15 * (1 - scene_complexity) + 0.10 * scene_complexity
    
    def compute(self, t, scene_complexity):
        t_norm = t.float() / self.T_max
        t_high = self.compute_t_cache_high(scene_complexity)

        return (t_norm >= self.t_cache_low) & (t_norm < t_high)
    
    def use_high_cache(self, t, scene_complexity):
        t_norm = t.float() / self.T_max
        return t_norm >= self.compute_t_cache_high(scene_complexity)
 
    def use_low_cache(self, t):
        return (t.float() / self.T_max) < self.t_cache_low
 
    def store_high(self, output):
        self._cache_high = output.detach().clone()
 
    def store_low(self, output):
        self._cache_low = output.detach().clone()
 
    def get_high(self):
        return self._cache_high
 
    def get_low(self):
        return self._cache_low
 
    def reset(self):
        self._cache_high = None
        self._cache_low = None

 
# group A router - expert-choice, spatially-anchored
class GroupARouter(nn.Module):
    """Expert-choice router for Group A (Part 1.2).
    Each EXPERT chooses its top-K tokens by affinity.
    """
 
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_A
        self.top_k = cfg.top_k_A
        self.embed_dim = cfg.embed_dim
        self.gate_proj = nn.Sequential(
            nn.Linear(cfg.embed_dim + 3, cfg.embed_dim // 2), nn.GELU(),
            nn.Linear(cfg.embed_dim // 2, self.num_experts),
        )
        self.expert_spatial_centers = nn.Parameter(torch.randn(self.num_experts, 3) * 10.0)
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)
        self._floor = cfg.routing_floor
 
    def forward(self, tokens, spatial_xyz, t):
        B, N, D = tokens.shape
        E = self.num_experts
        dists = torch.cdist(
            spatial_xyz.reshape(B * N, 1, 3),
            self.expert_spatial_centers.unsqueeze(0).expand(B * N, -1, -1),
        ).reshape(B, N, E)
        spatial_affinity = -dists / (dists.mean() + 1e-6)
        gate_input = torch.cat([tokens, spatial_xyz], dim=-1)
        content_logits = self.gate_proj(gate_input)
        logits = content_logits + spatial_affinity
 
        # Expert-choice: transpose so experts choose tokens
        expert_logits = logits.transpose(1, 2)  # (B, E, N)
        tokens_per_expert = max(1, (N * self.top_k) // E)
        topk_vals, topk_idx = expert_logits.topk(min(tokens_per_expert, N), dim=-1)
        dispatch = torch.zeros(B, E, N, device=tokens.device)
        dispatch.scatter_(-1, topk_idx, topk_vals.sigmoid())
        dispatch = dispatch.transpose(1, 2)  # (B, N, E)
 
        dispatch = _apply_routing_floor(dispatch, self._floor)
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)
        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
        return dispatch, combine
 
 
# Group C Router -- Structural Deterministic 
 
class GroupCRouter(nn.Module):
    """Structural near-deterministic router for Group C (Part 1.2).
 
    Two-level routing:
        1. Primary: deterministic token-type -> expert assignment.
        2. Secondary: lightweight 2-layer MLP for within-type fine-grained
           specialization. Small residual for known types, full soft gate
           for TTYPE_UNKNOWN tokens.
 
    Temperature: softmax(logits / temp). Low temp = sharp = deterministic.
 
    Gate freeze (Part 6.5): after Stage 2 exit, call freeze_gate() to
    permanently freeze the soft gate and secondary MLP. Near-deterministic
    routing structure must not change during Stage 3/4 joint training.
    """
 
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_C
        assert cfg.num_experts_C >= NUM_C_TYPES, (
            f"num_experts_C ({cfg.num_experts_C}) must be >= NUM_C_TYPES ({NUM_C_TYPES})"
        )
        self.top_k = cfg.top_k_C
        self.temp = cfg.struct_router_temp

        # add soft residual weight
        self.soft_residual_weight = cfg.struct_soft_residual_weight

        self._floor = cfg.routing_floor
        self._gate_frozen = False

        # Primary: deterministic token-type -> expert assignment matrix.
        # Distribute types across experts evenly (avoid dead experts).

        base_assignment = torch.zeros(NUM_C_TYPES, self.num_experts)
        for t_type in range(NUM_C_TYPES - 1): #known type 0..4
            # primary expert for this type
            primary_expert = t_type % self.num_experts
            base_assignment[t_type, t_type % self.num_experts] = 1.0

        # unknown: uniform prior
        base_assignment[TTYPE_UNKNOWN] = 1.0 / self.num_experts

        # if num_experts > NUM_C_TYPES-1, assign surplus experts as secondary
        # receivers for the most common types (spread load).
        num_known = NUM_C_TYPES - 1 # 5 known types
        if self.num_experts > num_known:
            for extra_idx in range(num_known, self.num_experts):
                # map surplus expert back to a known type (round-robin)
                paired_type = extra_idx % num_known
                # Add as secondary with lower weight (0.3) to primary's 1.0
                base_assignment[paired_type, extra_idx] = 0.3
            # renormalize rows to sum to 1
            for t_type in range(NUM_C_TYPES - 1):
                row_sum = base_assignment[t_type].sum()
                if row_sum > 0:
                    base_assignment[t_type] /= row_sum

        self.register_buffer("base_assignment", base_assignment)
        
        # secondary: within-type 2-layer MLP gate for fine-grained
        # specialization (Part 1.2). Applies to all token types with
        # different mixing weights (small for known, full for UNKNOWN)

 
        self.second_gate = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 4), nn.GELU(),
            nn.Linear(cfg.embed_dim // 4, self.num_experts),
        )
        for m in self.second_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Hard cap only (no Layer 2 for Group C — near-deterministic)
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)
    
    def freeze_gate(self) -> None:
        """
        Permanently freeze the Group C gate after Stage 2 exit (Part 6.5).
        Near-deterministic routing structure must not change.
        Call this once after stage 2 training completes.
        """
        self._gate_frozen = True
        for param in self.second_gate.parameters():
            param.requires_grad_(False)
        
    def unfreeze_gate(self) -> None:
        """unfreeze (only for debugging / testing - not recommended)"""
        self._gate_frozen = False
        for param in self.second_gate.parameters():
            param.requires_grad_(True)

    @property
    def gate_is_frozen(self) -> bool:
        return self._gate_frozen   

 
    def forward(self, tokens, token_types, t):
        """
        Args:
            tokens:      (B, N_C, D)
            token_types: (B, N_C) long — TTYPE_* codes
            t:           (B,) diffusion timestep
        Returns:
            dispatch_weights: (B, N_C, E)
            combine_weights:  (B, N_C, E)
        """

        B, N, D = tokens.shape
        # primary: deterministic base probabilities from token shape
        base_probs = self.base_assignment[token_types.reshape(-1)].reshape(B, N, -1)

        # secondary: learned within-type gate (part 1.2)
        # temporature as divisor: low temp -> sharp (deterministic)

        secondary_logits = self.second_gate(tokens) / self.temp
        secondary_probs = F.softmax(secondary_logits, dim=-1)

        # Mix primary (deterministic) and secondary (learned):
        #   - Known types: mostly deterministic + small learned residual
        #   - UNKNOWN:     fully learned (no deterministic prior)
        is_unknown = (token_types == TTYPE_UNKNOWN).float().unsqueeze(-1)

        # known types: (1 - alpha) * base + alpha * secondary, alpha = soft_residual_weight
        alpha = self.soft_residual_weight
        known_probs = (1-alpha)* base_probs + alpha * secondary_probs
        probs = (1 - is_unknown) * base_probs + is_unknown * secondary_logits


        probs = _apply_routing_floor(probs, self._floor)
        dispatch = _topk_mask(probs, self.top_k)
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)


        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
        return dispatch, combine

# # Group B Router — Scene-Conditioned Soft Router (cross-group gate)
# class GroupBRouter(nn.Module):
#     """Scene-conditioned soft router for Group B (interaction / agent tokens).

#     The routing decision must account for:
#         1. What Group A (sensory) saw — confirms agent velocities, detects crosswalks.
#         2. What Group C (map/context) saw — confirms intersection topology, TL state.
#         3. The current diffusion timestep t — Group B is most critical at low t
#            (fine trajectory shaping), so the gate should modulate confidence by t.

#     Gate architecture:
#         context_vec = CrossAttn(B_token_query, [GroupA_output | GroupC_output])
#         gate_B(token) = softmax(W_g · [B_token || context_vec || t_embed])

#     Stop-gradient:
#         Group C output is detached before the cross-attention query step when
#         cfg.stopgrad_C_to_B = True.  This prevents routing gradient from
#         flowing back into Group C representations during early training.

#     Args:
#         cfg: MoEConfig
#     """
#     def __init__(self, cfg: MoEConfig):
#         super().__init__()
#         self.num_experts = cfg.num_experts_B
#         self.top_k = cfg.top_k_B
#         self.capacity_factor = cfg.capacity_factor_B
#         self.routing_floor = cfg.routing_floor
#         self.stopgrad_C_to_B = cfg.stopgrad_C_to_B
#         self.embed_dim = cfg.embed_dim

#         # Cross-attention: Group B tokens query (Group A | Group C) context
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=cfg.embed_dim,
#             num_heads=cfg.gate_num_heads,
#             batch_first=True,
#             dropout=0.0,
#         )
#         # cross attention : Group B tokens query (Group A | group C) context 
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim= cfg.embed_dim, 
#             num_heads= cfg.gate_num_heads,
#             batch_first= True,
#             dropout=0.0
#         )
#         self.cross_attn_norm = nn.LayerNorm(cfg.embed_dim)

#         # diffusion step embedding
#         self.t_embed = TimestepEmbedding(cfg.embed_dim)

#         # gate mlp : [B_token (D) | context (D) | t_embed (D)] → expert logits
#         self.gate_mlp = nn.Sequential(
#             nn.Linear(cfg.embed_dim * 3, cfg.embed_dim),
#             nn.ReLU(),
#             nn.Linear(cfg.embed_dim, self.num_experts)
#         )

#         # Shared expert anchor (weight 0.1-0.15 from design conversation)
#         # One "generic" expert that always receives a small fraction of each token's
#         # routing probability, stabilising training and providing a fallback.
#         self.shared_expert_weight = 0.12   # constant, not learned

#     def forward(
#                 self, 
#                 token_B: torch.Tensor, # (B, N_B, D)
#                 output_A: torch.Tensor, # (B, N_A, D) - group A final output
#                 output_C: torch.Tensor, # (N, N_C, D) - group C final output
#                 t: torch.Tensor,        # (B,) int or float in [0, T_max]
#         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Returns:
#             dispatch_weights:  (B, N_B, E)
#             combine_weights:   (B, N_B, E)
#             capacity_penalty:  scalar
#         """
#         B, N_B, D = token_B.shape
#         E = self.num_experts

#         # stop gradient on Group C output when entering gate path
#         ctx_C = output_C.detach() if self.stopgrad_C_to_B else output_C

#         # build cross group context: concatenate A and C along token dim
#         ctx_AC = torch.cat([output_A, output_C], dim = 1) # (B, N_A + N_C, D)

#         # cross attention: B token query AC context
#         # query = tokens_B, key_values = ctx_AC
#         ctx_B, _ = self.cross_attn(
#             query = token_B, 
#             key = ctx_AC, 
#             value = ctx_AC
#         )               # (B,N_B, D)

#         ctx_B = self.cross_attn_norm(ctx_B + token_B) # residual

#         # timestep embedding
#         t_emb = self.t_embed(t) # (B, D)

#         t_emb = t_emb.unsqueeze(1).expand(-1, N_B, -1) # (B, N_B, D)

#         # gate MLP over [token | cross group context | t_embed]
#         gate_input = torch.cat([token_B, ctx_B, t_emb], dim = -1) # (B, N_B, 3D)
#         logits = self.gate_mlp(gate_input)          # (B, N_B, E)

#         # probabilities with routing floor
#         probs = self._apply_routing_floor(F.softmax(logits, dim = -1)) # (B, N_B, E)
        
#         # inject shared expert anchor.
#         #    Expert index 0 is designated the "shared" expert.
#         #    All tokens are guaranteed shared_expert_weight to expert 0,
#         #    and the rest is re-normalised across the remaining experts.        
        
#         probs = self._inject_shared_expert(probs)

#         # topk masking
#         dispatch_weights = self._topk_mask(probs, self.top_k)

#         # soft capacity penalty
#         capacity_penalty = self._soft_capacity_penalty(dispatch_weights, N_B)

#         # normalize combine weights
#         combine_weights = dispatch_weights / (dispatch_weights.sum(-1, keepdim = True))
        
#         return dispatch_weights, combine_weights, capacity_penalty
    

#     def _inject_shared_expert(self, probs: torch.Tensor) -> torch.Tensor:
#         """Guarantee shared_expert_weight to expert index 0, re-normalise rest.

#         E.g. if shared_expert_weight=0.12, expert 0 always gets 0.12 of the
#         probability mass, and experts 1..E-1 share the remaining 0.88.
#         """
#         w = self.shared_expert_weight
#         # Clamp existing prob of expert 0 to at least w
#         shared_prob = probs[..., 0:1]                              # (B, N, 1)
#         other_probs = probs[..., 1:]                               # (B, N, E-1)

#         # Redistribute remaining budget
#         new_shared = torch.clamp(shared_prob, min=w)
#         remaining  = 1.0 - new_shared
#         other_sum  = other_probs.sum(-1, keepdim=True).clamp(min=1e-8)
#         new_others = other_probs / other_sum * remaining

#         return torch.cat([new_shared, new_others], dim=-1)

#     def _apply_routing_floor(self, probs: torch.Tensor) -> torch.Tensor:
#         E = probs.shape[-1]
#         uniform = torch.full_like(probs, 1.0 / E)
#         alpha = min(self.routing_floor * E, 1.0)
#         return (1 - alpha) * probs + alpha * uniform

#     def _topk_mask(self, probs: torch.Tensor, k: int) -> torch.Tensor:
#         topk_vals, topk_idx = probs.topk(k, dim=-1)
#         mask = torch.zeros_like(probs)
#         mask.scatter_(-1, topk_idx, topk_vals)
#         return mask

#     def _soft_capacity_penalty(
#         self, dispatch_weights: torch.Tensor, num_tokens: int
#     ) -> torch.Tensor:
#         E = self.num_experts
#         fair_share = (num_tokens * self.top_k) / E
#         capacity   = fair_share * self.capacity_factor
#         expert_load = dispatch_weights.sum(dim=1)
#         excess = F.relu(expert_load - capacity)
#         return (excess ** 2).mean()

# group B -  scene-conditionod soft router
class GroupBRouter(nn.Module):
    """Scene-conditioned soft router for Group B with full internal pipeline.
 
    Integrates: Stage 1 ego-centric attention, Stage 2 agent-agent attention,
    A->B and C->B cross-attention, Stage 3 map re-weighting, intention heads.
 
    Gate query (Part 2.8): [agent_repr_3 | cross_A | cross_C | t_embed | intention_logits]
    Temperature tau(t) (Part 2.8), shared expert w_shared(t) (Part 2.9).
    Two-layer capacity (Part 4.2): hard cap + bias scalars.
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_B
        self.top_k = cfg.top_k_B
        self.stopgrad_C_to_B = cfg.stopgrad_C_to_B
        self.embed_dim = cfg.embed_dim
        self.T_max = cfg.T_max
        self._floor = cfg.routing_floor
        self.tau_min = cfg.tau_min
        self.tau_max = cfg.tau_max
        self.shared_w_min = cfg.shared_weight_min
        self.shared_w_max = cfg.shared_weight_max

        # internal pipeline stages
        self.stage1_ego_attn = EgoCentricCrossAttention(cfg)
        self.stage2_ego_attn = EgoProximityAgentAttention(cfg)

        #between-group cross-attention : A-> B (separate from C -> B)
        self.cross_attn_A_to_B = nn.MultiheadAttention(cfg.embed_dim, cfg.gate_num_heads, batch_first=True, dropout=0.0)
        self.cross_attn_A_to_B_norm = nn.LayerNorm(cfg.embed_dim)

        # Between-group cross-attention: C -> B
        self.cross_attn_C_to_B = nn.MultiheadAttention(cfg.embed_dim, cfg.gate_num_heads, batch_first=True, dropout=0.0)
        self.cross_attn_C_to_B_norm = nn.LayerNorm(cfg.embed_dim)

        # map context re-weighting
        self.stage3_map_reweight = MapContextReweighting(cfg)

        # intention prediction heads
        self.intention_head = IntentionHeads(embed_dim=cfg.embed_dim)

        # timestepembeding
        self.t_embed = TimestepEmbedding(cfg.embed_dim)

        # gate MLP [repr3 | cross_A | cross_C | t_emb | intention]
        #       gate_input_dim must include IntentionHeads.output_dim (= 6) in the
        #       gate_mlp Linear input size (already correct if using output_dim=6).
        intention_dim = self.intention_head.output_dim
        gate_input_dim = cfg.embed_dim * 4 + intention_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, self.num_experts)
        )

        # layer 2: per-expert bias scalar (part 4.2)
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)
        self.bias_penalty_coeff = cfg.bias_penalty_coeff

    def _gate_temperature(self, t):
        t_norm = t.float() / self.T_max
        tau = self.tau_min + (self.tau_max - self.tau_min) * t_norm

        return tau.view(-1, 1, 1)
    
    def _shared_expert_weight(self, t):
        t_norm = t.float() / self.T_max
        w = self.shared_w_min + (self.shared_w_max - self.shared_w_min) * t_norm
        return w.view(-1, 1, 1)
    
    def forward(self,
                tokens_B, 
                output_A, 
                output_C, 
                t,
                # add parametes in exchange for is_pedestrian
                agent_types: torch.Tensor, 
                ego_mask = None, 
                ego_distances = None, 
                ego_speed = None, 
                # is_pedestrian = None,
                ):
        B, N_B, D = tokens_B.shape

        # stage 1: ego-centric cross - attention
        if ego_mask is not None:
            agent_repr_1 = self.stage1_ego_attn(tokens_B, ego_mask)

        else:
            agent_repr_1 = tokens_B

        # step 2: ego-proximity-filtered agent-agent attention
        if ego_distances is not None and ego_mask is not None and ego_speed is not None:
            agent_repr_2 = self.stage2_ego_attn(agent_repr_1, ego_distances, ego_mask, ego_speed)
            # please consider that do we need relative and grid distance that we can make for distance correctly between agent
        
        else:
            agent_repr_2 = agent_repr_1

        
        # between group cross attention
        ctx_C = output_C.detach() if self.stopgrad_C_to_B else output_C
        A_det = output_A.detach() # stop-gradient on A (stable anchor)

        cross_A_out, _ = self.cross_attn_A_to_B(query = agent_repr_2, key = A_det, value = A_det)
        cross_A_out = self.cross_attn_A_to_B_norm(agent_repr_2 + cross_A_out)

        cross_C_out, _ = self.cross_attn_C_to_B(query = agent_repr_2, key =ctx_C, value = ctx_C)
        cross_C_out = self.cross_attn_C_to_B_norm(agent_repr_2 + cross_C_out)

        # stage 3: map context reweighting
        agent_repr_3 = self.stage3_map_reweight(agent_repr_2, cross_C_out)

        # intention prediction (part 2.7)
        # this part is wrong, we need to calculate, or not, still in relevant cause pedestrian is equal to 0.
        # if is_pedestrian is None:
        #     is_pedestrian = torch.zeros(B, N_B, dtype=torch.bool, device=tokens_B.device)
        #     intention_for_gate, vehicle_logits, ped_logits = self.intention_head(agent_repr_2, is_pedestrian)
        intention_for_gate,_ = self.intention_head(
            agent_repr = agent_repr_2,
            agent_types = agent_types,
            intention_gt = None, # loss computed in MoEBlock, not here
        )
        
        # Timestep embedding
        t_emb = self.t_embed(t)
        t_emb_exp = t_emb.unsqueeze(1).expand(-1, N_B, -1)

        # Gate query (Part 2.8): full concat
        gate_input = torch.cat([agent_repr_3, cross_A_out, cross_C_out, t_emb_exp, intention_for_gate], dim=-1)
        logits = self.gate_mlp(gate_input)
 
        # Add per-expert bias scalars (Layer 2)
        logits = logits + self.expert_bias
 
        # Gate temperature (Part 2.8)
        tau = self._gate_temperature(t)
        probs = F.softmax(logits / tau, dim=-1)
 
        # Routing floor
        probs = _apply_routing_floor(probs, self._floor)
 
        # Shared expert anchor (Part 2.9)
        probs = self._inject_shared_expert(probs, t)
 
        # Top-k masking
        dispatch = _topk_mask(probs, self.top_k)
 
        # Layer 1: Token-level hard cap
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)
 
        # Layer 2: Batch-level bias penalty
        bias_penalty = self._batch_bias_penalty(dispatch, N_B)
 
        # Normalise combine weights
        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
 
        return dispatch, combine, bias_penalty
 
    def _inject_shared_expert(self, probs, t):
        w = self._shared_expert_weight(t)
        shared_prob = probs[..., 0:1]
        other_probs = probs[..., 1:]
        new_shared = torch.clamp(shared_prob, min=w)
        remaining = 1.0 - new_shared
        other_sum = other_probs.sum(-1, keepdim=True).clamp(min=1e-8)
        new_others = other_probs / other_sum * remaining
        return torch.cat([new_shared, new_others], dim=-1)
 
    def _batch_bias_penalty(self, dispatch, num_tokens):
        E = self.num_experts
        fair_share = num_tokens / max(E - 1, 1)
        expert_load = dispatch.sum(dim=1).mean(dim=0)
        specialist_load = expert_load[1:]
        excess = F.relu(specialist_load - fair_share * 1.5)
        return self.bias_penalty_coeff * (excess ** 2).sum()



# time embedding (sinusodial, used by Group B gate)
# place holder for timestepEmbedding, please consider that we already have 

class TimestepEmbedding(nn.Module):
    """Maps integer/float diffusion timestep t → vector of shape (B, D).

    Uses sinusoidal embedding (same as original DDPM) followed by a 2-layer MLP.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        self.register_buffer(
            "freq",
            torch.exp(-math.log(10000) * torch.arange(half_dim) / (half_dim - 1))
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float or long tensor in [0, T_max]
        Returns:
            emb: (B, embed_dim)
        """
        t = t.float()
        args = t[:, None] * self.freq[None, :]   # (B, half_dim)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)
        return self.proj(emb)
    

# DyDiT skip scheduler

class DyDiTSkipScheduler:
    """
    Decides per-group skip based on diffusion timestep t.

    Opposite schedules for B and C (core insight from design conversation).
        Group C: most useful at HIGH t (early denoising, mode selection).
                 → skip when t/T_max < t_skip_C  (late denoising, mode committed)

        Group B: most useful at LOW t (late denoising, fine trajectory shaping).
                 → skip when t/T_max > t_skip_B  (early denoising, coarse structure)

        Group A: timestep-independent (raw sensor).
                 → cache and reuse every cache_interval_A steps.

    No learnable parameters — purely a decision function applied per forward pass.

    Args:
        cfg: MoEConfig 
    
    """

    def __init__(self, cfg : MoEConfig):
        self.t_skip_C = cfg.t_skip_C
        self.t_skip_B = cfg.t_skip_B
        self.T_max = cfg.T_max
        self.cache_interval_A = cfg.cache_interval_A

    def skip(
            self,
            group: int,
            t: torch.Tensor,
    ) -> torch.BoolTensor:
        """Return a boolean tensor (B,) indicating which batch elements skip.

        Args:
            group: GROUP_A, GROUP_B, or GROUP_C
            t:     (B,) diffusion timestep tensor
        Returns:
            skip: (B,) bool — True means skip expert computation for this sample
        """
        t_norm = t.float() / self.T_max # normalize to [0, 1]

        if group == GROUP_C:
            # Skip at late denoising (low t/T_max)
            return t_norm < self.t_skip_C

        elif group == GROUP_B:
            # Skip at early denoising (high t/T_max)
            return t_norm > self.t_skip_B

        elif group == GROUP_A:
            # Group A: skip logic handled externally via cache_interval_A.
            # Here we return False (never skip, use cache logic in MoERouter).
            return torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)

        else:
            raise ValueError(f"Unknown group id: {group}")        
        
    def cache_step_for_A(self, step: int) -> bool:
        """True if Group A should recompute this step (else use cached output)."""
        return (step % self.cache_interval_A) == 0
    
# group tokenizer

class GroupTokenizer(nn.Module):
    """Partitions raw token tensors into Group A / B / C and adds group identity.

    At this stage tokens arrive pre-encoded from the upstream pipeline
    (GaussianFormer pooling → Gaussian tokens, agent encoder → agent tokens, etc.)
    This module:
        1. Projects each token set to the shared embed_dim if needed.
        2. Adds the per-group orthogonal identity embedding.
        3. Returns the three groups ready for routing.

    Args:
        cfg:      MoEConfig
        dim_A_in: input feature dim for Group A tokens (from pooling stage)
        dim_B_in: input feature dim for Group B tokens (from agent encoder)
        dim_C_in: input feature dim for Group C tokens (from map encoder)
    """


    def __init__(
        self,
        cfg: MoEConfig,
        dim_A_in: int,
        dim_B_in: int,
        dim_C_in: int,
    ):
        super().__init__()
        D = cfg.embed_dim
        self.proj_A = nn.Linear(dim_A_in, D) if dim_A_in != D else nn.Identity()
        self.proj_B = nn.Linear(dim_B_in, D) if dim_B_in != D else nn.Identity()
        self.proj_C = nn.Linear(dim_C_in, D) if dim_C_in != D else nn.Identity()
        self.group_id_emb = GroupIdentityEmbedding(D, num_groups=3)

    def forward(
        self,
        raw_A: torch.Tensor,    # (B, N_A, dim_A_in)
        raw_B: torch.Tensor,    # (B, N_B, dim_B_in)
        raw_C: torch.Tensor,    # (B, N_C, dim_C_in)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens_A: (B, N_A, D)
            tokens_B: (B, N_B, D)
            tokens_C: (B, N_C, D)
        """
        tA = self.group_id_emb(self.proj_A(raw_A), GROUP_A)
        tB = self.group_id_emb(self.proj_B(raw_B), GROUP_B)
        tC = self.group_id_emb(self.proj_C(raw_C), GROUP_C)
        return tA, tB, tC

    def orthogonality_loss(self) -> torch.Tensor:
        return self.group_id_emb.orthogonality_loss()
    

# top level ModalityMoERRouter

@dataclass
class RouterOutput:
    """All routing decisions, weights, and diagnostics from one forward pass."""
    dispatch_A:     torch.Tensor
    combine_A:      torch.Tensor
    dispatch_B:     torch.Tensor
    combine_B:      torch.Tensor
    dispatch_C:     torch.Tensor
    combine_C:      torch.Tensor
    skip_A:         torch.BoolTensor
    skip_B:         torch.BoolTensor   # (B,) or (B, N_B) for learned predictor
    skip_C:         torch.BoolTensor   # (B,) True = skip (use cache)
    # Group C two-level cache zone indicator (Part 4.5):
    #   0 = COMPUTE (recompute window between t_cache_low and t_cache_high)
    #   1 = USE_HIGH_CACHE (t >= t_cache_high, mode-selection era)
    #   2 = USE_LOW_CACHE (t < t_cache_low, fine-shaping era)
    # Only meaningful when scene_complexity was provided; otherwise -1.
    skip_C_zone:    torch.Tensor       # (B,) long: 0=compute, 1=high_cache, 2=low_cache, -1=N/A
    bias_penalty:   torch.Tensor
    ortho_loss:     torch.Tensor
    vehicle_logits: Optional[torch.Tensor]
    ped_logits:     Optional[torch.Tensor]
    attn_mask:      torch.Tensor
    routing_entropy_A: torch.Tensor
    routing_entropy_B: torch.Tensor
    routing_entropy_C: torch.Tensor


class ModalityMoERouter(nn.Module):
    """Top-level router that wires GroupTokenizer, three group routers, and skip scheduler.

    This is the entry point used by GaussianTransDiffuser (or any decoder that
    needs to dispatch tokens to modality-specific expert groups).

    Usage example:
        router = ModalityMoERouter(cfg, dim_A_in=256, dim_B_in=128, dim_C_in=256)
        out = router(
            raw_A=gaussian_tokens,      # (B, N_A, 256)
            raw_B=agent_tokens,         # (B, N_B, 128)
            raw_C=map_tokens,           # (B, N_C, 256)
            spatial_xyz=gauss_means,    # (B, N_A, 3)   for Group A spatial gate
            token_types_C=ttype_ids,    # (B, N_C) long for Group C structural gate
            t=diffusion_t,              # (B,)      diffusion timestep
            step=global_step,           # int       training step (for A cache)
            output_A=None,              # pass Group A expert output once available
            output_C=None,              # pass Group C expert output once available
        )
        # Then dispatch tokens to experts using out.dispatch_A / combine_A, etc.

    Two-pass forward (required by directed attention):
        Pass 1: route Group A (no cross-group context needed)
        Pass 2: route Group C (conditioned on Group A output from expert computation)
        Pass 3: route Group B (conditioned on Group A + Group C expert outputs)
        This router computes all three route plans, but the caller must run expert
        FFNs between passes to produce the actual output_A and output_C.
        See forward() docstring for the full protocol.

    Args:
        cfg:      MoEConfig
        dim_A_in: raw input dim for Group A tokens
        dim_B_in: raw input dim for Group B tokens
        dim_C_in: raw input dim for Group C tokens
    """

    def __init__(
            self, 
            cfg: MoEConfig,
            dim_A_in: int = 256,
            dim_B_in: int = 256,
            dim_C_in: int = 256,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = GroupTokenizer(cfg, dim_A_in, dim_B_in, dim_C_in)
        self.ln_groups = GroupLocalLayerNorm(cfg.embed_dim)

        # router
        self.router_A = GroupARouter(cfg)
        self.router_C = GroupCRouter(cfg)
        self.router_B = GroupBRouter(cfg)


        self.skip_sched = DyDiTSkipScheduler(cfg)
        self.warmup_cross_attn = WarmUpCrossAttention(cfg)
        self.skip_predictor_B = LearnedSkipPredictor(cfg)
        self.cache_C = GroupCTwoLevelCache(cfg)

        # static attention mask - register as buffer (no grad, moves with device)
        mask_builder = DirectedAttentionMask(
            cfg.num_tokens_A, cfg.num_tokens_C, cfg.num_tokens_B
        )

        self.register_buffer(
            "attn_mask",
            mask_builder.build_additive()
        )

        # Group A feature cache (for timestep-independent sensory features)
        self._cache_A: Optional[torch.Tensor] = None
        self._cache_step: int = -1

    def forward(
            self,
            raw_A: torch.Tensor,            # (B, N_A, dim_A_in)
            raw_B: torch.Tensor,            # (B, N_B, dim_B_in)
            raw_C: torch.Tensor,            # (B, N_C, dim_C_in)
            spatial_xyz: torch.Tensor,      # (B, N_A, 3)   group A spatial coordinates 
            token_types_C: torch.Tensor,    # (B, N_C) long group C token type IDs
            t: torch.Tensor,                # (B,)  diffusion steps
            output_A: Optional[torch.Tensor] = None, # (B, N_A, D) group A expert output
            output_C: Optional[torch.Tensor] = None, # (B, N_C, D) group c expert output
            step: int = 0,                  # global training step (for A cache)
            ego_mask = None,
            ego_distance = None,
            ego_speed = None, 
            # is_pedestrian = None, 
            agent_types = None,
            scene_complexity = None,
            rare_token_mask_B = None,
            use_learned_skip_B = False,

    ) -> RouterOutput:
        """Compute all routing decisions for one forward pass.

        IMPORTANT — calling protocol (two-pass pattern required by caller):

            # Pass 1: get routing for Group A (no cross-group context yet)
            out1 = router(raw_A, raw_B, raw_C, ..., output_A=None, output_C=None)
            # → use out1.dispatch_A / combine_A to run Group A experts → get expert_output_A

            # Pass 2: get routing for Group C (conditioned on A)
            out2 = router(raw_A, raw_B, raw_C, ..., output_A=expert_output_A, output_C=None)
            # → use out2.dispatch_C / combine_C to run Group C experts → get expert_output_C

            # Pass 3: get routing for Group B (conditioned on A + C)
            out3 = router(raw_A, raw_B, raw_C, ..., output_A=expert_output_A, output_C=expert_output_C)
            # → use out3.dispatch_B / combine_B to run Group B experts

        In practice, a single forward call returns routing for whatever groups
        have their upstream context available:
            - output_A=None, output_C=None  →  only Group A routing is valid
            - output_A provided, output_C=None  →  A+C routing valid
            - output_A+output_C provided  →  all routing valid

        Returns RouterOutput with all routing decisions, losses, and diagnostics.
        """

        # Tokenize and group-local normalize
        tokens_A, tokens_B, tokens_C = self.tokenizer(raw_A, raw_B, raw_C)
        tokens_A, tokens_B, tokens_C = self.ln_groups(tokens_A, tokens_B, tokens_C)

        # warm-up cross-attention (part 3.3) -- before any routing
        tokens_A, tokens_B, tokens_C = self.warmup_cross_attn(tokens_A, tokens_B, tokens_C)

        # skip decisions
        skip_A = self.skip_sched.skip(GROUP_B, t)

        # group C skip: two-level cache with 3 zones (part 4.5)
        # zone 0 = compute, zone 1 = use_high_cache, zone 2 = use_low_cache, -1 = N/A
        B_sz = t.shape[0]
        if scene_complexity is not None:
            is_compute = self.cache_C.compute(t, scene_complexity)
            high = self.cache_C.use_high_cache(t, scene_complexity)
            low = self.cache_C.use_low_cache(t)
            skip_C = ~is_compute #  True = skip (either high or low cache)

            # build zone indicator: 0 = compute, 1 = high_cache, 2 = low_cache
            skip_C_zone = torch.zeros(B_sz, dtype=torch.long, device=t.device)
            skip_C_zone[high] = 1
            skip_C_zone[low] = 2
            # Zone 0 stays for the recompute window (default)
        else:
            skip_C = self.skip_sched.skip(GROUP_C, t)
            skip_C_zone = torch.full((B_sz,), -1, dtype=torch.long, device=t.device)

        if use_learned_skip_B and output_C is not None:
            t_emb = self.router_B.t_embed(t)
            ctx_C_summary = output_C.mean(dim=1, keepdim=True).expand(-1, tokens_B.shape[1], -1)
            skip_B = self.skip_predictor_B(tokens_B, ctx_C_summary, t_emb, t, rare_token_mask_B)
        else:
            skip_B = self.skip_sched.skip(GROUP_B, t)
 
        # Route Group A
        dispatch_A, combine_A = self.router_A(tokens_A, spatial_xyz, t)
 
        # Route Group C
        dispatch_C, combine_C = self.router_C(tokens_C, token_types_C, t)
 
        # Route Group B (needs cross-group context)
        bias_penalty = torch.tensor(0.0, device=tokens_B.device)
        vehicle_logits = None
        ped_logits = None
 
        if output_A is not None and output_C is not None:
            dispatch_B, combine_B, bias_penalty = self.router_B(
                tokens_B, output_A, output_C, t,
                agent_types = agent_types,
                ego_mask=ego_mask, ego_distances=ego_distance,
                ego_speed=ego_speed,
            )
        else:
            B_sz, N_B = tokens_B.shape[:2]
            E_B = self.cfg.num_experts_B
            uniform = torch.full((B_sz, N_B, E_B), 1.0 / E_B, device=tokens_B.device)
            dispatch_B = combine_B = uniform
 
        ortho_loss = self.cfg.ortho_reg_weight * self.tokenizer.orthogonality_loss()
 
        return RouterOutput(
            dispatch_A=dispatch_A, combine_A=combine_A,
            dispatch_B=dispatch_B, combine_B=combine_B,
            dispatch_C=dispatch_C, combine_C=combine_C,
            skip_A=skip_A, skip_B=skip_B, skip_C=skip_C,
            skip_C_zone=skip_C_zone,
            bias_penalty=bias_penalty, ortho_loss=ortho_loss,
            vehicle_logits=vehicle_logits, ped_logits=ped_logits,
            attn_mask=self.attn_mask,
            routing_entropy_A=self._routing_entropy(dispatch_A),
            routing_entropy_B=self._routing_entropy(dispatch_B),
            routing_entropy_C=self._routing_entropy(dispatch_C),
        )

        # # skip decisions
        # skip_A = self.skip_sched.skip(GROUP_A, t)
        # skip_B = self.skip_sched.skip(GROUP_B, t)
        # skip_C = self.skip_sched.skip(GROUP_C, t)

        # # route group A
        # dispatch_A, combine_A, cap_loss_A = self.router_A(tokens_A, spatial_xyz)

        # # route group C (conditioned on Group A output if provided)
        # #    If output_A is not yet available, we still compute Group C routing
        # #    but pass a zero tensor as Group A context.  The caller should
        # #    prefer to call in the two-pass pattern described above.

        # dispatch_C, combine_C, cap_loss_C = self.router_C(tokens_C, token_types_C)

        # # router group B
        # if output_A is not None and output_C is not None:
        #     dispatch_B, combine_B, cap_loss_B = self.router_B(
        #         tokens_B, output_A, output_C, t
        #     )

        # else:
        #     # Placeholder routing (uniform) until cross-group context is available
        #     B, N_B = tokens_B.shape[:2]
        #     E_B = self.cfg.num_experts_B
        #     uniform = torch.full(
        #         (B, N_B, E_B), 1.0 / E_B, device=tokens_B.device
        #     )
        #     dispatch_B = combine_B = uniform
        #     cap_loss_B = torch.tensor(0.0, device=tokens_B.device)

        # # loss
        # capacity_loss = (
        #     self.cfg.capacity_penalty_coeff * (cap_loss_A + cap_loss_B + cap_loss_C)
        # )

        # ortho_loss = (
        #     self.cfg.ortho_reg_weight * self.tokenizer.orthogonality_loss()
        # )

        # # routing entropy (fopr logging/ probes)
        # ent_A = self._routing_entropy(dispatch_A)
        # ent_B = self._routing_entropy(dispatch_B)
        # ent_C = self._routing_entropy(dispatch_C)

        # return RouterOutput(
        #     dispatch_A=dispatch_A,
        #     combine_A=combine_A,
        #     dispatch_B=dispatch_B,
        #     combine_B=combine_B,
        #     dispatch_C=dispatch_C,
        #     combine_C=combine_C,
        #     skip_A=skip_A,
        #     skip_B=skip_B,
        #     skip_C=skip_C,
        #     capacity_loss=capacity_loss,
        #     ortho_loss=ortho_loss,
        #     attn_mask=self.attn_mask,
        #     routing_entropy_A=ent_A,
        #     routing_entropy_B=ent_B,
        #     routing_entropy_C=ent_C,
        # )

    @staticmethod
    def _routing_entropy(dispatch: torch.Tensor) -> torch.Tensor:
        """Mean routing entropy (bits) across batch and tokens.  Lower = more decisive."""
        # Normalise rows to sum to 1 (some entries may be zero from top-k)
        p = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
        # Entropy: -sum p log p  (add eps to avoid log(0))
        ent = -(p * (p + 1e-8).log()).sum(-1)   # (B, N)
        return ent.mean().detach()

# Utility: build router from config dict / kwargs

def build_moe_router(
    embed_dim: int = 256,
    num_tokens_A: int = 64,
    num_tokens_B: int = 64,
    num_tokens_C: int = 128,
    num_experts: int = 4,
    dim_A_in: int = 256,
    dim_B_in: int = 256,
    dim_C_in: int = 256,
    **kwargs,
) -> ModalityMoERouter:
    """Convenience factory that fills in a MoEConfig and returns a router.

    Extra kwargs are forwarded to MoEConfig (e.g. stopgrad_C_to_B=False for
    late fine-tuning).
    """
    cfg = MoEConfig(
        embed_dim=embed_dim,
        num_tokens_A=num_tokens_A,
        num_tokens_B=num_tokens_B,
        num_tokens_C=num_tokens_C,
        num_experts_A=num_experts,
        num_experts_B=num_experts,
        num_experts_C=max(num_experts, NUM_C_TYPES),
        **{k: v for k, v in kwargs.items() if hasattr(MoEConfig, k)},
    )
    return ModalityMoERouter(cfg, dim_A_in=dim_A_in, dim_B_in=dim_B_in, dim_C_in=dim_C_in)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run as main)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 2

    cfg = MoEConfig(
        num_tokens_A=64, num_tokens_B=32, num_tokens_C=64,
        embed_dim=128,
        num_experts_A=4, num_experts_B=4, num_experts_C=6,
    )

    router = ModalityMoERouter(cfg, dim_A_in=128, dim_B_in=64, dim_C_in=128)

    raw_A = torch.randn(B, cfg.num_tokens_A, 128)
    raw_B = torch.randn(B, cfg.num_tokens_B, 64)
    raw_C = torch.randn(B, cfg.num_tokens_C, 128)
    xyz   = torch.randn(B, cfg.num_tokens_A, 3) * 20.0
    ttypes = torch.randint(0, NUM_C_TYPES, (B, cfg.num_tokens_C))
    t_vec  = torch.randint(0, 1000, (B,))

    # Pass 1: route A only
    out = router(raw_A, raw_B, raw_C, xyz, ttypes, t_vec)
    print(f"dispatch_A shape : {out.dispatch_A.shape}")    # (2, 64, 4)
    print(f"dispatch_C shape : {out.dispatch_C.shape}")    # (2, 64, 6)
    print(f"dispatch_B shape : {out.dispatch_B.shape}")    # (2, 32, 4)  uniform (no ctx)
    print(f"attn_mask shape  : {out.attn_mask.shape}")     # (160, 160)
    # print(f"capacity_loss    : {out.capacity_loss.item():.6f}")
    print(f"ortho_loss       : {out.ortho_loss.item():.6f}")
    print(f"entropy_A        : {out.routing_entropy_A.item():.4f}")
    print(f"entropy_C        : {out.routing_entropy_C.item():.4f}")
    print(f"skip_B (t>0.7)   : {out.skip_B.tolist()}")
    print(f"skip_C (t<0.2)   : {out.skip_C.tolist()}")
    print("All shapes OK.")

    # Verify attention mask structure
    mask = router.attn_mask
    N_A, N_C, N_B = cfg.num_tokens_A, cfg.num_tokens_C, cfg.num_tokens_B
    # A→C should be blocked (A queries should NOT attend to C keys)
    sA = slice(0, N_A)
    sC = slice(N_A, N_A + N_C)
    sB = slice(N_A + N_C, N_A + N_C + N_B)
    assert torch.all(mask[sA, sC] == float("-inf")), "FAIL: A should not attend to C"
    assert torch.all(mask[sA, sB] == float("-inf")), "FAIL: A should not attend to B"
    assert torch.all(mask[sC, sB] == float("-inf")), "FAIL: C should not attend to B"
    assert torch.all(mask[sB, sA] == 0.0),           "FAIL: B should attend to A"
    assert torch.all(mask[sB, sC] == 0.0),           "FAIL: B should attend to C"
    print("Attention mask directionality: PASS")
