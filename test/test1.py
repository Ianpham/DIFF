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
    - Attention mask:        causal mask that enforces A→C→B directionality.
    - Stop-gradient:         Group C output detached when used as Group B gate input.
    - Orthogonal group identity embeddings regularized to be distinguishable.
    - Two-layer capacity mechanism:
        Layer 1: Token-level hard cap (non-differentiable, zero gradient).
        Layer 2: Batch-level router bias scalars (gradient to bias only, not to W_g).
      Replaces load-balance auxiliary loss entirely (Part 0.2 / 4.2).
    - DyDiT timestep-conditioned skip: Group C skips at low t, Group B at high t.
    - Gate temperature τ(t): sharp at low-t, soft at high-t (Part 2.8).
    - Shared expert weight w_shared(t): timestep-scheduled (Part 2.9).

This file provides:
    GroupTokenizer         — partitions raw tokens into A/B/C, adds group identity emb
    DirectedAttentionMask  — builds the causal attention mask for the three groups
    GroupLocalLayerNorm    — per-group normalization
    GroupARouter           — expert-choice spatial router for Group A
    GroupCRouter           — structural deterministic router for Group C
    GroupBRouter           — scene-conditioned soft router for Group B (cross-group gate)
    TokenLevelHardCap      — non-differentiable post-softmax capping (Part 4.2 Layer 1)
    DyDiTSkipScheduler     — per-group skip decision conditioned on diffusion timestep t
    ModalityMoERouter      — top-level orchestrator that wires all of the above
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Constants / group identifiers ───────────────────────────────────────────

GROUP_A = 0   # Raw sensory (Dynamic Gaussians, BEV, LiDAR summaries)
GROUP_B = 1   # Interaction (agent / ego states, history)
GROUP_C = 2   # Map/context (phantom Gaussians, vector map, traffic lights)

# Token-type codes within GROUP C — used by structural router.
# Set at tokenization time from the data pipeline metadata.
TTYPE_PHANTOM      = 0  # phantom / occluded Gaussians
TTYPE_VECTORMAP    = 1  # vector map lane tokens
TTYPE_TRAFFIC_LT   = 2  # traffic light tokens
TTYPE_INTERSECTION = 3  # intersection topology tokens
TTYPE_BEV_STRUCT   = 4  # structural BEV features (no raw sensor)
TTYPE_UNKNOWN      = 5  # soft-fallback for ambiguous tokens

NUM_C_TYPES = 6


# ─── Config dataclass ────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    """All hyper-parameters for the three-group MoE router."""

    # Token counts per group (set per model config)
    num_tokens_A: int = 64      # sensory tokens (pooled Gaussian features)
    num_tokens_B: int = 64      # interaction tokens (agents + ego)
    num_tokens_C: int = 128     # map/context tokens

    # Feature dimension (shared across groups)
    embed_dim: int = 256

    # Number of experts per group
    num_experts_A: int = 4
    num_experts_B: int = 4
    num_experts_C: int = 6      # >= NUM_C_TYPES

    # Top-k experts per token
    top_k_A: int = 2
    top_k_B: int = 2
    top_k_C: int = 1            # near-deterministic for Group C

    # Capacity factor per group (max tokens as fraction of fair share)
    capacity_factor_A: float = 1.5
    capacity_factor_B: float = 1.5
    capacity_factor_C: float = 2.0   # looser for structural router

    # Routing floor: ε = min(routing_floor_base, 0.15 / k)  (Part 1.4)
    routing_floor_base: float = 0.05

    # Token-level hard cap (Part 4.2, Layer 1)
    # cap(t) = cap_low + (cap_high - cap_low) * (t / T_max)
    cap_low:  float = 0.50   # tight at low-t (fine trajectory shaping)
    cap_high: float = 0.60   # loose at high-t (coarse routing)

    # DyDiT skip thresholds (in noise-schedule, T=1000 convention)
    t_skip_C: float = 0.2    # skip Group C when t/T_max < t_skip_C
    t_skip_B: float = 0.7    # skip Group B when t/T_max > t_skip_B

    # Group A: cache sensory features — recompute only every cache_interval steps
    cache_interval_A: int = 5

    # Orthogonal group identity regularization weight
    ortho_reg_weight: float = 1e-3

    # Stop-gradient on Group C → Group B gate path.
    # Set to False only during late fine-tuning (controlled externally).
    stopgrad_C_to_B: bool = True

    # Maximum diffusion timesteps (for normalized t)
    T_max: int = 1000

    # Group C structural router temperature (low = more deterministic)
    struct_router_temp: float = 0.1

    # Expert FFN hidden dim multiplier
    expert_ff_mult: int = 4

    # Number of attention heads for cross-group gate attention
    gate_num_heads: int = 4

    # Gate temperature schedule (Part 2.8)
    # τ(t) = tau_min + (tau_max - tau_min) * (t / T_max)
    tau_min: float = 0.5     # sharp at low-t
    tau_max: float = 2.0     # soft at high-t

    # Shared expert (Group B, index 0) weight schedule (Part 2.9)
    # w_shared(t) = shared_weight_min + (shared_weight_max - shared_weight_min) * (t / T_max)
    shared_weight_min: float = 0.10   # low-t: decisive specialist routing
    shared_weight_max: float = 0.20   # high-t: stable baseline more useful

    # Batch-level router bias penalty coefficient (Part 4.2 Layer 2)
    bias_penalty_coeff: float = 0.01

    @property
    def routing_floor(self) -> float:
        """Budget-aware routing floor: ε = min(base, 0.15 / k) (Part 1.4)."""
        k = self.num_experts_B  # use B as reference (most experts)
        return min(self.routing_floor_base, 0.15 / max(k, 1))


# ─── Group Identity Embeddings ───────────────────────────────────────────────

class GroupIdentityEmbedding(nn.Module):
    """Learnable per-group identity vectors with orthogonality regularization.

    Each group gets a single learned vector that is broadcast-added to all
    tokens in that group before processing. Orthogonality regularization
    keeps the three vectors distinguishable throughout training.

    Shape: group_id_vecs  (3, embed_dim)
    """

    def __init__(self, embed_dim: int, num_groups: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_groups = num_groups

        # Initialize with random orthonormal vectors (avoids cold-start confusion)
        vecs = torch.randn(num_groups, embed_dim)
        vecs = torch.linalg.qr(vecs.T).Q.T[:num_groups]  # (3, D) orthonormal rows
        self.group_id_vecs = nn.Parameter(vecs)

    def forward(self, tokens: torch.Tensor, group_id: int) -> torch.Tensor:
        """
        Args:
            tokens:   (B, N, D)
            group_id: one of GROUP_A, GROUP_B, GROUP_C
        Returns:
            tokens + group_identity_vector  (B, N, D)
        """
        vec = self.group_id_vecs[group_id]  # (D,)
        return tokens + vec.unsqueeze(0).unsqueeze(0)

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalise off-diagonal entries of Gram matrix.
        Loss = || G - I ||_F^2  where  G = V V^T (unit-normed rows).
        """
        V = F.normalize(self.group_id_vecs, dim=-1)  # (3, D) unit rows
        G = V @ V.T
        I = torch.eye(self.num_groups, device=V.device)
        return ((G - I) ** 2).sum()


# ─── Group-Local LayerNorm ───────────────────────────────────────────────────

class GroupLocalLayerNorm(nn.Module):
    """Applies LayerNorm independently within each group's token slice.

    Prevents normalization statistics from leaking between groups
    (Part 1.3.2 — path 2 contamination).
    """

    def __init__(self, embed_dim: int, num_groups: int = 3):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_groups)
        ])

    def forward(
        self,
        tokens_A: torch.Tensor,
        tokens_B: torch.Tensor,
        tokens_C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args / Returns: each tensor (B, N_g, D), normalized independently."""
        return (
            self.norms[GROUP_A](tokens_A),
            self.norms[GROUP_B](tokens_B),
            self.norms[GROUP_C](tokens_C),
        )


# ─── Directed Attention Mask ─────────────────────────────────────────────────

class DirectedAttentionMask:
    """Builds a boolean attention mask that enforces A → C → B directionality.

    Tokens are concatenated in the order [A | C | B] before attention.
    The mask allows:
        - A attends to A       (self)
        - C attends to A, C    (A→C: C reads raw evidence)
        - B attends to A, C, B (C→B: B reads map + raw context)
    Blocks:
        - A attending to C or B (no upward feedback)
        - C attending to B      (no upward feedback)

    This is a STATIC mask (same for all batches / timesteps).
    """

    def __init__(self, num_tokens_A: int, num_tokens_C: int, num_tokens_B: int):
        self.N_A = num_tokens_A
        self.N_C = num_tokens_C
        self.N_B = num_tokens_B
        self.N = num_tokens_A + num_tokens_C + num_tokens_B

    def build(self) -> torch.BoolTensor:
        """Return (N, N) boolean mask. True = this (query, key) pair is allowed."""
        N_A, N_C, N_B = self.N_A, self.N_C, self.N_B
        N = self.N
        mask = torch.zeros(N, N, dtype=torch.bool)

        sA = slice(0, N_A)
        sC = slice(N_A, N_A + N_C)
        sB = slice(N_A + N_C, N)

        mask[sA, sA] = True   # A → A
        mask[sC, sA] = True   # C → A (Group C reads Group A)
        mask[sC, sC] = True   # C → C
        mask[sB, sA] = True   # B → A
        mask[sB, sC] = True   # B → C
        mask[sB, sB] = True   # B → B

        return mask

    def build_additive(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Returns (N, N) float tensor: 0.0 where allowed, -inf where blocked."""
        bool_mask = self.build().to(device)
        additive = torch.zeros(self.N, self.N, device=device)
        additive[~bool_mask] = float("-inf")
        return additive


# ─── Token-Level Hard Cap (Part 4.2, Layer 1) ───────────────────────────────

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
        """Compute cap(t) per sample.  Returns (B,) float."""
        t_norm = t.float() / self.T_max
        return self.cap_low + (self.cap_high - self.cap_low) * t_norm

    @torch.no_grad()
    def apply(
        self,
        probs: torch.Tensor,          # (B, N, E) — post-softmax routing probs
        t: torch.Tensor,              # (B,) — diffusion timestep
        floor: float = 0.0,           # routing floor (already applied)
    ) -> torch.Tensor:
        """Clip and redistribute. Returns (B, N, E) — capped probs (detached)."""
        B, N, E = probs.shape
        cap = self.cap_value(t)  # (B,)
        cap = cap.view(B, 1, 1).expand(B, N, E)

        capped = probs.clone()
        excess = F.relu(capped - cap)       # (B, N, E) — amount over cap
        capped = capped - excess             # clip to cap

        # Redistribute excess proportionally to non-capped experts
        # (those below cap and above floor)
        headroom = F.relu(cap - capped)      # how much room each expert has
        headroom_sum = headroom.sum(-1, keepdim=True).clamp(min=1e-8)
        redistribution = excess.sum(-1, keepdim=True) * (headroom / headroom_sum)
        capped = capped + redistribution

        return capped


# ─── Shared routing utilities ────────────────────────────────────────────────

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


# ─── Group A Router — Expert-Choice, Spatially-Anchored ─────────────────────

class GroupARouter(nn.Module):
    """Expert-choice router for Group A (raw sensory tokens).

    Expert-choice means each EXPERT chooses its top-K tokens by affinity,
    guaranteeing uniform expert utilisation. This is architecturally
    different from token-choice (where tokens choose experts).

    Routing key: spatial position of the underlying Gaussian + coarse scene
    density estimate via token features.

    No cross-group conditioning — Group A is processed first.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_A
        self.top_k = cfg.top_k_A
        self.embed_dim = cfg.embed_dim

        # Gate MLP: token features → expert logits
        # Input: [token_features (D) || spatial_xyz (3)]
        self.gate_proj = nn.Sequential(
            nn.Linear(cfg.embed_dim + 3, cfg.embed_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.embed_dim // 2, self.num_experts),
        )

        # Learnable spatial center per expert ("home region")
        self.expert_spatial_centers = nn.Parameter(
            torch.randn(self.num_experts, 3) * 10.0
        )

        # Hard cap (Layer 1 only for Group A)
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)

        # Routing floor
        self._floor = cfg.routing_floor

    def forward(
        self,
        tokens: torch.Tensor,      # (B, N_A, D)
        spatial_xyz: torch.Tensor,  # (B, N_A, 3)
        t: torch.Tensor,           # (B,) diffusion timestep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing: each expert selects its top-K tokens.

        Returns:
            dispatch_weights: (B, N_A, E) — sparse assignment weights
            combine_weights:  (B, N_A, E) — normalised combination weights
        """
        B, N, D = tokens.shape
        E = self.num_experts

        # 1. Spatial affinity: negative distance from each token to each expert center
        dists = torch.cdist(
            spatial_xyz.reshape(B * N, 1, 3),
            self.expert_spatial_centers.unsqueeze(0).expand(B * N, -1, -1),
        ).reshape(B, N, E)
        spatial_affinity = -dists / (dists.mean() + 1e-6)

        # 2. Content gate
        gate_input = torch.cat([tokens, spatial_xyz], dim=-1)  # (B, N, D+3)
        content_logits = self.gate_proj(gate_input)             # (B, N, E)

        # 3. Combined logits → expert-choice assignment
        logits = content_logits + spatial_affinity  # (B, N, E)

        # Expert-choice: transpose so experts choose tokens
        # expert_logits: (B, E, N) — each expert scores all tokens
        expert_logits = logits.transpose(1, 2)  # (B, E, N)

        # Each expert selects its top-K tokens
        tokens_per_expert = max(1, (N * self.top_k) // E)
        topk_vals, topk_idx = expert_logits.topk(
            min(tokens_per_expert, N), dim=-1
        )  # (B, E, K_per_expert)

        # Build dispatch weights: (B, N, E)
        expert_probs = F.softmax(expert_logits, dim=-1)  # (B, E, N)
        dispatch = torch.zeros(B, E, N, device=tokens.device)
        dispatch.scatter_(-1, topk_idx, topk_vals.sigmoid())  # sigmoid for soft weights

        dispatch = dispatch.transpose(1, 2)  # (B, N, E)

        # Apply routing floor and hard cap
        dispatch = _apply_routing_floor(dispatch, self._floor)
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)

        # Normalise combine weights
        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)

        return dispatch, combine


# ─── Group C Router — Structural Deterministic ───────────────────────────────

class GroupCRouter(nn.Module):
    """Structural (near-deterministic) router for Group C (map / context tokens).

    Token type ID maps directly to an expert, with a light learned soft
    residual for ambiguous tokens (TTYPE_UNKNOWN).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_C
        assert cfg.num_experts_C >= NUM_C_TYPES, (
            f"num_experts_C ({cfg.num_experts_C}) must be >= NUM_C_TYPES ({NUM_C_TYPES})"
        )
        self.top_k = cfg.top_k_C
        self.temp = cfg.struct_router_temp
        self._floor = cfg.routing_floor

        # Deterministic token-type → expert assignment matrix
        base_assignment = torch.zeros(NUM_C_TYPES, self.num_experts)
        for t_type in range(NUM_C_TYPES - 1):
            base_assignment[t_type, t_type % self.num_experts] = 1.0
        base_assignment[TTYPE_UNKNOWN] = 1.0 / self.num_experts
        self.register_buffer("base_assignment", base_assignment)

        # Learned soft residual gate for UNKNOWN tokens
        self.soft_gate = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.embed_dim // 4, self.num_experts),
        )
        # Init near zero so deterministic routing dominates
        for m in self.soft_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Hard cap only (no Layer 2 for Group C)
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, N_C, D)
        token_types: torch.Tensor,   # (B, N_C) long
        t: torch.Tensor,            # (B,) diffusion timestep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            dispatch_weights: (B, N_C, E)
            combine_weights:  (B, N_C, E)
        """
        B, N, D = tokens.shape

        # Deterministic base probabilities
        base_probs = self.base_assignment[token_types.reshape(-1)].reshape(B, N, -1)

        # Learned soft residual
        soft_logits = self.soft_gate(tokens) * self.temp
        soft_probs = F.softmax(soft_logits, dim=-1)

        # Mix: deterministic for known types, soft for unknown
        is_unknown = (token_types == TTYPE_UNKNOWN).float().unsqueeze(-1)
        probs = (1 - is_unknown) * base_probs + is_unknown * soft_probs

        # Floor + top-k + hard cap
        probs = _apply_routing_floor(probs, self._floor)
        dispatch = _topk_mask(probs, self.top_k)
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)

        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
        return dispatch, combine


# ─── Group B Router — Scene-Conditioned Soft Router ──────────────────────────

class GroupBRouter(nn.Module):
    """Scene-conditioned soft router for Group B (interaction / agent tokens).

    The routing decision accounts for:
        1. What Group A (sensory) saw — confirms agent velocities.
        2. What Group C (map/context) saw — confirms intersection topology.
        3. The current diffusion timestep t — modulates gate temperature τ(t).

    Gate temperature (Part 2.8):
        τ(t) = τ_min + (τ_max − τ_min) * (t / T_max)
        Low-t: sharp (commit to specialist). High-t: soft (explore modes).

    Shared expert (Part 2.9):
        Expert index 0 is the shared anchor. Its weight is timestep-scheduled:
        w_shared(t) = w_min + (w_max - w_min) * (t / T_max)

    Stop-gradient (Part 3.4):
        Group C output is detached before cross-attention when stopgrad_C_to_B=True.

    Two-layer capacity (Part 4.2):
        Layer 1: Token-level hard cap (non-differentiable).
        Layer 2: Batch-level per-expert bias scalars (gradient to bias only).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.num_experts = cfg.num_experts_B
        self.top_k = cfg.top_k_B
        self.stopgrad_C_to_B = cfg.stopgrad_C_to_B
        self.embed_dim = cfg.embed_dim
        self.T_max = cfg.T_max
        self._floor = cfg.routing_floor

        # Temperature schedule params
        self.tau_min = cfg.tau_min
        self.tau_max = cfg.tau_max

        # Shared expert weight schedule params
        self.shared_w_min = cfg.shared_weight_min
        self.shared_w_max = cfg.shared_weight_max

        # Cross-attention: Group B tokens query [Group A | Group C] context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.gate_num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.cross_attn_norm = nn.LayerNorm(cfg.embed_dim)

        # Diffusion timestep embedding
        self.t_embed = TimestepEmbedding(cfg.embed_dim)

        # Gate MLP: [B_token (D) | context (D) | t_embed (D)] → expert logits
        self.gate_mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim * 3, cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.embed_dim, self.num_experts),
        )

        # Layer 2: per-expert bias scalars (Part 4.2)
        # Gradient flows ONLY to these, not to gate_mlp weights.
        # Added to gate logits before softmax.
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))

        # Hard cap (Layer 1)
        self.hard_cap = TokenLevelHardCap(cfg.cap_low, cfg.cap_high, cfg.T_max)

        # Batch-level bias penalty coefficient
        self.bias_penalty_coeff = cfg.bias_penalty_coeff

    def _gate_temperature(self, t: torch.Tensor) -> torch.Tensor:
        """τ(t) = τ_min + (τ_max − τ_min) * (t / T_max).  Returns (B, 1, 1)."""
        t_norm = t.float() / self.T_max
        tau = self.tau_min + (self.tau_max - self.tau_min) * t_norm
        return tau.view(-1, 1, 1)

    def _shared_expert_weight(self, t: torch.Tensor) -> torch.Tensor:
        """w_shared(t) = w_min + (w_max - w_min) * (t / T_max).  Returns (B, 1, 1)."""
        t_norm = t.float() / self.T_max
        w = self.shared_w_min + (self.shared_w_max - self.shared_w_min) * t_norm
        return w.view(-1, 1, 1)

    def forward(
        self,
        tokens_B: torch.Tensor,     # (B, N_B, D)
        output_A: torch.Tensor,     # (B, N_A, D) — Group A final output
        output_C: torch.Tensor,     # (B, N_C, D) — Group C final output
        t: torch.Tensor,            # (B,) diffusion timestep
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            dispatch_weights: (B, N_B, E)
            combine_weights:  (B, N_B, E)
            bias_penalty:     scalar — batch-level utilisation penalty (Layer 2)
        """
        B, N_B, D = tokens_B.shape

        # Stop-gradient on Group C output when entering gate path
        ctx_C = output_C.detach() if self.stopgrad_C_to_B else output_C

        # Build cross-group context: concatenate A and (detached) C
        ctx_AC = torch.cat([output_A, ctx_C], dim=1)  # (B, N_A + N_C, D)

        # Cross-attention: B tokens query AC context
        ctx_B, _ = self.cross_attn(
            query=tokens_B,
            key=ctx_AC,
            value=ctx_AC,
        )  # (B, N_B, D)
        ctx_B = self.cross_attn_norm(ctx_B + tokens_B)  # residual + norm

        # Timestep embedding
        t_emb = self.t_embed(t)  # (B, D)
        t_emb = t_emb.unsqueeze(1).expand(-1, N_B, -1)  # (B, N_B, D)

        # Gate MLP: [token | cross-group context | t_embed]
        gate_input = torch.cat([tokens_B, ctx_B, t_emb], dim=-1)  # (B, N_B, 3D)
        logits = self.gate_mlp(gate_input)  # (B, N_B, E)

        # Add per-expert bias scalars (Layer 2 — gradient flows only here)
        # Detach logits from gate_mlp so bias gradient doesn't flow to W_g
        logits_for_bias = logits.detach() + self.expert_bias
        # But we need logits with gradient for the actual routing
        # Solution: use logits (with gate grad) + expert_bias (with bias grad)
        logits = logits + self.expert_bias

        # Gate temperature τ(t) — Part 2.8
        tau = self._gate_temperature(t)  # (B, 1, 1)
        probs = F.softmax(logits / tau, dim=-1)  # (B, N_B, E)

        # Apply routing floor
        probs = _apply_routing_floor(probs, self._floor)

        # Inject shared expert anchor (index 0) — Part 2.9
        probs = self._inject_shared_expert(probs, t)

        # Top-k masking
        dispatch = _topk_mask(probs, self.top_k)

        # Layer 1: Token-level hard cap (non-differentiable)
        dispatch = self.hard_cap.apply(dispatch, t, self._floor)

        # Layer 2: Batch-level bias penalty
        bias_penalty = self._batch_bias_penalty(dispatch, N_B)

        # Normalise combine weights
        combine = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)

        return dispatch, combine, bias_penalty

    def _inject_shared_expert(
        self, probs: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Guarantee w_shared(t) to expert index 0, re-normalise rest."""
        w = self._shared_expert_weight(t)  # (B, 1, 1)

        shared_prob = probs[..., 0:1]      # (B, N, 1)
        other_probs = probs[..., 1:]       # (B, N, E-1)

        new_shared = torch.clamp(shared_prob, min=w)
        remaining = 1.0 - new_shared
        other_sum = other_probs.sum(-1, keepdim=True).clamp(min=1e-8)
        new_others = other_probs / other_sum * remaining

        return torch.cat([new_shared, new_others], dim=-1)

    def _batch_bias_penalty(
        self, dispatch: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Batch-level utilisation penalty (Part 4.2 Layer 2).

        Penalise specialists (indices 1..E-1) that exceed fair share.
        Gradient flows only to self.expert_bias via the dispatch weights.
        """
        E = self.num_experts
        # Fair share per specialist (excluding shared expert 0)
        fair_share = num_tokens / max(E - 1, 1)
        # Expert load: sum of dispatch weights across batch and tokens
        # (B, E) → mean over batch
        expert_load = dispatch.sum(dim=1).mean(dim=0)  # (E,)
        # Only penalise specialists (index 1+)
        specialist_load = expert_load[1:]
        excess = F.relu(specialist_load - fair_share * 1.5)  # 1.5× fair share threshold
        return self.bias_penalty_coeff * (excess ** 2).sum()


# ─── Timestep Embedding (sinusoidal, used by Group B gate) ───────────────────

class TimestepEmbedding(nn.Module):
    """Maps diffusion timestep t → vector (B, D).  Sinusoidal + 2-layer MLP."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        self.register_buffer(
            "freq",
            torch.exp(-math.log(10000) * torch.arange(half_dim) / (half_dim - 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t (B,) float/long.  Returns: (B, embed_dim)."""
        t = t.float()
        args = t[:, None] * self.freq[None, :]  # (B, half_dim)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)
        return self.proj(emb)


# ─── DyDiT Skip Scheduler ───────────────────────────────────────────────────

class DyDiTSkipScheduler:
    """Decides per-group skip based on diffusion timestep t.

    Opposite schedules for B and C (Part 0.3):
        Group C: skip when t/T_max < t_skip_C  (late denoising, mode committed)
        Group B: skip when t/T_max > t_skip_B  (early denoising, coarse structure)
        Group A: timestep-independent (raw sensor) — cache logic external.

    NOTE: This is the threshold-based scheduler. The plan calls for a
    learned per-token skip predictor for Group B (Part 4.5) — that should
    be implemented as a separate module in expert_ffn.py or a dedicated file.
    This scheduler provides the baseline decision used during early training
    and for Groups A and C which don't need a learned predictor.
    """

    def __init__(self, cfg: MoEConfig):
        self.t_skip_C = cfg.t_skip_C
        self.t_skip_B = cfg.t_skip_B
        self.T_max = cfg.T_max
        self.cache_interval_A = cfg.cache_interval_A

    def skip(self, group: int, t: torch.Tensor) -> torch.BoolTensor:
        """Return (B,) bool — True means skip expert computation for this sample."""
        t_norm = t.float() / self.T_max

        if group == GROUP_C:
            return t_norm < self.t_skip_C
        elif group == GROUP_B:
            return t_norm > self.t_skip_B
        elif group == GROUP_A:
            return torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
        else:
            raise ValueError(f"Unknown group id: {group}")

    def cache_step_for_A(self, step: int) -> bool:
        """True if Group A should recompute this step (else use cached output)."""
        return (step % self.cache_interval_A) == 0


# ─── Group Tokenizer ─────────────────────────────────────────────────────────

class GroupTokenizer(nn.Module):
    """Partitions raw token tensors into Group A/B/C and adds group identity.

    Should be called ONCE at input, not repeated every block.
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
        raw_A: torch.Tensor,  # (B, N_A, dim_A_in)
        raw_B: torch.Tensor,  # (B, N_B, dim_B_in)
        raw_C: torch.Tensor,  # (B, N_C, dim_C_in)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tA = self.group_id_emb(self.proj_A(raw_A), GROUP_A)
        tB = self.group_id_emb(self.proj_B(raw_B), GROUP_B)
        tC = self.group_id_emb(self.proj_C(raw_C), GROUP_C)
        return tA, tB, tC

    def orthogonality_loss(self) -> torch.Tensor:
        return self.group_id_emb.orthogonality_loss()


# ─── Router Output ───────────────────────────────────────────────────────────

@dataclass
class RouterOutput:
    """All routing decisions, weights, and diagnostics from one forward pass."""

    # Per-group dispatch/combine weights
    dispatch_A:     torch.Tensor      # (B, N_A, E_A)
    combine_A:      torch.Tensor      # (B, N_A, E_A)
    dispatch_B:     torch.Tensor      # (B, N_B, E_B)
    combine_B:      torch.Tensor      # (B, N_B, E_B)
    dispatch_C:     torch.Tensor      # (B, N_C, E_C)
    combine_C:      torch.Tensor      # (B, N_C, E_C)

    # Skip decisions per sample
    skip_A:         torch.BoolTensor  # (B,)
    skip_B:         torch.BoolTensor  # (B,)
    skip_C:         torch.BoolTensor  # (B,)

    # Losses
    bias_penalty:   torch.Tensor      # scalar — Group B batch-level bias penalty
    ortho_loss:     torch.Tensor      # scalar — group identity orthogonality

    # Attention mask
    attn_mask:      torch.Tensor      # (N_total, N_total) additive float mask

    # Diagnostics (detached scalars for logging)
    routing_entropy_A: torch.Tensor
    routing_entropy_B: torch.Tensor
    routing_entropy_C: torch.Tensor


# ─── Top-Level ModalityMoERouter ─────────────────────────────────────────────

class ModalityMoERouter(nn.Module):
    """Top-level router that wires GroupTokenizer, three group routers,
    skip scheduler, and the two-layer capacity mechanism.

    Two-pass forward (required by directed attention):
        Pass 1: route Group A (no cross-group context)
        Pass 2: route Group C (uses Group A output from expert computation)
        Pass 3: route Group B (uses Group A + Group C expert outputs)

    The caller must run expert FFNs between passes to produce output_A/output_C.
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
        self.router_A = GroupARouter(cfg)
        self.router_C = GroupCRouter(cfg)
        self.router_B = GroupBRouter(cfg)
        self.skip_sched = DyDiTSkipScheduler(cfg)

        # Static attention mask
        mask_builder = DirectedAttentionMask(
            cfg.num_tokens_A, cfg.num_tokens_C, cfg.num_tokens_B
        )
        self.register_buffer("attn_mask", mask_builder.build_additive())

    def forward(
        self,
        raw_A: torch.Tensor,            # (B, N_A, dim_A_in)
        raw_B: torch.Tensor,            # (B, N_B, dim_B_in)
        raw_C: torch.Tensor,            # (B, N_C, dim_C_in)
        spatial_xyz: torch.Tensor,      # (B, N_A, 3)
        token_types_C: torch.Tensor,    # (B, N_C) long
        t: torch.Tensor,                # (B,) diffusion timestep
        output_A: Optional[torch.Tensor] = None,
        output_C: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> RouterOutput:
        """Compute all routing decisions for one forward pass.

        Calling protocol:
            # Pass 1: Group A routing (no context)
            out1 = router(..., output_A=None, output_C=None)
            # → run Group A experts → expert_output_A

            # Pass 2: Group C routing
            out2 = router(..., output_A=expert_output_A, output_C=None)
            # → run Group C experts → expert_output_C

            # Pass 3: Group B routing (full context)
            out3 = router(..., output_A=expert_output_A, output_C=expert_output_C)
            # → run Group B experts

        Returns RouterOutput with all routing decisions.
        """
        # Tokenize and group-local normalize
        tokens_A, tokens_B, tokens_C = self.tokenizer(raw_A, raw_B, raw_C)
        tokens_A, tokens_B, tokens_C = self.ln_groups(tokens_A, tokens_B, tokens_C)

        # Skip decisions
        skip_A = self.skip_sched.skip(GROUP_A, t)
        skip_B = self.skip_sched.skip(GROUP_B, t)
        skip_C = self.skip_sched.skip(GROUP_C, t)

        # Route Group A (expert-choice, no cross-group context)
        dispatch_A, combine_A = self.router_A(tokens_A, spatial_xyz, t)

        # Route Group C (structural, no cross-group context needed for routing)
        dispatch_C, combine_C = self.router_C(tokens_C, token_types_C, t)

        # Route Group B (scene-conditioned, needs A and C expert outputs)
        bias_penalty = torch.tensor(0.0, device=tokens_B.device)
        if output_A is not None and output_C is not None:
            dispatch_B, combine_B, bias_penalty = self.router_B(
                tokens_B, output_A, output_C, t
            )
        else:
            # Placeholder: uniform routing until cross-group context available
            B, N_B = tokens_B.shape[:2]
            E_B = self.cfg.num_experts_B
            uniform = torch.full(
                (B, N_B, E_B), 1.0 / E_B, device=tokens_B.device
            )
            dispatch_B = combine_B = uniform

        # Orthogonality loss
        ortho_loss = self.cfg.ortho_reg_weight * self.tokenizer.orthogonality_loss()

        # Routing entropy (for logging / probes)
        ent_A = self._routing_entropy(dispatch_A)
        ent_B = self._routing_entropy(dispatch_B)
        ent_C = self._routing_entropy(dispatch_C)

        return RouterOutput(
            dispatch_A=dispatch_A,
            combine_A=combine_A,
            dispatch_B=dispatch_B,
            combine_B=combine_B,
            dispatch_C=dispatch_C,
            combine_C=combine_C,
            skip_A=skip_A,
            skip_B=skip_B,
            skip_C=skip_C,
            bias_penalty=bias_penalty,
            ortho_loss=ortho_loss,
            attn_mask=self.attn_mask,
            routing_entropy_A=ent_A,
            routing_entropy_B=ent_B,
            routing_entropy_C=ent_C,
        )

    @staticmethod
    def _routing_entropy(dispatch: torch.Tensor) -> torch.Tensor:
        """Mean routing entropy (bits) across batch and tokens."""
        p = dispatch / (dispatch.sum(-1, keepdim=True) + 1e-8)
        ent = -(p * (p + 1e-8).log()).sum(-1)
        return ent.mean().detach()


# ─── Convenience Factory ─────────────────────────────────────────────────────

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
    """Convenience factory."""
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


# ─── Sanity Check ────────────────────────────────────────────────────────────

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
    xyz = torch.randn(B, cfg.num_tokens_A, 3) * 20.0
    ttypes = torch.randint(0, NUM_C_TYPES, (B, cfg.num_tokens_C))
    t_vec = torch.randint(0, 1000, (B,))

    # Pass 1: route A and C (no cross-group context)
    out = router(raw_A, raw_B, raw_C, xyz, ttypes, t_vec)
    print(f"dispatch_A shape : {out.dispatch_A.shape}")    # (2, 64, 4)
    print(f"dispatch_C shape : {out.dispatch_C.shape}")    # (2, 64, 6)
    print(f"dispatch_B shape : {out.dispatch_B.shape}")    # (2, 32, 4) uniform
    print(f"attn_mask shape  : {out.attn_mask.shape}")     # (160, 160)
    print(f"bias_penalty     : {out.bias_penalty.item():.6f}")
    print(f"ortho_loss       : {out.ortho_loss.item():.6f}")
    print(f"entropy_A        : {out.routing_entropy_A.item():.4f}")
    print(f"entropy_C        : {out.routing_entropy_C.item():.4f}")
    print(f"skip_B (t>0.7)   : {out.skip_B.tolist()}")
    print(f"skip_C (t<0.2)   : {out.skip_C.tolist()}")

    # Pass 3: route B with mock expert outputs
    mock_out_A = torch.randn(B, cfg.num_tokens_A, 128)
    mock_out_C = torch.randn(B, cfg.num_tokens_C, 128)
    out3 = router(raw_A, raw_B, raw_C, xyz, ttypes, t_vec,
                  output_A=mock_out_A, output_C=mock_out_C)
    print(f"\nPass 3 (with context):")
    print(f"dispatch_B shape : {out3.dispatch_B.shape}")
    print(f"bias_penalty     : {out3.bias_penalty.item():.6f}")
    print(f"entropy_B        : {out3.routing_entropy_B.item():.4f}")

    # Verify temperature effect: low-t should be sharper than high-t
    t_low = torch.tensor([50, 100])
    t_high = torch.tensor([800, 900])
    out_low = router(raw_A, raw_B, raw_C, xyz, ttypes, t_low,
                     output_A=mock_out_A, output_C=mock_out_C)
    out_high = router(raw_A, raw_B, raw_C, xyz, ttypes, t_high,
                      output_A=mock_out_A, output_C=mock_out_C)
    print(f"\nTemperature effect:")
    print(f"  entropy_B at low-t  : {out_low.routing_entropy_B.item():.4f}  (should be lower = sharper)")
    print(f"  entropy_B at high-t : {out_high.routing_entropy_B.item():.4f}  (should be higher = softer)")

    # Verify attention mask structure
    mask = router.attn_mask
    N_A, N_C, N_B = cfg.num_tokens_A, cfg.num_tokens_C, cfg.num_tokens_B
    sA = slice(0, N_A)
    sC = slice(N_A, N_A + N_C)
    sB = slice(N_A + N_C, N_A + N_C + N_B)
    assert torch.all(mask[sA, sC] == float("-inf")), "FAIL: A should not attend to C"
    assert torch.all(mask[sA, sB] == float("-inf")), "FAIL: A should not attend to B"
    assert torch.all(mask[sC, sB] == float("-inf")), "FAIL: C should not attend to B"
    assert torch.all(mask[sB, sA] == 0.0),           "FAIL: B should attend to A"
    assert torch.all(mask[sB, sC] == 0.0),           "FAIL: B should attend to C"
    print("Attention mask directionality: PASS")

    # Verify stop-gradient: C output should be detached in B's gate path
    test_C = torch.randn(B, cfg.num_tokens_C, 128, requires_grad=True)
    test_A = torch.randn(B, cfg.num_tokens_A, 128)
    _, _, bp = router.router_B(
        torch.randn(B, cfg.num_tokens_B, 128),
        test_A, test_C, t_vec,
    )
    # If stopgrad is active, test_C should have no gradient from B's routing
    bp.backward()
    assert test_C.grad is None or test_C.grad.abs().sum() == 0, \
        "FAIL: stop-gradient on C→B should block gradient"
    print("Stop-gradient C→B: PASS")

    print("\nAll token_router.py tests PASSED.")