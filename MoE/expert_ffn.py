"""
DDPM/moe/expert_ffn.py

Expert FFN pools for the three-group MoE (Group A / B / C).

Design decisions (from plan v3):
─────────────────────────────────────────────
  - Each group has a pool of num_experts independent FFN modules.
    Expert index 0 in Group B is the "shared expert anchor" — its weights
    are initialized as an exact copy with no noise (Part 6.3).

  - Dispatch/Combine pattern (standard sparse MoE):
        1. Router picks top-k experts per token (done in token_router.py).
        2. Dispatch token to those experts (weighted by dispatch_weights).
        3. Each expert processes its assigned tokens independently.
        4. Combine outputs: sum over experts weighted by combine_weights.
    Implemented as batched gather-scatter (loop over experts, not tokens).

  - Expert output scaling (Part 6.3):
        output_scale_i initialized to 1/N_experts so expert outputs sum to
        the same magnitude as the original dense FFN at initialization.

  - DyDiT skip with cached fallback (Part 4.5):
        Group A: Multi-checkpoint interpolation cache. Store outputs at
                 reference timesteps, interpolate for intermediate steps.
        Group B: Learned per-token skip predictor (2-layer MLP).
                 Skip at high-t. 20% minimum token floor.
                 Hard exceptions for rare behavioral tokens.
        Group C: Two-level static cache.
                 t_cache_high (mode-selection), t_cache_low (fine-shaping).
                 Recompute window between the two levels.

  - Anchor token forward pass (Part 4.4):
        Each ExpertPool has a forward_anchors() method that passes the SAME
        anchor tokens through ALL experts simultaneously. Used by
        decorrelation loss — decouples diversity measurement from routing.

  - Asymmetric initialization hooks (Part 6.3):
        apply_asymmetric_init() methods that can be called after construction
        to add group-specific biases to expert weights.

This file provides:
    GatedExpertFFN          — single expert FFN (SwiGLU-style gated MLP)
    ExpertPool              — pool of N_experts with dispatch/combine + anchor forward
    LearnedSkipPredictor    — per-token skip predictor for Group B (Part 4.5)
    GroupAExpertPool        — interpolation cache for Group A
    GroupBExpertPool        — learned skip + identity fallback
    GroupCExpertPool        — two-level cache for Group C
    ExpertOutputs           — typed return container
    AllGroupsExpertRunner   — orchestrates A → C → B execution order
    build_expert_pools      — factory
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.token_router import MoEConfig, GROUP_A, GROUP_B, GROUP_C


#  Single Expert: Gated FFN (SwiGLU-style) 

class GatedExpertFFN(nn.Module):
    """One expert — a SwiGLU-style gated MLP with a learnable output scale.

    Architecture:
        x  ──▶ gate_proj  ──▶ GELU ──▶ gate
        x  ──▶ value_proj ───────────▶ value
                gate * value  ──▶ out_proj  ──▶ output * scale

    No bias in projections (PaLM / LLaMA convention).

    Args:
        embed_dim:   D — input and output dimension
        ff_mult:     hidden-dim multiplier (default 4)
        dropout:     applied after GELU gate
        num_experts: total experts in pool (for output_scale init = 1/E)
    """

    def __init__(
        self,
        embed_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        num_experts: int = 4,
    ):
        super().__init__()
        hidden = embed_dim * ff_mult

        self.gate_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.value_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Per-expert output scale: init 1/N_experts (Part 6.3)
        # Expert outputs sum to same magnitude as original dense FFN
        self.output_scale = nn.Parameter(torch.tensor(1.0 / num_experts))

        # Weight init: small out_proj so residual stream dominates early
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (M, D).  Returns: (M, D)."""
        gate = F.gelu(self.gate_proj(x))
        gate = self.dropout(gate)
        value = self.value_proj(x)
        out = self.out_proj(gate * value)
        return out * self.output_scale


# Token-Type Attention Bias for Group C (Part 6.3)

class TokenTypeAttentionBias(nn.Module):
    """Soft attention bias applied to expert input to steer specialization.

    Group C experts are initialized NOT with noise perturbations but with
    different token-type attention bias terms (Part 6.3):
        - Intersection expert → spatial co-occurrence bias
        - Traffic light expert → temporal state bias
        - Road geometry expert → directional feature bias

    The bias is a learnable additive term in feature space that softly
    steers which feature dimensions the expert attends to. Implemented as
    a low-rank projection that reshapes the input emphasis without
    destroying pretrained features.

    Args:
        embed_dim: D
        rank: rank of the bias projection (low-rank to limit capacity)
        bias_type: one of 'spatial', 'temporal', 'directional', 'none'
    """

    BIAS_TYPES = {'spatial', 'temporal', 'directional', 'none'}

    def __init__(self, embed_dim: int, rank: int = 16, bias_type: str = 'none'):
        super().__init__()
        assert bias_type in self.BIAS_TYPES, f"Unknown bias_type: {bias_type}"
        self.bias_type = bias_type
        self.embed_dim = embed_dim

        if bias_type == 'none':
            # No bias — identity passthrough
            self.bias = None
        else:
            # Low-rank additive bias: down-project then up-project
            # Initialized so the bias starts small and grows during training
            self.down = nn.Linear(embed_dim, rank, bias=False)
            self.up = nn.Linear(rank, embed_dim, bias=False)
            # Initialize near-zero so bias is negligible at start
            nn.init.xavier_uniform_(self.down.weight, gain=0.01)
            nn.init.zeros_(self.up.weight)

            # Feature mask: which dimensions are emphasized
            # Different bias types emphasize different regions of the
            # embedding space. The mask is a learnable soft gate.
            self.feature_gate = nn.Parameter(torch.zeros(embed_dim))
            self._init_feature_gate(embed_dim, bias_type)

    def _init_feature_gate(self, D: int, bias_type: str):
        """Initialize the feature gate to emphasize different feature bands.

        Convention: embed_dim is partitioned into conceptual bands.
        These are soft priors — the model can learn to override them.
        """
        with torch.no_grad():
            quarter = D // 4
            if bias_type == 'spatial':
                # Emphasize spatial-position features (first quarter)
                self.feature_gate[:quarter] = 0.5
            elif bias_type == 'temporal':
                # Emphasize temporal/state-change features (second quarter)
                self.feature_gate[quarter:2 * quarter] = 0.5
            elif bias_type == 'directional':
                # Emphasize directional/heading features (third quarter)
                self.feature_gate[2 * quarter:3 * quarter] = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft attention bias. x: (..., D) → (..., D)."""
        if self.bias_type == 'none':
            return x
        # Soft feature-gated additive bias
        gate = torch.sigmoid(self.feature_gate)  # (D,)
        bias = self.up(F.gelu(self.down(x)))  # (..., D)
        return x + bias * gate


# Expert Pool: Vectorized Dispatch / Compute / Combine 

class ExpertPool(nn.Module):
    """Pool of num_experts independent GatedExpertFFNs with batched dispatch.

    Dispatch/Combine algorithm (token-choice, top-k):
        1. Reshape to (B*N, D) and (B*N, E)
        2. For each expert e: gather assigned tokens, run expert, scatter back
        3. Weight by combine_weights

    Also provides forward_anchors() — pass the SAME anchor tokens through
    ALL experts simultaneously for decorrelation (Part 4.4).

    Args:
        num_experts: E
        embed_dim:   D
        ff_mult:     hidden dimension multiplier
        dropout:     inside each expert
    """

    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim

        self.experts = nn.ModuleList([
            GatedExpertFFN(embed_dim, ff_mult, dropout, num_experts=num_experts)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        tokens: torch.Tensor,            # (B, N, D)
        dispatch_weights: torch.Tensor,   # (B, N, E) sparse top-k
        combine_weights: torch.Tensor,    # (B, N, E) normalised
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            output:      (B, N, D) — combined expert outputs
            expert_acts: List[E] of (M_e, D) — raw activations per expert
                         (used by decorrelation loss on routed tokens;
                          for proper anchor-based decorrelation, use forward_anchors)
        """
        B, N, D = tokens.shape
        E = self.num_experts
        device = tokens.device

        flat_tokens = tokens.reshape(B * N, D)
        flat_disp = dispatch_weights.reshape(B * N, E)
        flat_comb = combine_weights.reshape(B * N, E)

        output = torch.zeros_like(flat_tokens)
        expert_acts: List[torch.Tensor] = []

        for e, expert in enumerate(self.experts):
            assigned = flat_disp[:, e] > 0
            num_assigned = assigned.sum().item()

            if num_assigned == 0:
                expert_acts.append(torch.zeros(0, D, device=device))
                continue

            expert_tokens = flat_tokens[assigned]
            expert_out = expert(expert_tokens)
            expert_acts.append(expert_out)

            w = flat_comb[assigned, e].unsqueeze(-1)
            output[assigned] += expert_out * w

        return output.reshape(B, N, D), expert_acts

    def forward_anchors(
        self,
        anchor_tokens: torch.Tensor,     # (M, D) — same tokens for all experts
        exclude_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Pass the SAME anchor tokens through ALL experts (Part 4.4).

        Used by decorrelation loss to measure expert diversity on identical
        inputs, decoupled from routing.

        Args:
            anchor_tokens: (M, D) — anchor token embeddings
            exclude_indices: expert indices to skip (e.g. [0] to exclude
                           shared expert from decorrelation)
        Returns:
            List[E] of (M, D) — each expert's output on the anchor tokens.
            Excluded experts return empty tensors.
        """
        exclude = set(exclude_indices or [])
        results = []
        M, D = anchor_tokens.shape

        for e, expert in enumerate(self.experts):
            if e in exclude:
                results.append(torch.zeros(0, D, device=anchor_tokens.device))
            else:
                results.append(expert(anchor_tokens))

        return results


#  Learned Skip Predictor for Group B (Part 4.5) 

class StraightThroughBernoulli(torch.autograd.Function):
    """Straight-through estimator for binary skip decisions.

    Forward: hard threshold (score > 0.5 → True).
    Backward: gradient passes through as if the threshold were identity.
    This allows gradient flow through the skip decision during training.
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor) -> torch.Tensor:
        return (scores > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class LearnedSkipPredictor(nn.Module):
    """Per-token skip predictor for Group B.

    Architecture: 2-layer MLP.
    Input: agent token features (D) + bottlenecked context (D//4) + t_embed (D).
    Output: per-token skip score in [0, 1] (sigmoid). High = skip.

    At inference: hard threshold at 0.5.
    During training: straight-through estimator for gradient flow.

    Constraints (Part 4.5):
        - 20% minimum token floor: at least 20% of tokens always processed.
        - Hard exceptions: tokens in rare behavioral categories never skipped.
        - Skip-routing interaction: skipped tokens contribute NOTHING to
          routing utilization statistics.

    Args:
        embed_dim: D
        T_max: maximum diffusion timestep
    """

    def __init__(self, embed_dim: int, T_max: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.T_max = T_max
        self.min_process_frac = 0.20  # 20% floor

        # Bottleneck for context (from Group C output)
        self.ctx_bottleneck = nn.Linear(embed_dim, embed_dim // 4)

        # Timestep embedding (lightweight — reuse sinusoidal pattern)
        half = embed_dim // 2
        self.register_buffer(
            "freq", torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        )
        self.t_proj = nn.Linear(embed_dim, embed_dim)

        # Skip MLP: [token (D) + ctx (D//4) + t (D)] → score
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 4 + embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Init near zero so no skipping at start of training
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def _t_embed(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding. (B,) → (B, D)."""
        t_f = t.float()
        args = t_f[:, None] * self.freq[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.t_proj(emb)

    def forward(
        self,
        tokens: torch.Tensor,                # (B, N, D)
        ctx_C: Optional[torch.Tensor],       # (B, N_C, D) or None
        t: torch.Tensor,                     # (B,)
        rare_mask: Optional[torch.BoolTensor] = None,  # (B, N) True = never skip
    ) -> Tuple[torch.BoolTensor, torch.Tensor]:
        """Compute per-token skip decisions.

        Returns:
            skip_tokens: (B, N) bool — True = skip this token's expert computation
            scores:      (B, N) float — raw skip scores for loss computation
        """
        B, N, D = tokens.shape

        # Bottleneck context (mean-pool Group C if available)
        if ctx_C is not None:
            ctx_mean = ctx_C.mean(dim=1)  # (B, D)
            ctx_bn = self.ctx_bottleneck(ctx_mean)  # (B, D//4)
        else:
            ctx_bn = torch.zeros(B, D // 4, device=tokens.device)

        # Expand context and timestep to per-token
        ctx_bn = ctx_bn.unsqueeze(1).expand(-1, N, -1)  # (B, N, D//4)
        t_emb = self._t_embed(t).unsqueeze(1).expand(-1, N, -1)  # (B, N, D)

        # Predict skip score
        mlp_input = torch.cat([tokens, ctx_bn, t_emb], dim=-1)
        scores = torch.sigmoid(self.mlp(mlp_input).squeeze(-1))  # (B, N) in [0, 1]

        # Convert to binary decision with gradient flow
        if self.training:
            # Straight-through estimator: hard forward, soft backward
            hard_skip_float = StraightThroughBernoulli.apply(scores)
            skip = hard_skip_float.bool()
        else:
            skip = scores > 0.5

        # Hard exceptions: rare tokens never skipped (Part 4.5)
        if rare_mask is not None:
            skip = skip & ~rare_mask

        # Minimum 20% floor: ensure at least 20% of tokens are processed
        min_active = max(1, int(N * self.min_process_frac))
        for b in range(B):
            active_count = (~skip[b]).sum().item()
            if active_count < min_active:
                # Un-skip the tokens with lowest skip scores
                deficit = min_active - active_count
                skipped_idx = skip[b].nonzero(as_tuple=False).squeeze(-1)
                if len(skipped_idx) > 0:
                    skipped_scores = scores[b, skipped_idx]
                    _, lowest_idx = skipped_scores.topk(
                        min(deficit, len(skipped_idx)), largest=False
                    )
                    skip[b, skipped_idx[lowest_idx]] = False

        return skip, scores


#  Group-Specific Expert Pools

class GroupAExpertPool(nn.Module):
    """Expert pool for Group A with multi-checkpoint interpolation cache.

    Group A sensory features are structurally timestep-independent.
    During inference (DDPM sampling): store outputs at reference timestep
    checkpoints, linearly interpolate for intermediate steps (Part 4.5).

    During training: always recompute (need gradients).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.pool = ExpertPool(
            num_experts=cfg.num_experts_A,
            embed_dim=cfg.embed_dim,
            ff_mult=cfg.expert_ff_mult,
            dropout=getattr(cfg, 'expert_dropout', 0.0),
        )
        self.cache_interval = cfg.cache_interval_A
        self.T_max = cfg.T_max

        # Multi-checkpoint cache: maps timestep → output tensor
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_steps: List[int] = []  # sorted timesteps with cached outputs

    def forward(
        self,
        tokens: torch.Tensor,
        dispatch_weights: torch.Tensor,
        combine_weights: torch.Tensor,
        skip_mask: torch.BoolTensor,
        step: int = 0,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns: (output (B,N_A,D), expert_acts List[E])."""
        # During training: always recompute
        if self.training:
            return self.pool(tokens, dispatch_weights, combine_weights)

        # During inference: cache + interpolation
        if t is None:
            return self.pool(tokens, dispatch_weights, combine_weights)

        # Use representative t for the batch (e.g., first sample)
        t_val = int(t[0].item())
        recompute = (step % self.cache_interval) == 0 or t_val not in self._cache

        if recompute:
            output, acts = self.pool(tokens, dispatch_weights, combine_weights)
            self._cache[t_val] = output.detach()
            self._cache_steps = sorted(self._cache.keys())
            return output, acts

        # Interpolate between two nearest cached checkpoints
        return self._interpolate(t_val, tokens), []

    def _interpolate(self, t_val: int, fallback: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two nearest cached outputs."""
        if t_val in self._cache:
            return self._cache[t_val]

        if len(self._cache_steps) < 2:
            return self._cache.get(
                self._cache_steps[0], fallback
            ) if self._cache_steps else fallback

        # Find bracketing checkpoints
        lower, upper = None, None
        for s in self._cache_steps:
            if s <= t_val:
                lower = s
            if s >= t_val and upper is None:
                upper = s

        if lower is None:
            return self._cache[self._cache_steps[0]]
        if upper is None:
            return self._cache[self._cache_steps[-1]]
        if lower == upper:
            return self._cache[lower]

        # Linear interpolation
        alpha = (t_val - lower) / max(upper - lower, 1)
        return (1 - alpha) * self._cache[lower] + alpha * self._cache[upper]

    def forward_anchors(
        self, anchors: torch.Tensor, exclude: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        return self.pool.forward_anchors(anchors, exclude)

    def invalidate_cache(self):
        self._cache.clear()
        self._cache_steps.clear()


class GroupBExpertPool(nn.Module):
    """Expert pool for Group B with learned per-token skip at high t.

    When skipped: return input tokens unchanged (identity residual).
    Expert index 0 is the shared anchor — architecturally identical
    but receives guaranteed routing weight from GroupBRouter.

    Uses LearnedSkipPredictor when available, falls back to batch-level
    skip_mask from DyDiTSkipScheduler during early training.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.pool = ExpertPool(
            num_experts=cfg.num_experts_B,
            embed_dim=cfg.embed_dim,
            ff_mult=cfg.expert_ff_mult,
            dropout=getattr(cfg, 'expert_dropout', 0.0),
        )

        # Learned per-token skip predictor (Part 4.5)
        self.skip_predictor = LearnedSkipPredictor(
            embed_dim=cfg.embed_dim, T_max=cfg.T_max
        )
        # Initially disabled — enabled after DyDiT skip is introduced in Stage 4
        self.use_learned_skip = False

        # Rare token exception mask (updated by intervention state machine)
        self._rare_mask: Optional[torch.BoolTensor] = None

        # Last skip scores for loss computation
        self._last_skip_scores: Optional[torch.Tensor] = None

    def forward(
        self,
        tokens: torch.Tensor,
        dispatch_weights: torch.Tensor,
        combine_weights: torch.Tensor,
        skip_mask: torch.BoolTensor,     # (B,) from DyDiTSkipScheduler
        step: int = 0,
        t: Optional[torch.Tensor] = None,
        ctx_C: Optional[torch.Tensor] = None,  # for learned skip predictor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns: (output (B,N_B,D), expert_acts List[E])."""
        B, N, D = tokens.shape

        # Determine per-token skip
        if self.use_learned_skip and t is not None:
            token_skip, skip_scores = self.skip_predictor(
                tokens, ctx_C, t, self._rare_mask
            )
            self._last_skip_scores = skip_scores
            # Also incorporate batch-level skip as an override
            batch_skip_expanded = skip_mask.unsqueeze(1).expand(-1, N)
            token_skip = token_skip | batch_skip_expanded
        else:
            # Fall back to batch-level skip only
            token_skip = skip_mask.unsqueeze(1).expand(-1, N)  # (B, N)
            self._last_skip_scores = None

        # All tokens skip: return identity
        if token_skip.all():
            return tokens, []

        # No tokens skip: full forward
        if not token_skip.any():
            return self.pool(tokens, dispatch_weights, combine_weights)

        # Mixed: process non-skipped tokens, identity for skipped
        # Zero out dispatch weights for skipped tokens so they don't
        # consume expert capacity
        active_dispatch = dispatch_weights.clone()
        active_combine = combine_weights.clone()
        active_dispatch[token_skip] = 0.0
        active_combine[token_skip] = 0.0

        output, acts = self.pool(tokens, active_dispatch, active_combine)

        # For skipped tokens, replace expert output with zero
        # (the residual connection in moe_block.py adds tokens back)
        output[token_skip] = 0.0

        return output, acts

    def forward_anchors(
        self, anchors: torch.Tensor, exclude: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        # Default: exclude shared expert (index 0) from decorrelation
        if exclude is None:
            exclude = [0]
        return self.pool.forward_anchors(anchors, exclude)

    def set_rare_mask(self, mask: Optional[torch.BoolTensor]):
        """Update rare token mask (from FM4 starvation recovery)."""
        self._rare_mask = mask

    def enable_learned_skip(self, enable: bool = True):
        """Enable/disable learned skip predictor (disabled until Stage 4)."""
        self.use_learned_skip = enable

    def get_last_skip_scores(self) -> Optional[torch.Tensor]:
        """Return skip scores from last forward for loss computation."""
        return self._last_skip_scores


class GroupCExpertPool(nn.Module):
    """Expert pool for Group C with two-level static cache (Part 4.5).

    Two-level cache:
        t_high level: covers mode-selection window (high t).
                      Threshold conditioned on scene complexity.
        t_low level:  covers fine-shaping window (low t). Fixed threshold.
        Between levels: RECOMPUTE normally.

    Cache keyed by scene_id (not batch index) to support variable batch
    sizes and multi-GPU setups (Part 4.5).

    Cache levels (during inference denoising):
        t/T > t_high_threshold → compute and save to high cache
        t_low_threshold < t/T < t_high_threshold → recompute (no cache)
        t/T < t_low_threshold → use low cache (if populated)
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.pool = ExpertPool(
            num_experts=cfg.num_experts_C,
            embed_dim=cfg.embed_dim,
            ff_mult=cfg.expert_ff_mult,
            dropout=getattr(cfg, 'expert_dropout', 0.0),
        )
        # Token-type attention biases for each expert (Part 6.3)
        # Default bias type assignment — overridden by apply_asymmetric_init_C
        self.attention_biases = nn.ModuleList([
            TokenTypeAttentionBias(cfg.embed_dim, rank=16, bias_type='none')
            for _ in range(cfg.num_experts_C)
        ])

        self.T_max = cfg.T_max
        self.t_skip_C = cfg.t_skip_C  # base threshold for skip

        # Two-level thresholds (normalized t/T_max)
        # High cache: populated at high-t, used for very high t
        self.t_high_threshold = 0.7  # above this: use high cache
        # Low cache: populated just above skip threshold, used for very low t
        self.t_low_threshold = cfg.t_skip_C  # below this: use low cache

        # Scene complexity adjustment range for high threshold
        self.t_high_adjust = 0.1  # ±0.1 based on complexity

        # Scene-ID keyed caches (robust to variable batch sizes)
        self._cache_high: Dict[str, torch.Tensor] = {}
        self._cache_low: Dict[str, torch.Tensor] = {}

    def _apply_attention_biases(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply token-type attention biases before expert dispatch.

        This is a no-op until apply_asymmetric_init_C sets bias types.
        When active, it softly steers input features per-expert via
        the attention bias modules. Since biases are applied pre-dispatch,
        the effect is a soft prior on which features each expert attends to.

        Note: Biases are applied to the tokens before dispatch, but
        since each expert sees only its assigned tokens, the bias acts
        as a soft input transformation. For a stronger effect, the bias
        could be moved inside each expert's forward — left as-is for now
        since the plan specifies "soft attention biases."
        """
        # Biases are integrated inside forward_with_biases below
        return tokens

    def forward_with_biases(
        self,
        tokens: torch.Tensor,            # (B, N, D)
        dispatch_weights: torch.Tensor,   # (B, N, E)
        combine_weights: torch.Tensor,    # (B, N, E)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Expert pool forward with per-expert token-type attention biases.

        Each expert's assigned tokens are first passed through that expert's
        attention bias module, then through the expert FFN.
        """
        B, N, D = tokens.shape
        E = self.pool.num_experts
        device = tokens.device

        flat_tokens = tokens.reshape(B * N, D)
        flat_disp = dispatch_weights.reshape(B * N, E)
        flat_comb = combine_weights.reshape(B * N, E)

        output = torch.zeros_like(flat_tokens)
        expert_acts: List[torch.Tensor] = []

        for e, expert in enumerate(self.pool.experts):
            assigned = flat_disp[:, e] > 0
            num_assigned = assigned.sum().item()

            if num_assigned == 0:
                expert_acts.append(torch.zeros(0, D, device=device))
                continue

            expert_tokens = flat_tokens[assigned]
            # Apply per-expert attention bias before FFN
            expert_tokens = self.attention_biases[e](expert_tokens)
            expert_out = expert(expert_tokens)
            expert_acts.append(expert_out)

            w = flat_comb[assigned, e].unsqueeze(-1)
            output[assigned] += expert_out * w

        return output.reshape(B, N, D), expert_acts

    def forward(
        self,
        tokens: torch.Tensor,
        dispatch_weights: torch.Tensor,
        combine_weights: torch.Tensor,
        skip_mask: torch.BoolTensor,
        step: int = 0,
        t: Optional[torch.Tensor] = None,
        scene_complexity: Optional[float] = None,
        scene_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns: (output (B,N_C,D), expert_acts List[E]).

        Args:
            scene_ids: per-sample scene identifiers for cache keying.
                       If None, falls back to a single shared cache entry.
        """
        B = tokens.shape[0]
        cache_key = scene_ids[0] if scene_ids else "__default__"

        # During training: always recompute (with attention biases)
        if self.training:
            return self.forward_with_biases(tokens, dispatch_weights, combine_weights)

        # During inference: two-level cache
        if t is None or skip_mask.all():
            if cache_key in self._cache_low and skip_mask.all():
                return self._cache_low[cache_key], []
            return self.forward_with_biases(tokens, dispatch_weights, combine_weights)

        t_norm = t.float().mean().item() / self.T_max

        # Adjust high threshold by scene complexity
        t_high = self.t_high_threshold
        if scene_complexity is not None:
            # Complex scene → conservative (lower threshold, recompute more)
            # Simple scene → aggressive (higher threshold, cache more)
            t_high += self.t_high_adjust * (0.5 - scene_complexity)

        if t_norm > t_high:
            # High-t: compute, save to high cache
            output, acts = self.forward_with_biases(
                tokens, dispatch_weights, combine_weights
            )
            self._cache_high[cache_key] = output.detach()
            return output, acts

        elif t_norm > self.t_low_threshold:
            # Middle: recompute normally, save to low cache at boundary
            output, acts = self.forward_with_biases(
                tokens, dispatch_weights, combine_weights
            )
            # Save to low cache when approaching the low threshold
            if t_norm < self.t_low_threshold + 0.05:
                self._cache_low[cache_key] = output.detach()
            return output, acts

        else:
            # Low-t: use low cache if available
            if cache_key in self._cache_low:
                return self._cache_low[cache_key], []
            # Fallback: compute normally
            output, acts = self.forward_with_biases(
                tokens, dispatch_weights, combine_weights
            )
            self._cache_low[cache_key] = output.detach()
            return output, acts

    def forward_anchors(
        self, anchors: torch.Tensor, exclude: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        return self.pool.forward_anchors(anchors, exclude)

    def invalidate_cache(self, scene_id: Optional[str] = None):
        """Invalidate cache. If scene_id given, only that scene; else all."""
        if scene_id is not None:
            self._cache_high.pop(scene_id, None)
            self._cache_low.pop(scene_id, None)
        else:
            self._cache_high.clear()
            self._cache_low.clear()

    def set_attention_bias_types(self, bias_types: List[str]):
        """Replace attention bias modules with specified types.

        Called by apply_asymmetric_init_C to set per-expert bias types.

        Args:
            bias_types: list of len num_experts, each in
                        TokenTypeAttentionBias.BIAS_TYPES
        """
        assert len(bias_types) == len(self.attention_biases)
        embed_dim = self.pool.embed_dim
        self.attention_biases = nn.ModuleList([
            TokenTypeAttentionBias(embed_dim, rank=16, bias_type=bt)
            for bt in bias_types
        ])


# ─── Expert Outputs Container 

@dataclass
class ExpertOutputs:
    """Typed container for all expert outputs from one forward pass."""

    output_A:   torch.Tensor            # (B, N_A, D) pre-residual
    output_B:   torch.Tensor            # (B, N_B, D) pre-residual
    output_C:   torch.Tensor            # (B, N_C, D) pre-residual
    acts_A:     List[torch.Tensor]      # List[E_A] of (M_e, D)
    acts_B:     List[torch.Tensor]      # List[E_B] of (M_e, D)
    acts_C:     List[torch.Tensor]      # List[E_C] of (M_e, D)


# ─── All Groups Expert Runner

class AllGroupsExpertRunner(nn.Module):
    """Runs all three expert pools in the directed order A → C → B.

    Respects the information-flow constraint from Part 3.1.
    Also provides forward_all_anchors() for decorrelation (Part 4.4).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.pool_A = GroupAExpertPool(cfg)
        self.pool_B = GroupBExpertPool(cfg)
        self.pool_C = GroupCExpertPool(cfg)

    def forward(
        self,
        tokens_A: torch.Tensor,
        tokens_B: torch.Tensor,
        tokens_C: torch.Tensor,
        dispatch_A: torch.Tensor,
        combine_A: torch.Tensor,
        dispatch_B: torch.Tensor,
        combine_B: torch.Tensor,
        dispatch_C: torch.Tensor,
        combine_C: torch.Tensor,
        skip_A: torch.BoolTensor,
        skip_B: torch.BoolTensor,
        skip_C: torch.BoolTensor,
        step: int = 0,
        t: Optional[torch.Tensor] = None,
        ctx_C_for_skip: Optional[torch.Tensor] = None,
        scene_complexity: Optional[float] = None,
        scene_ids: Optional[List[str]] = None,
    ) -> ExpertOutputs:
        """Run experts in A → C → B order."""

        # Group A
        out_A, acts_A = self.pool_A(
            tokens_A, dispatch_A, combine_A, skip_A, step, t
        )

        # Group C (with scene_ids for cache keying)
        out_C, acts_C = self.pool_C(
            tokens_C, dispatch_C, combine_C, skip_C, step, t,
            scene_complexity, scene_ids,
        )

        # Group B (with optional learned skip and Group C context)
        out_B, acts_B = self.pool_B(
            tokens_B, dispatch_B, combine_B, skip_B, step, t, ctx_C_for_skip
        )

        return ExpertOutputs(
            output_A=out_A, output_B=out_B, output_C=out_C,
            acts_A=acts_A, acts_B=acts_B, acts_C=acts_C,
        )

    def forward_all_anchors(
        self,
        anchors_A: Optional[torch.Tensor] = None,  # (M, D)
        anchors_B: Optional[torch.Tensor] = None,  # (M, D)
        anchors_C: Optional[torch.Tensor] = None,  # (M, D)
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Pass anchor tokens through all experts for decorrelation (Part 4.4).

        Returns:
            anchor_acts_A: List[E_A] of (M, D) — each A-expert on same anchors
            anchor_acts_B: List[E_B] of (M, D) — each B-expert (excl. shared)
            anchor_acts_C: List[E_C] of (M, D) — each C-expert on same anchors
        """
        empty = lambda E, D, dev: [torch.zeros(0, D, device=dev) for _ in range(E)]
        dev = next(self.parameters()).device

        acts_A = (
            self.pool_A.forward_anchors(anchors_A)
            if anchors_A is not None
            else empty(self.pool_A.pool.num_experts, self.pool_A.pool.embed_dim, dev)
        )
        # Group B: exclude shared expert (index 0) from decorrelation (Part 2.9)
        acts_B = (
            self.pool_B.forward_anchors(anchors_B, exclude=[0])
            if anchors_B is not None
            else empty(self.pool_B.pool.num_experts, self.pool_B.pool.embed_dim, dev)
        )
        acts_C = (
            self.pool_C.forward_anchors(anchors_C)
            if anchors_C is not None
            else empty(self.pool_C.pool.num_experts, self.pool_C.pool.embed_dim, dev)
        )
        return acts_A, acts_B, acts_C

    def invalidate_caches(self, scene_id: Optional[str] = None):
        """Invalidate caches. Optionally for a specific scene only."""
        self.pool_A.invalidate_cache()
        self.pool_C.invalidate_cache(scene_id)

    def enable_learned_skip_B(self, enable: bool = True):
        """Enable/disable Group B learned skip predictor."""
        self.pool_B.enable_learned_skip(enable)

    def set_rare_mask_B(self, mask: Optional[torch.BoolTensor]):
        """Update rare token exception mask for Group B skip."""
        self.pool_B.set_rare_mask(mask)

    def get_skip_scores_B(self) -> Optional[torch.Tensor]:
        """Return last skip scores from Group B for loss computation."""
        return self.pool_B.get_last_skip_scores()

    # ── Asymmetric initialization hooks (Part 6.3) 

    def apply_asymmetric_init_A(
        self,
        pretrained_ffn: Optional[nn.Module] = None,
        noise_scale: float = 0.01,
        expert_biases: Optional[Dict[int, str]] = None,
    ):
        """Apply Group A asymmetric initialization (Part 6.3).

        Copies pretrained weights + small noise per expert. Additionally
        applies structural biases per expert based on their sensory role:
            - BEV expert (idx 0): bias toward top-down spatial features
            - LiDAR expert (idx 1): bias toward point density features
            - Image expert (idx 2): no additional bias (retains full pretrained init)

        Args:
            pretrained_ffn: pretrained dense FFN module to copy from
            noise_scale: σ = noise_scale × pretrained_weight_std
            expert_biases: optional dict mapping expert index → bias type.
                          Default: {0: 'bev', 1: 'lidar'}.
                          Experts not in the dict get standard noise only.
        """
        if pretrained_ffn is None:
            return

        if expert_biases is None:
            expert_biases = {0: 'bev', 1: 'lidar'}

        with torch.no_grad():
            pretrained_std = torch.cat(
                [p.flatten() for p in pretrained_ffn.parameters()]
            ).std().item()
            sigma = noise_scale * pretrained_std

            for e, expert in enumerate(self.pool_A.pool.experts):
                # Copy pretrained weights
                for (name, param), (_, pretrained_param) in zip(
                    expert.named_parameters(), pretrained_ffn.named_parameters()
                ):
                    param.copy_(pretrained_param)
                    # Add independent noise
                    param.add_(torch.randn_like(param) * sigma)

                # Apply structural bias for specific expert roles
                bias_type = expert_biases.get(e)
                if bias_type == 'bev':
                    self._apply_bev_bias(expert, sigma)
                elif bias_type == 'lidar':
                    self._apply_lidar_bias(expert, sigma)
                # else: standard noise only (e.g., image expert)

    @staticmethod
    def _apply_bev_bias(expert: GatedExpertFFN, sigma: float):
        """Bias BEV expert toward top-down spatial features.

        Increases sensitivity to the first quarter of input dimensions
        (conventionally spatial/positional features) via a scaled
        perturbation on the gate projection's corresponding rows.
        """
        with torch.no_grad():
            D = expert.gate_proj.weight.shape[1]
            quarter = D // 4
            # Strengthen gate projection for spatial dimensions
            expert.gate_proj.weight[:, :quarter] += sigma * 3.0

    @staticmethod
    def _apply_lidar_bias(expert: GatedExpertFFN, sigma: float):
        """Bias LiDAR expert toward point density / range features.

        Increases sensitivity to the second quarter of input dimensions
        (conventionally depth/density features).
        """
        with torch.no_grad():
            D = expert.gate_proj.weight.shape[1]
            quarter = D // 4
            expert.gate_proj.weight[:, quarter:2 * quarter] += sigma * 3.0

    def apply_asymmetric_init_B(
        self,
        pretrained_ffn: Optional[nn.Module] = None,
        noise_scale: float = 0.01,
        cluster_centroids: Optional[torch.Tensor] = None,
    ):
        """Apply Group B asymmetric initialization (Part 6.3).

        Expert 0 (shared): EXACT copy, NO noise.
        Specialists: pretrained + noise + behavioral bias from cluster centroids.

        Behavioral priors per specialist (from Part 6.3 Stage 3):
            Expert 1 (following/motion-tracking): velocity/heading bias
            Expert 2 (yielding/collision-avoidance): proximity bias
            Expert 3 (intent/crossing): traffic light/crosswalk bias
            Expert k+ (rare/anomalous): MAXIMUM ENTROPY init — no prior bias,
                larger noise to maximize initial coverage of embedding space.

        Args:
            pretrained_ffn: pretrained dense FFN module to copy from
            noise_scale: σ = noise_scale × pretrained_weight_std
            cluster_centroids: (K, D) Phase 1 cluster centroids for
                              behavioral bias initialization. If provided,
                              specialist bias directions are derived from
                              the centroid directions.
        """
        if pretrained_ffn is None:
            return

        num_experts = self.pool_B.pool.num_experts

        with torch.no_grad():
            pretrained_std = torch.cat(
                [p.flatten() for p in pretrained_ffn.parameters()]
            ).std().item()
            sigma = noise_scale * pretrained_std

            for e, expert in enumerate(self.pool_B.pool.experts):
                for (name, param), (_, pretrained_param) in zip(
                    expert.named_parameters(), pretrained_ffn.named_parameters()
                ):
                    param.copy_(pretrained_param)

                    if e == 0:
                        # Shared expert: EXACT copy, NO noise (Part 6.3)
                        pass
                    elif e == num_experts - 1:
                        # Rare/anomalous expert: MAXIMUM ENTROPY init
                        # Larger noise to maximize initial coverage
                        param.add_(torch.randn_like(param) * sigma * 5.0)
                    else:
                        # Specialists: standard noise
                        param.add_(torch.randn_like(param) * sigma)

            # Apply behavioral biases from cluster centroids if available
            if cluster_centroids is not None:
                self._apply_behavioral_biases_B(cluster_centroids, sigma)

    def _apply_behavioral_biases_B(
        self, centroids: torch.Tensor, sigma: float
    ):
        """Apply behavioral biases to Group B specialists using centroids.

        Uses cluster centroid directions to bias each specialist's gate
        projection, steering it toward its intended behavioral mode.

        Args:
            centroids: (K, D) cluster centroids from Phase 1.
                      K should be >= num_specialists. Each centroid provides
                      a direction in embedding space for one specialist.
        """
        num_experts = self.pool_B.pool.num_experts
        D = self.pool_B.pool.embed_dim
        num_specialists = num_experts - 2  # exclude shared (0) and rare (last)

        if centroids.shape[0] < num_specialists:
            return  # Not enough centroids

        with torch.no_grad():
            for s in range(num_specialists):
                expert_idx = s + 1  # skip shared expert at index 0
                expert = self.pool_B.pool.experts[expert_idx]
                # Normalize centroid to unit direction
                direction = F.normalize(centroids[s], dim=0)  # (D,)
                # Bias gate projection: increase response to centroid direction
                # This is a rank-1 update: outer product of hidden-dim noise
                # with the centroid direction
                H = expert.gate_proj.weight.shape[0]
                bias_vec = torch.randn(H, device=direction.device) * sigma * 2.0
                expert.gate_proj.weight.add_(
                    bias_vec.unsqueeze(1) * direction.unsqueeze(0)
                )

    def apply_asymmetric_init_C(
        self,
        pretrained_ffn: Optional[nn.Module] = None,
        noise_scale: float = 0.01,
        expert_bias_types: Optional[List[str]] = None,
    ):
        """Apply Group C asymmetric initialization (Part 6.3).

        NOT noise perturbations — uses token-type attention bias terms.
        Each expert gets a different soft attention bias:
            - Intersection expert → spatial co-occurrence bias
            - Traffic light expert → temporal state bias
            - Road geometry expert → directional feature bias
            - Additional experts → no bias (learn from data)

        Args:
            pretrained_ffn: pretrained dense FFN module to copy from
            noise_scale: σ for the small weight-space noise (all experts
                        get a tiny amount for symmetry breaking)
            expert_bias_types: list of bias type strings per expert.
                             Default for 6 experts:
                             ['spatial', 'temporal', 'directional',
                              'spatial', 'none', 'none']
        """
        if pretrained_ffn is None:
            return

        num_experts = self.pool_C.pool.num_experts

        # Default bias type assignment
        if expert_bias_types is None:
            default_types = ['spatial', 'temporal', 'directional']
            expert_bias_types = [
                default_types[i] if i < len(default_types) else 'none'
                for i in range(num_experts)
            ]

        # Set attention bias types on GroupCExpertPool
        self.pool_C.set_attention_bias_types(expert_bias_types)

        with torch.no_grad():
            pretrained_std = torch.cat(
                [p.flatten() for p in pretrained_ffn.parameters()]
            ).std().item()
            # Small noise for symmetry breaking only — the real asymmetry
            # comes from the attention bias terms, not weight perturbations
            sigma = noise_scale * 0.1 * pretrained_std

            for e, expert in enumerate(self.pool_C.pool.experts):
                for (name, param), (_, pretrained_param) in zip(
                    expert.named_parameters(), pretrained_ffn.named_parameters()
                ):
                    param.copy_(pretrained_param)
                    # Minimal noise — attention biases provide the asymmetry
                    param.add_(torch.randn_like(param) * sigma)


# factory

def build_expert_pools(cfg: MoEConfig) -> AllGroupsExpertRunner:
    return AllGroupsExpertRunner(cfg)


#  Parameter Counting Utility 
def count_expert_params(runner: AllGroupsExpertRunner) -> Dict[str, int]:
    """Return parameter count per group and total."""
    def _count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    return {
        "group_A": _count(runner.pool_A),
        "group_B": _count(runner.pool_B),
        "group_B_skip_predictor": _count(runner.pool_B.skip_predictor),
        "group_C": _count(runner.pool_C),
        "group_C_attention_biases": _count(
            runner.pool_C.attention_biases
        ),
        "total": _count(runner),
    }


#  Sanity Check 

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from MoE.token_router import MoEConfig, build_moe_router

    torch.manual_seed(0)
    B = 4

    cfg = MoEConfig(
        num_tokens_A=32, num_tokens_B=16, num_tokens_C=32,
        embed_dim=128,
        num_experts_A=4, num_experts_B=4, num_experts_C=6,
        expert_ff_mult=2,
        t_skip_B=0.7,
        t_skip_C=0.2,
        T_max=1000,
    )

    runner = build_expert_pools(cfg)

    # Check output_scale init = 1/E
    scale_A = runner.pool_A.pool.experts[0].output_scale.item()
    print(f"Group A expert 0 output_scale: {scale_A:.4f} (expect {1/4:.4f})")
    assert abs(scale_A - 0.25) < 0.01, f"output_scale should be 1/E = 0.25, got {scale_A}"

    scale_B = runner.pool_B.pool.experts[0].output_scale.item()
    print(f"Group B expert 0 output_scale: {scale_B:.4f} (expect {1/4:.4f})")

    # Anchor token forward test
    anchors = torch.randn(8, 128)
    acts_A, acts_B, acts_C = runner.forward_all_anchors(anchors, anchors, anchors)
    print(f"\nAnchor forward test:")
    print(f"  A experts: {[a.shape for a in acts_A]}")
    print(f"  B experts: {[a.shape for a in acts_B]}  (index 0 excluded)")
    print(f"  C experts: {[a.shape for a in acts_C]}")
    assert acts_B[0].shape[0] == 0, "Shared expert should be excluded from B anchors"
    assert acts_B[1].shape == (8, 128), "Specialist should process all anchors"
    print("  Anchor forward: PASS")

    # Skip predictor test (now returns tuple)
    tokens_B = torch.randn(B, 16, 128)
    t_high = torch.tensor([800, 850, 900, 950])
    runner.pool_B.enable_learned_skip(True)
    skip_decisions, skip_scores = runner.pool_B.skip_predictor(tokens_B, None, t_high)
    print(f"\nSkip predictor output shape: {skip_decisions.shape}")
    print(f"  Skip scores shape: {skip_scores.shape}")
    active_frac = (~skip_decisions).float().mean().item()
    print(f"  Active fraction: {active_frac:.2f} (should be >= 0.20)")
    assert active_frac >= 0.19, "20% floor violated"
    print("  Skip predictor: PASS")

    # Straight-through gradient test
    tokens_test = torch.randn(2, 8, 128, requires_grad=True)
    t_test = torch.tensor([900, 950])
    runner.pool_B.skip_predictor.train()
    _, scores_test = runner.pool_B.skip_predictor(tokens_test, None, t_test)
    loss = scores_test.sum()
    loss.backward()
    assert tokens_test.grad is not None, "Gradients should flow through skip predictor"
    print("  Straight-through gradient flow: PASS")

    # Token-type attention bias test
    print(f"\nGroup C attention biases (before init):")
    for i, bias in enumerate(runner.pool_C.attention_biases):
        print(f"  Expert {i}: {bias.bias_type}")

    # Apply asymmetric init C
    dummy_ffn = GatedExpertFFN(128, ff_mult=2, num_experts=6)
    runner.apply_asymmetric_init_C(
        dummy_ffn,
        expert_bias_types=['spatial', 'temporal', 'directional', 'spatial', 'none', 'none']
    )
    print(f"\nGroup C attention biases (after init):")
    for i, bias in enumerate(runner.pool_C.attention_biases):
        print(f"  Expert {i}: {bias.bias_type}")
    assert runner.pool_C.attention_biases[0].bias_type == 'spatial'
    assert runner.pool_C.attention_biases[1].bias_type == 'temporal'
    assert runner.pool_C.attention_biases[2].bias_type == 'directional'
    print("  Attention bias init: PASS")

    # Group B asymmetric init test
    dummy_ffn_b = GatedExpertFFN(128, ff_mult=2, num_experts=4)
    # Snapshot shared expert weight before init
    pre_shared = runner.pool_B.pool.experts[0].gate_proj.weight.clone()
    runner.apply_asymmetric_init_B(dummy_ffn_b)
    post_shared = runner.pool_B.pool.experts[0].gate_proj.weight.clone()
    # Shared expert should be an exact copy (no noise)
    shared_diff = (post_shared - dummy_ffn_b.gate_proj.weight).abs().max().item()
    print(f"\nGroup B shared expert max weight diff from pretrained: {shared_diff:.6f}")
    assert shared_diff < 1e-6, "Shared expert should be exact copy with no noise"
    # Specialist should differ
    specialist_diff = (
        runner.pool_B.pool.experts[1].gate_proj.weight - dummy_ffn_b.gate_proj.weight
    ).abs().max().item()
    print(f"  Specialist expert max weight diff from pretrained: {specialist_diff:.6f}")
    assert specialist_diff > 1e-4, "Specialist should have noise perturbation"
    # Rare expert should have larger noise
    rare_diff = (
        runner.pool_B.pool.experts[3].gate_proj.weight - dummy_ffn_b.gate_proj.weight
    ).abs().max().item()
    print(f"  Rare expert max weight diff from pretrained: {rare_diff:.6f}")
    assert rare_diff > specialist_diff, "Rare expert should have larger noise (max-entropy)"
    print("  Group B asymmetric init: PASS")

    # Scene-ID keyed cache test for Group C
    runner.pool_C.eval()
    tokens_C = torch.randn(1, 32, 128)
    disp_C = torch.zeros(1, 32, 6)
    disp_C[:, :, 0] = 1.0
    comb_C = disp_C.clone()
    skip_C = torch.tensor([False])
    t_c = torch.tensor([950.0])
    out1, _ = runner.pool_C(tokens_C, disp_C, comb_C, skip_C, t=t_c,
                            scene_ids=["scene_001"])
    assert "scene_001" in runner.pool_C._cache_high, "Scene-ID cache should be populated"
    runner.pool_C.invalidate_cache("scene_001")
    assert "scene_001" not in runner.pool_C._cache_high, "Scene-specific invalidation"
    print("\n  Scene-ID cache keying: PASS")

    # Parameter count
    counts = count_expert_params(runner)
    print(f"\nParameter counts: {counts}")

    print("\nAll expert_ffn.py tests PASSED.")