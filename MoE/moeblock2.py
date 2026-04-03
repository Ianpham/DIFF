"""
DDPM/moe/moe_block.py

Full MoE Transformer Block — wires router, directed attention, expert FFNs,
decorrelation loss, and residual connections into one coherent forward pass.

This is the single building block that replaces a standard FFN transformer
layer inside the trajectory-prediction decoder.  Stack N of these blocks to
form the full MoE decoder backbone.


Forward pass protocol (two-pass routing required by directed A->C->B flow):


  Pass 1: Shared Self-Attention over all tokens (cross-modal integration)
           Tokens laid out as [A | C | B], directed attention mask applied.
           After this, every token has already "seen" its allowed upstream
           groups — Group C has attended to A, Group B has attended to A+C.

  Pass 2 (Routing pass 1 — Group A + Group C):
           Group A router:  spatial gate -> dispatch/combine weights for A.
           Group C router:  structural gate -> dispatch/combine weights for C.
           Run Group A expert pool -> expert_output_A.
           Run Group C expert pool -> expert_output_C.
           Apply residuals: tokens_A += expert_output_A
                            tokens_C += expert_output_C

  Pass 3 (Routing pass 2 — Group B, cross-group conditioned):
           Group B router:  cross-attention gate over (expert_output_A,
                            expert_output_C) -> dispatch/combine weights for B.
           Run Group B expert pool -> expert_output_B.
           Apply residual: tokens_B += expert_output_B

  Pass 4: Decorrelation loss computation from expert activations collected
           in passes 2-3.  Loss is returned to the caller (training loop)
           to add to the total objective.


Why self-attention comes BEFORE routing (not after):


  MoMa (Meta, 2024): "shared self-attention provides cross-modal integration;
  expert FFNs provide modality-specific deep processing."
  Doing attention first means:
    • Group B's gate query already has map-aware context when it fires
      (because B tokens attended to C tokens in the shared attn).
    • Group C's structural router has confirmation from Group A sensors.
    • Experts then process tokens that are already cross-modally enriched.
  Doing experts first would route on impoverished token representations.


Residual structure:


  Pre-norm convention (same as existing GaussianRefinementBlock):
    x = x + Attention(LayerNorm(x))     [shared self-attn]
    x = x + Expert(LayerNorm(x))        [group expert FFN]

  Group-local LayerNorm is applied before BOTH the attention and the expert
  FFN (separate LN modules for each, since attention LN operates on the
  full concatenated sequence while expert LN is per-group).


ARCHITECTURE STATUS
-------------------
Fixes applied in this version:
  [FIX-1]  Double residual removed — SharedSelfAttention now returns only
           attn_out (no internal residual).  MoEBlock applies the single
           correct external residual.
  [FIX-2]  Shared LayerNorm inside SharedSelfAttention replaced with
           GroupLocalLayerNorm — eliminates the cross-group statistics
           leakage path warned against in 1.3.2 of the plan.
  [FIX-3]  Double tokenization removed — router is called with a flag to
           skip re-tokenization on blocks after the first; group identity
           embeddings are only added once.
  [FIX-4]  Router called twice per block with redundant A/C computation
           eliminated — second call uses route_B_only=True fast path.
  [FIX-5]  Per-group capacity factors made explicit (no derived *1.33).
  [FIX-6]  MoEBlockOutput field renamed decor_loss -> decorr_loss.
  [FIX-7]  Typos fixed throughout.

Scaffolded stubs (not yet implemented — require new modules):
  [STUB-A] DirectedCrossAttention — separate A->C, A->B, C->B modules with
           per-direction Q/K/V projections and stop-gradients (3.2).
  [STUB-B] WarmUpCrossAttentionLayer — lightweight layer before main stack
           (3.3).
  [STUB-C] GroupBInternalPipeline — Stages 1-3: ego-centric cross-attention,
           ego-proximity-filtered agent-agent attention, map re-weighting
           scalar gate (2.3-2.6).
  [STUB-D] IntentionHeads — vehicle 6-class + pedestrian 2-class MLP heads
           with L_intention loss (2.7).

This file provides:
    MoEBlockConfig          — merged config (wraps MoEConfig + DecorrConfig)
    SharedSelfAttention     — multi-head attention with directed mask
    DirectedCrossAttention  — [STUB-A] per-direction cross-attn modules
    GroupBInternalPipeline  — [STUB-C] Group B stages 1-3
    IntentionHeads          — [STUB-D] pre-gate intention prediction
    MoEBlock                — single transformer block (stack of these)
    MoEBlockOutput          — typed return bundle
    StackedMoEBlocks        — N stacked blocks (the full backbone module)
    build_moe_backbone      — factory
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.token_router import (
    MoEConfig, ModalityMoERouter, RouterOutput,
    GroupLocalLayerNorm, DirectedAttentionMask,
    GROUP_A, GROUP_B, GROUP_C
)

from MoE.expert_ffn import AllGroupsExpertRunner, ExpertOutputs, build_expert_pools
from MoE.decorrelation_loss import (
    DecorrConfig, ThreeGroupDecorrLoss, build_decorr_loss
)

from MoE.direct_attention import DirectedCrossAttention
from MoE.warmup_attention import WarmUpCrossAttentionLayer
from MoE.intention_heads  import IntentionHeads
from MoE.groupb_pipeline  import GroupBInternalPipeline
 
# Merged block config
 
@dataclass
class MoEBlockConfig:
    """All hyperparameters for one MoEBlock (and stacks of them).

    Combines MoEConfig (routing) and DecorrConfig (decorrelation) so callers
    only need to manage one config object.
    """

    # Token / feature dimensions
    embed_dim:      int = 256
    num_tokens_A:   int = 64
    num_tokens_B:   int = 64
    num_tokens_C:   int = 128

    # Expert configuration
    num_experts_A:  int = 4
    num_experts_B:  int = 4
    num_experts_C:  int = 6   # >= NUM_C_TYPES (6)
    top_k_A:        int = 2
    top_k_B:        int = 2
    top_k_C:        int = 1
    expert_ff_mult: int = 4
    expert_dropout: float = 0.0

    # Shared self-attention
    num_attn_heads: int = 8
    attn_dropout:   float = 0.0

    # Routing hyperparameters
    # [FIX-5] Per-group capacity factors are now explicit — no derived *1.33.
    capacity_factor_A:      float = 1.5
    capacity_factor_B:      float = 1.5
    capacity_factor_C:      float = 2.0   # plan 1.2: C has heavy-tailed distribution
    routing_floor_base:          float = 0.05
    ortho_reg_weight:       float = 1e-3
    stopgrad_C_to_B:        bool  = True
    # capacity_penalty_coeff: float = 0.01
    struct_router_temp:     float = 0.1
    gate_num_heads:         int   = 4

    # DyDiT skip thresholds
    t_skip_C:          float = 0.2
    t_skip_B:          float = 0.7
    cache_interval_A:  int   = 5
    T_max:             int   = 1000

    # Input dimensions for tokeniser projections
    # Set these if upstream features have different dims than embed_dim.
    dim_A_in: int = 256
    dim_B_in: int = 256
    dim_C_in: int = 256

    # Decorrelation
    step_start_A:  int   = 2_000
    step_start_B:  int   = 6_000
    step_start_C:  int   = 10_000
    warmup_steps_A: int  = 2_000
    warmup_steps_B: int  = 3_000
    warmup_steps_C: int  = 2_000
    lambda_A:      float = 0.02
    lambda_B:      float = 0.03
    lambda_C:      float = 0.02
    t_decorr_B_threshold: float = 0.3
    buffer_size:   int   = 512
    min_tokens_to_update: int = 8
    sigma_C:       float = 1.0
    sigma_B:       float = 0.5
    min_expert_tokens: int = 4
    # intention heads
    lambda_intention: float = 0.1 # weight for L_intention in total loss    
    # group b configuration
    num_attn_heads_pipeline:    int = 4 # head for stage 1 & 2
    history_len:                int = 15 # LTSM steps
    use_history_encoder:        bool = True

    def to_moe_config(self) -> MoEConfig:
        return MoEConfig(
            num_tokens_A=self.num_tokens_A,
            num_tokens_B=self.num_tokens_B,
            num_tokens_C=self.num_tokens_C,
            embed_dim=self.embed_dim,
            num_experts_A=self.num_experts_A,
            num_experts_B=self.num_experts_B,
            num_experts_C=self.num_experts_C,
            top_k_A=self.top_k_A,
            top_k_B=self.top_k_B,
            top_k_C=self.top_k_C,
            capacity_factor_A=self.capacity_factor_A,
            capacity_factor_B=self.capacity_factor_B,
            capacity_factor_C=self.capacity_factor_C,
            routing_floor_base=self.routing_floor_base,
            t_skip_C=self.t_skip_C,
            t_skip_B=self.t_skip_B,
            cache_interval_A=self.cache_interval_A,
            ortho_reg_weight=self.ortho_reg_weight,
            stopgrad_C_to_B=self.stopgrad_C_to_B,
            T_max=self.T_max,
            # capacity_penalty_coeff=self.capacity_penalty_coeff,
            struct_router_temp=self.struct_router_temp,
            gate_num_heads=self.gate_num_heads,
            expert_ff_mult=self.expert_ff_mult,
        )

    def to_decorr_config(self) -> DecorrConfig:
        return DecorrConfig(
            step_start_A=self.step_start_A,
            step_start_B=self.step_start_B,
            step_start_C=self.step_start_C,
            warmup_steps_A=self.warmup_steps_A,
            warmup_steps_B=self.warmup_steps_B,
            warmup_steps_C=self.warmup_steps_C,
            lambda_A=self.lambda_A,
            lambda_B=self.lambda_B,
            lambda_C=self.lambda_C,
            buffer_size=self.buffer_size,
            min_tokens_to_update=self.min_tokens_to_update,
            sigma_C=self.sigma_C,
            sigma_B=self.sigma_B,
            min_expert_tokens=self.min_expert_tokens,
            T_max=self.T_max,
        )


 
# Shared self-attention with directed mask
 

class SharedSelfAttention(nn.Module):
    """Multi-head self-attention applied to the full [A | C | B] sequence.

    The directed attention mask enforces the A->C->B information flow:
    tokens can only attend to their allowed upstream groups (built by
    DirectedAttentionMask and registered as a buffer in ModalityMoERouter).

    Pre-norm convention: GroupLocalLayerNorm is applied OUTSIDE this module
    (by MoEBlock) before the call.  This module receives already-normed
    tokens and returns only the attention output (no residual).

    [FIX-1] Internal residual removed — caller applies the single correct
            external residual: tokens_X = tokens_X + attn_out_X.
    [FIX-2] Internal nn.LayerNorm removed — it was a single shared LN over
            the concatenated [A|C|B] sequence, creating cross-group
            statistics leakage (plan 1.3.2 warning).  Normalization is
            now handled entirely by the caller's GroupLocalLayerNorm, which
            normalizes each group independently.

    Args:
        embed_dim:   D
        num_heads:   H
        dropout:     applied to attention weights
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout:   float = 0.0,   # [FIX-7] was "droupout"
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # [FIX-2] No LayerNorm here — normalization is group-local and
        # applied upstream by MoEBlock.pre_attn_ln (GroupLocalLayerNorm).

    def forward(
        self,
        x: torch.Tensor,         # (B, N_total, D)  [A | C | B] pre-normed
        attn_mask: torch.Tensor, # (N_total, N_total) additive float mask  [FIX-7] was "addictive"
    ) -> torch.Tensor:
        """Returns (B, N_total, D) — attention output only, NO residual.

        [FIX-1] Residual is applied by the caller:
            tokens_X = tokens_X + seq_out[:, slice_X, :]
        """
        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        # [FIX-1] Return raw attention output — do NOT add residual here.
        return attn_out


 
# # [STUB-A] Directed cross-attention modules (3.2 steps 2-4)
 

# class DirectedCrossAttention(nn.Module):
#     """[STUB-A] One directed cross-attention path (e.g. A -> C).

#     Plan 3.2 requires three separate cross-attention modules — A->C, A->B,
#     C->B — each with its own Q/K/V projection matrices and stop-gradients.
#     This stub raises NotImplementedError so the missing module is loudly
#     visible at runtime rather than silently absent.

#     Implementation requirements (3.2):
#       - query_group tokens attend over key_group keys/values.
#       - Separate nn.Linear Q/K/V projections per direction (no weight sharing).
#       - stop_grad_on_keys: if True, detach key_group before projection
#         (e.g. Group A in A->C and A->B paths — Group A must not be shaped
#         by downstream groups).
#       - stop_grad_on_query_enrichment: if True, detach enriched query
#         output from key_group's gradient graph (C->B gate query path —
#         prevents Group C from being shaped by Group B routing decisions).
#       - Pre-norm: caller applies GroupLocalLayerNorm before this module.
#       - Returns residual-updated query_group tokens only.

#     Args:
#         embed_dim:                    D
#         num_heads:                    H (may be fewer than shared self-attn)
#         stop_grad_on_keys:            detach source group before K/V projection
#         stop_grad_on_query_enrichment: detach output from source gradient graph
#         dropout:                      attention dropout
#     """

#     def __init__(
#         self,
#         embed_dim: int,
#         num_heads: int,
#         stop_grad_on_keys:             bool  = False,
#         stop_grad_on_query_enrichment: bool  = False,
#         dropout:                       float = 0.0,
#     ):
#         super().__init__()
#         self.stop_grad_on_keys             = stop_grad_on_keys
#         self.stop_grad_on_query_enrichment = stop_grad_on_query_enrichment
#         # TODO: implement per-direction Q/K/V projections and cross-attention
#         raise NotImplementedError(
#             "[STUB-A] DirectedCrossAttention is not yet implemented. "
#             "See plan 3.2 steps 2-4 and the docstring above."
#         )

#     def forward(
#         self,
#         query_tokens: torch.Tensor,  # (B, N_query, D) — pre-normed
#         key_tokens:   torch.Tensor,  # (B, N_key,   D) — pre-normed
#     ) -> torch.Tensor:
#         """Returns (B, N_query, D) — cross-attn output only, NO residual."""
#         raise NotImplementedError("[STUB-A] DirectedCrossAttention.forward")


 
# # [STUB-C] Group B internal pipeline — Stages 1-3 (2.3-2.6)
 

# class GroupBInternalPipeline(nn.Module):
#     """[STUB-C] Group B Stages 1-3 before gate query formation (2.3-2.6).

#     Plan 2.3-2.6 requires a five-stage pipeline enriching Group B tokens
#     BEFORE the gate fires.  Currently Group B tokens go from LayerNorm
#     directly to the router — this is the largest architectural gap.

#     Stage 1 — Ego-centric cross-attention (2.3):
#       Each agent attends to the EGO TOKEN ONLY.
#       Query = each agent token.  Key/Value = ego token.
#       Cost: O(N) — one key per agent.

#     Stage 2 — Ego-proximity-filtered agent-agent attention (2.4):
#       Top-K neighbors by d_{i->ego} (NOT agent-agent distance).
#       K from deterministic lookup table: f(ego_speed, local_agent_density).
#       Attention bias: score(i,j) += bias_mlp(d_{i->ego}, d_{j->ego}).
#       Ego uses SEPARATE projections (ego_q_proj, ego_k_proj, ego_v_proj).
#       Applied as RESIDUAL over Stage 1 output.

#     Stage 3 — Lightweight map context re-weighting (2.6):
#       After C->B cross-attention (DirectedCrossAttention stub-A):
#         agent_repr_3 = agent_repr_2 * sigmoid(W_reweight · concat[repr_2, map_ctx])
#       Scalar correction per agent — NOT full re-attention.

#     This stub raises NotImplementedError so the missing pipeline is loudly
#     visible at runtime.

#      Decision Required Before Coding (2.2):
#       History encoder placement — should the LSTM run AFTER ego-relative
#       geometry is computed?  This affects token input structure and training
#       stages.  Current plan assumes HistoryEncoder is upstream.

#      Decision Required Before Coding (2.4):
#       Ego token routing competition — should ego have a privileged expert
#       that never participates in routing competition, or does ego go through
#       the same gate as other agents?
#     """

#     def __init__(self, embed_dim: int, num_heads: int):
#         super().__init__()
#         # TODO: implement ego-centric cross-attention (Stage 1)
#         # TODO: implement K lookup table + distance-biased sparse attention (Stage 2)
#         # TODO: implement map context re-weighting scalar gate (Stage 3)
#         raise NotImplementedError(
#             "[STUB-C] GroupBInternalPipeline is not yet implemented. "
#             "See plan 2.3-2.6 and the docstring above."
#         )

#     def forward(
#         self,
#         tokens_B:       torch.Tensor,  # (B, N_B, D) — pre-normed
#         ego_token:      torch.Tensor,  # (B, 1, D)
#         map_context:    torch.Tensor,  # (B, N_C, D) — Group C output (for Stage 3)
#         ego_speed:      torch.Tensor,  # (B,) scalar — for K lookup
#         agent_density:  torch.Tensor,  # (B,) scalar — for K lookup
#     ) -> torch.Tensor:
#         """Returns (B, N_B, D) — enriched Group B tokens, NO residual."""
#         raise NotImplementedError("[STUB-C] GroupBInternalPipeline.forward")


 
# # [STUB-D] Intention heads — pre-gate (2.7)
 

# class IntentionHeads(nn.Module):
#     """[STUB-D] Vehicle + pedestrian intention prediction heads (2.7).

#     Plan 2.7 requires explicit intention prediction before the gate fires,
#     inspired by arXiv:2409.15821.  Making the gate's implicit intention
#     recognition explicit helps decorrelation pressure differentiate experts
#     by behavioral mode rather than arbitrary activation patterns.

#     Vehicle intention head:   6-class (Lateral: L/S/R × Long: Accel/Const/Decel)
#     Pedestrian intention head: 2-class (not-crossing / crossing)
#                                SEPARATE from vehicle — do NOT force pedestrians
#                                through the 6-class vehicle structure.

#     Loss: L_intention = cross-entropy against ground truth intention labels
#     derived from future trajectories.  Acts on agent_repr_2 (Stage 2 output).
#     Strictly within Group B.

#     Logits feed directly into gate query formation (2.8).

#      Decision Required Before Coding (2.7):
#       Risk-adaptive gate confidence threshold — should high-risk agents
#       (close, converging heading, high speed) route with lower fallback to
#       shared expert?  Current plan uses a fixed threshold of 0.4 for all
#       tokens.

#     MoEBlockOutput must be extended with an `intention_loss` field once
#     this stub is implemented.
#     """

#     def __init__(self, embed_dim: int, num_vehicle_classes: int = 6):
#         super().__init__()
#         # TODO: implement 2-layer MLP vehicle head (6-class)
#         # TODO: implement 2-layer MLP pedestrian head (2-class, separate)
#         # TODO: agent_type mask to route tokens to the correct head
#         raise NotImplementedError(
#             "[STUB-D] IntentionHeads is not yet implemented. "
#             "See plan 2.7 and the docstring above."
#         )

#     def forward(
#         self,
#         agent_repr:  torch.Tensor,  # (B, N_B, D) — Stage 2 output
#         agent_types: torch.Tensor,  # (B, N_B) long — 0=vehicle, 1=pedestrian, 2=cyclist
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Returns (intention_logits (B, N_B, 6), intention_loss scalar)."""
#         raise NotImplementedError("[STUB-D] IntentionHeads.forward")


 
# MoEBlock output
 

@dataclass
class MoEBlockOutput:
    """All outputs from a single MoEBlock forward pass.

    Attributes:
        tokens_A:      (B, N_A, D)  updated Group A tokens (post-residual)
        tokens_B:      (B, N_B, D)  updated Group B tokens (post-residual)
        tokens_C:      (B, N_C, D)  updated Group C tokens (post-residual)
        decorr_loss:   scalar — decorrelation loss for this block  [FIX-6]
        capacity_loss: scalar — soft capacity constraint loss
        ortho_loss:    scalar — group identity orthogonality loss
        decorr_log:    dict   — per-group decorr breakdown (for logging)
        router_out:    RouterOutput — full routing decisions (for diagnostics)
    """
    tokens_A: torch.Tensor
    tokens_B: torch.Tensor
    tokens_C: torch.Tensor

    decorr_loss:   torch.Tensor  # [FIX-6] was "decor_loss" (one 'r')
    # capacity_loss: torch.Tensor
    bias_penalty:  torch.Tensor
    ortho_loss:    torch.Tensor

    decorr_log:  Dict
    router_out:  RouterOutput

    # TODO [STUB-D]: add intention_loss: torch.Tensor once IntentionHeads
    # is implemented (plan 2.7, contributes λ_intention · L_intention to
    # the total training objective 4.1).
    total_intention: torch.Tensor
    intention_loss: torch.Tensor   # scalar
 
# MoE Block
 

class MoEBlock(nn.Module):
    """Single MoE Transformer Block.

    Replaces the FFN sub-layer of a standard transformer block.
    The attention sub-layer is SHARED across all groups (cross-modal fusion);
    the FFN sub-layer is EXPERT-partitioned per group.

    Architecture per block:
        1.  Concat tokens as [A | C | B]
        2.  Group-local LN (pre-attention)
        3.  Shared self-attention (directed mask)
        4.  Residual: tokens_* += attn_out_*           [FIX-1]
        5.  Group-local LN (pre-expert)
        6a. Route Group A -> expert_output_A
        6b. Route Group C -> expert_output_C
        7.  Residual: tokens_A += expert_output_A
            Residual: tokens_C += expert_output_C
        8.  Route Group B (cross-group ctx = expert_output_A + C)
            -> expert_output_B                          [FIX-4]
        9.  Residual: tokens_B += expert_output_B
       10.  Decorrelation loss from expert activations

    NOT YET IMPLEMENTED IN THIS BLOCK (stubs above):
        [STUB-A] Directed cross-attention (A->C, A->B, C->B) — 3.2 steps 2-4
        [STUB-B] Warm-up cross-attention layer — 3.3 (belongs in StackedMoEBlocks)
        [STUB-C] Group B internal pipeline (Stages 1-3) — 2.3-2.6
        [STUB-D] Intention heads + loss — 2.7

    Args:
        cfg:          MoEBlockConfig
        block_id:     integer label (0-indexed) used for logging/diagnostics
        is_first_block: if True, the router's tokenizer runs normally
                        (project + add group identity embeddings).
                        If False, tokenization is skipped — group identity
                        embeddings must NOT be re-applied every block.
                        [FIX-3]
    """

    def __init__(self, cfg: MoEBlockConfig, block_id: int = 0,
                 is_first_block: bool = True):
        super().__init__()
        self.cfg            = cfg
        self.block_id       = block_id
        self.is_first_block = is_first_block   # [FIX-3]
        D  = cfg.embed_dim
        NA = cfg.num_tokens_A
        NB = cfg.num_tokens_B
        NC = cfg.num_tokens_C

        moe_cfg    = cfg.to_moe_config()
        decorr_cfg = cfg.to_decorr_config()

        # Sub-modules
        # Pre-attention group-local LayerNorm  [FIX-2]
        self.pre_attn_ln = GroupLocalLayerNorm(D)
        # Shared directed self-attention (no internal LN, no internal residual)
        self.self_attn   = SharedSelfAttention(D, cfg.num_attn_heads,
                                               cfg.attn_dropout)
        # Pre-expert group-local LayerNorm
        self.pre_exp_ln  = GroupLocalLayerNorm(D)
        # direct cross attention
        self.cross_attn_A_to_C = DirectedCrossAttention(
            embed_dim = D,
            num_heads=max(1, cfg.num_attn_heads // 2), # half head
            stop_grad_on_keys= True,
            dropout = cfg.attn_dropout,
        )
        self.cross_attn_A_to_B = DirectedCrossAttention(
            embed_dim=D,
            num_heads=max(1, cfg.num_attn_heads // 2),
            stop_grad_on_keys=True,
            stop_grad_on_query_enrichment=False,
            dropout=cfg.attn_dropout,
        )
        self.cross_attn_C_to_B = DirectedCrossAttention(
            embed_dim=D,
            num_heads=max(1, cfg.num_attn_heads // 2),
            stop_grad_on_keys=False,
            stop_grad_on_query_enrichment=True,   # closes C→B gradient path
            dropout=cfg.attn_dropout,
        )
        # Router (handles tokenization + 3-group routing + skip scheduler)
        self.router = ModalityMoERouter(
            moe_cfg,
            dim_A_in=cfg.dim_A_in,
            dim_B_in=cfg.dim_B_in,
            dim_C_in=cfg.dim_C_in,
        )

        # Expert pools
        self.experts = build_expert_pools(moe_cfg)

        # Decorrelation loss (one per block — separate buffer per block)
        self.decorr = ThreeGroupDecorrLoss(decorr_cfg,
                                           num_experts_B=cfg.num_experts_B)

        # Slice indices into the concatenated [A | C | B] sequence.
        # Layout: A first (processed earliest), then C (depends on A), then B.
        # This matches the attention mask layout in DirectedAttentionMask.
        self._sA = slice(0,        NA)
        self._sC = slice(NA,       NA + NC)
        self._sB = slice(NA + NC,  NA + NC + NB)

        # # warmup-cross attention
        # self.warm_layer = WarmUpCrossAttentionLayer(cfg.to_decorr_config())
        # 
        # intention heads
        self.intention_heads = IntentionHeads(embed_dim=D)

        # grouip B pipeline
        self.group_b_pipeline = GroupBInternalPipeline(
            embed_dim= D, 
            num_heads=cfg.num_attn_heads,
            history_len= getattr(cfg, 'history_len', 15),
            use_history_encoder= getattr(cfg, 'use_history_encoder', True),
        )

     
    def forward(
        self,
        tokens_A:      torch.Tensor,   # (B, N_A, D)
        tokens_B:      torch.Tensor,   # (B, N_B, D)
        tokens_C:      torch.Tensor,   # (B, N_C, D)
        spatial_xyz:   torch.Tensor,   # (B, N_A, 3) Gaussian mean coords
        token_types_C: torch.Tensor,   # (B, N_C) long — Group C token type IDs
        t:             torch.Tensor,   # (B,) diffusion timestep
        # intention heads attributes
        agent_types:    torch.Tensor, # (B, N_B) long - 0=veh, 1=ped, 2=cyc (this later will be motorcycist)
        
        # group b attribute
        ego_mask:       torch.Tensor,   # (B, N_B) boo;
        ego_distances:  torch.Tensor,   # (B, N_B) float
        ego_speed:      torch.Tensor,   # (B, ) float m/s
        history_traj:   Optional[torch.Tensor] = None,   # (B, N_B, H, 4)
        history_abs:    Optional[torch.Tensor] = None,  #(B, N_B, 2)
        gps_confidence: Optional[torch.Tensor] = None,  # (B, 2)
        # intention groudtruth
        intention_gt:   Optional[torch.Tensor] = None,  # (B, N_B) long or None
        step:          int = 0,        # global training step
        
    ) -> MoEBlockOutput:
        """One full MoE block forward pass.

        Args:
            tokens_A:      (B, N_A, D)  Group A token representations
            tokens_B:      (B, N_B, D)  Group B token representations
            tokens_C:      (B, N_C, D)  Group C token representations
            spatial_xyz:   (B, N_A, 3)  spatial coordinates for Group A gate
            token_types_C: (B, N_C)     token type IDs for Group C structural gate
            t:             (B,)         current diffusion timestep (int or float)
            step:          int          global training step (for decorr schedules)

        Returns:
            MoEBlockOutput with updated tokens and auxiliary losses.
        """

         
        # 1.  Pre-attention group-local LayerNorm  [FIX-2]
        #     Each group is normalized over its own statistics independently.
        #     No shared LN — see plan 1.3.2 warning.
         
        tokens_A_ln, tokens_B_ln, tokens_C_ln = self.pre_attn_ln(
            tokens_A, tokens_B, tokens_C
        )

         
        # 2.  Concatenate as [A | C | B] for directed self-attention.
        #     Order: A first (processed earliest), then C (depends on A),
        #     then B (depends on A+C). Matches DirectedAttentionMask layout.
         
        seq_ln = torch.cat([tokens_A_ln, tokens_C_ln, tokens_B_ln], dim=1)

         
        # 3.  Shared self-attention (cross-modal, directed mask).
        #     Returns attention output only — NO internal residual.  [FIX-1]
         
        attn_mask = self.router.attn_mask  # (N_tot, N_tot) additive
        attn_out  = self.self_attn(seq_ln, attn_mask)  # (B, N_tot, D)

         
        # 4.  Split and apply attention residuals.
        #     tokens_X = tokens_X + attn_out_X  (standard pre-norm residual)
        #     Residual is onto the original un-normed tokens — correct.  [FIX-1]
         
        tokens_A = tokens_A + attn_out[:, self._sA, :]
        tokens_C = tokens_C + attn_out[:, self._sC, :]
        tokens_B = tokens_B + attn_out[:, self._sB, :]

        # # STUB-B shall be here - no I was wrong
        # tokens_A, tokens_B, tokens_C = self.warm_layer(
        #     tokens_A, tokens_B, tokens_C
        # )

                 
        # 5.  Pre-expert group-local LayerNorm.
         
        tokens_A_ln2, tokens_B_ln2, tokens_C_ln2 = self.pre_exp_ln(
            tokens_A, tokens_B, tokens_C
        )
 
        # [STUB-A] TODO: directed cross-attention steps (3.2 steps 2-4).
        #   After shared self-attention, run:
        #     tokens_C = tokens_C + DirectedCrossAttention_A_to_C(
        #                    query=tokens_C_ln2, key=tokens_A_ln2,
        #                    stop_grad_on_keys=True)
        #     tokens_B = tokens_B + DirectedCrossAttention_A_to_B(
        #                    query=tokens_B_ln2, key=tokens_A_ln2,
        #                    stop_grad_on_keys=True)
        #     tokens_B = tokens_B + DirectedCrossAttention_C_to_B(
        #                    query=tokens_B_ln2, key=tokens_C_ln2,
        #                    stop_grad_on_query_enrichment=True)
        #   Each uses separate Q/K/V projection matrices (no weight sharing).
        # uses pre-expert normed tokens. A->C agent toknens read confirmed sensor velocity/crosswalk detections
        tokens_C = tokens_C + self.cross_attn_A_to_C(tokens_C_ln2, tokens_A_ln2)

        # A→B: agent tokens read confirmed sensor velocities / crosswalk detections.'
        tokens_B = tokens_B + self.cross_attn_A_to_B(tokens_B_ln2, tokens_A_ln2) # 
        # C→B: agent tokens read map context (stop-grad on output closes C grad path).
        tokens_B = tokens_B + self.cross_attn_C_to_B(tokens_B_ln2, tokens_C_ln2)

         
        # 5.  Pre-expert group-local LayerNorm.
         
        tokens_A_ln2, tokens_B_ln2, tokens_C_ln2 = self.pre_exp_ln(
            tokens_A, tokens_B, tokens_C
        )

        # router A + C, run experts pools
        router_out_ac = self.router(
            raw_A=tokens_A_ln2,
            raw_B=tokens_B_ln2,
            raw_C=tokens_C_ln2,
            spatial_xyz=spatial_xyz,
            token_types_C=token_types_C,
            t=t,
            output_A=None,   # not yet available
            output_C=None,   # not yet available
            step=step,
            # skip_tokenization=not self.is_first_block,  # [FIX-3]
            agent_types=agent_types,
        )
        # 7.  Run Group A and Group C expert pools, apply residuals.
         
        exp_out_A, acts_A = self.experts.pool_A(
            tokens_A_ln2,
            router_out_ac.dispatch_A,
            router_out_ac.combine_A,
            router_out_ac.skip_A,
            step,
        )
        exp_out_C, acts_C = self.experts.pool_C(
            tokens_C_ln2,
            router_out_ac.dispatch_C,
            router_out_ac.combine_C,
            router_out_ac.skip_C,
            step,
        )

        tokens_A = tokens_A + exp_out_A
        tokens_C = tokens_C + exp_out_C
         
        # [STUB-C] TODO: Group B internal pipeline (2.3-2.6).
        #   Before routing, enrich Group B tokens through three stages:
        #     tokens_B_ln2 = GroupBInternalPipeline(
        #                        tokens_B_ln2,
        #                        ego_token=tokens_B_ln2[:, ego_idx:ego_idx+1],
        #                        map_context=tokens_C,
        #                        ego_speed=ego_speed,
        #                        agent_density=agent_density)
         

         
        # 6a & 6b.  Routing pass 1 — Group A and Group C.
        #           Group B routing deferred until we have A/C expert outputs.
        #
        # [FIX-3]  skip_tokenization passed so group identity embeddings and
        #          projection layers do not re-run on blocks after the first.
        # [FIX-4]  route_B_only=False here (A+C pass); True in the B pass.
        tokens_B_ln2 = self.group_b_pipeline(
            tokens_B = tokens_B_ln2,
            ego_mask = ego_mask,
            map_context = exp_out_C,        # group C expert output
            ego_speed = ego_speed, 
            ego_distances = ego_distances, 
            history_traj = history_traj,
            history_abs = history_abs, 
            gps_confidence = gps_confidence,
            
        )        


         
        # 8.  Routing pass 2 — Group B only, cross-group conditioned.
        #     Now that A/C expert outputs are available, the Group B gate
        #     can attend to them for cross-group context.
        #
        # [FIX-4]  route_B_only=True — only Group B routing is recomputed.
        #          Group A/C routing weights from router_out_ac are valid
        #          and must not be recomputed (doubles A/C routing cost).
         
        router_out_b = self.router(
            raw_A=tokens_A_ln2,       # same pre-normed inputs as pass 1
            raw_B=tokens_B_ln2,
            raw_C=tokens_C_ln2,
            spatial_xyz=spatial_xyz,
            token_types_C=token_types_C,
            t=t,
            output_A=exp_out_A,   # Group A expert output (context for B gate)
            output_C=exp_out_C,   # Group C expert output (context for B gate)
            step=step,
            agent_types = agent_types
        )

         
        # 9.  Run Group B expert pool, apply residual.
         
        exp_out_B, acts_B = self.experts.pool_B(
            tokens_B_ln2,
            router_out_b.dispatch_B,
            router_out_b.combine_B,
            router_out_b.skip_B,
            step,
        )

        tokens_B = tokens_B + exp_out_B

         
        # [STUB-D] TODO: intention loss (2.7).
        #   intention_logits, intention_loss = IntentionHeads(
        #       agent_repr=tokens_B_after_stage2, agent_types=agent_types)
        #   Add intention_loss to MoEBlockOutput and to total_aux in
        #   StackedMoEBlocks.
        intention_for_gate , intention_loss = self.intention_heads(
            agent_repr = tokens_B_ln2,  # stage 2 normed B tokens
            agent_types = agent_types,
            intention_gt = intention_gt,
        )

         
        # 10.  Decorrelation loss — only fires when global training step
        #      and diffusion timestep t permit (schedule in DecorrConfig).
         
        decorr_loss, decorr_log = self.decorr(
            acts_A=acts_A,
            acts_B=acts_B,
            acts_C=acts_C,
            step=step,
            t=t,
        )

        # Prefix log keys with block_id for multi-block diagnostics.
        decorr_log = {f"block{self.block_id}/{k}": v
                      for k, v in decorr_log.items()}

        # Use A/C routing outputs for capacity + ortho losses.
        # (B routing is the same model; capacity_loss and ortho_loss are
        # identical in both router_out_ac and router_out_b — use ac.)
        # capacity_loss = router_out_ac.capacity_loss
        # use bias penalty replace
        bias_penalty  = router_out_ac.bias_penalty
        ortho_loss    = router_out_ac.ortho_loss

        return MoEBlockOutput(
            tokens_A=tokens_A,
            tokens_B=tokens_B,
            tokens_C=tokens_C,
            decorr_loss=decorr_loss,    # [FIX-6] field renamed
            bias_penalty = bias_penalty,
            # capacity_loss=capacity_loss,
            ortho_loss=ortho_loss,
            decorr_log=decorr_log,
            router_out=router_out_b,    # full routing decision (diagnostics)]
            total_intention=intention_loss,
            intention_loss=intention_loss
        )

    def invalidate_caches(self):
        """Reset A/C expert caches — call at scene boundary."""
        self.experts.invalidate_caches()

    def reset_decorr_buffer(self):
        """Reset Group B anchor buffer — call at stage transitions."""
        self.decorr.reset_B_buffer()

    def set_stopgrad_C_to_B(self, active: bool):
        """Toggle stop-gradient on C->B gate path (relax in late fine-tuning)."""
        self.router.cfg.stopgrad_C_to_B = active
        for module in self.router.modules():
            if hasattr(module, "stopgrad_C_to_B"):
                module.stopgrad_C_to_B = active


 
# Stacked MoE Blocks
 

@dataclass
class StackedMoEOutput:
    """Output from N stacked MoE blocks."""
    tokens_A:       torch.Tensor   # (B, N_A, D) final Group A
    tokens_B:       torch.Tensor   # (B, N_B, D) final Group B
    tokens_C:       torch.Tensor   # (B, N_C, D) final Group C
    total_decorr:   torch.Tensor   # scalar — summed decorr loss
    total_capacity: torch.Tensor   # scalar — summed capacity loss
    total_ortho:    torch.Tensor   # scalar — summed ortho loss
    aux_loss:       torch.Tensor   # scalar — total auxiliary loss
    log_dict:       Dict           # merged per-block log dicts
    last_router:    RouterOutput   # router output from the last block
    total_intention: torch.Tensor

class StackedMoEBlocks(nn.Module):
    """N stacked MoEBlocks forming the full MoE decoder backbone.

    Each block shares the same MoEBlockConfig (same number of experts, same
    skip thresholds).  Blocks are INDEPENDENT — separate parameters, separate
    expert pools, separate decorrelation buffers.

    [STUB-B] TODO: add a WarmUpCrossAttentionLayer BEFORE self.blocks (3.3).
      This is a lighter-weight cross-attention layer (fewer heads, no expert
      dispatch) that gives Group B's gate a cross-modal representation before
      ANY routing decision in the main block stack.  Without it, the first
      block's Group B router makes routing decisions with no prior cross-modal
      enrichment.  Estimated cost: ~3-5% FLOPs.

    Auxiliary loss aggregation:
        total_aux = sum_blocks(decorr_loss + capacity_loss + ortho_loss)

    Args:
        cfg:        MoEBlockConfig (shared across all blocks)
        num_blocks: number of stacked MoEBlocks
    """

    def __init__(self, cfg: MoEBlockConfig, num_blocks: int = 4):
        super().__init__()
        self.cfg        = cfg
        self.num_blocks = num_blocks

        # [STUB-B] TODO: self.warmup_layer = WarmUpCrossAttentionLayer(cfg)
        self.warm_layer = WarmUpCrossAttentionLayer(cfg.to_moe_config())

        # [FIX-3] Only the first block runs the tokenizer (project + group
        # identity embeddings). Subsequent blocks receive already-projected
        # tokens and must not re-apply the tokenizer.
        self.blocks = nn.ModuleList([
            MoEBlock(cfg, block_id=i, is_first_block=(i == 0))
            for i in range(num_blocks)
        ])  


    def forward(
        self,
        tokens_A:       torch.Tensor,                   # (B, N_A, D)
        tokens_B:       torch.Tensor,                   # (B, N_B, D)
        tokens_C:       torch.Tensor,                   # (B, N_C, D)
        spatial_xyz:    torch.Tensor,                   # (B, N_A, 3)
        token_types_C:  torch.Tensor,                   # (B, N_C)
        t:              torch.Tensor,                   # (B,)
        agent_types:    torch.Tensor,                   # (B, N_B) long  
        ego_mask:       torch.Tensor,                   # (B, N_B) bool  
        ego_distances:  torch.Tensor,                   # (B, N_B) float 
        ego_speed:      torch.Tensor,                   # (B,) float     
        history_traj:   Optional[torch.Tensor] = None,  # (B, N_B, H, 4)
        history_abs:    Optional[torch.Tensor] = None,  # (B, N_B, 2)   
        gps_confidence: Optional[torch.Tensor] = None,  # (B,)           
        intention_gt:   Optional[torch.Tensor] = None,  # (B, N_B) long  
        step:           int = 0,
    ) -> StackedMoEOutput:
        """Run all N MoE blocks in sequence, accumulating auxiliary losses."""
        device = tokens_A.device
        total_decorr   = torch.tensor(0.0, device=device)
        total_capacity = torch.tensor(0.0, device=device)
        total_ortho    = torch.tensor(0.0, device=device)
        # total intention loss
        total_intention = torch.tensor(0.0, device = device)
        merged_log:  Dict                    = {}
        last_router: Optional[RouterOutput]  = None

        # [STUB-B] TODO: run warmup layer before block loop:
        # tokens_A, tokens_B, tokens_C = self.warmup_layer(
        #     tokens_A, tokens_B, tokens_C, ...)
        tokens_A, tokens_B, tokens_C = self.warm_layer(
            tokens_A, tokens_B, tokens_C
        )
        for block in self.blocks:
            out: MoEBlockOutput = block(
                tokens_A=tokens_A,
                tokens_B=tokens_B,
                tokens_C=tokens_C,
                spatial_xyz=spatial_xyz,
                token_types_C=token_types_C,
                t=t,
                agent_types=agent_types,       
                ego_mask=ego_mask,             
                ego_distances=ego_distances,   
                ego_speed=ego_speed,           
                history_traj=history_traj,     
                history_abs=history_abs,       
                gps_confidence=gps_confidence, 
                intention_gt=intention_gt,     
                step=step,
            )
            tokens_A = out.tokens_A
            tokens_B = out.tokens_B
            tokens_C = out.tokens_C
            total_decorr   = total_decorr   + out.decorr_loss   # [FIX-6]
            total_capacity = total_capacity + out.bias_penalty
            total_ortho    = total_ortho    + out.ortho_loss
            total_intention = total_intention + out.intention_loss
            merged_log.update(out.decorr_log)
            last_router = out.router_out
            
        total_aux = total_decorr + total_capacity + total_ortho + total_intention

        return StackedMoEOutput(
            tokens_A=tokens_A,
            tokens_B=tokens_B,
            tokens_C=tokens_C,
            total_decorr=total_decorr,
            total_capacity=total_capacity,
            total_ortho=total_ortho,
            total_intention= total_intention,
            aux_loss=total_aux,
            log_dict=merged_log,
            last_router=last_router,
        )

    def invalidate_caches(self):
        for block in self.blocks:
            block.invalidate_caches()

    def reset_decorr_buffers(self):
        for block in self.blocks:
            block.reset_decorr_buffer()

    def set_stopgrad_C_to_B(self, active: bool):
        """Broadcast stop-gradient toggle to all blocks."""
        for block in self.blocks:
            block.set_stopgrad_C_to_B(active)

    def get_param_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """Return per-component parameter groups with LR scaling.

        Learning rate schedule from plan 6.4:
            router_A / router_C:       LR x 1.0
            router_B:                  LR x 1.0
            expert_A:                  LR x 0.3
            expert_C:                  LR x 0.3
            expert_B specialist:       LR x 0.5
            expert_B shared (e0):      LR x 0.05  (anchor, moves very slowly)
            shared_attn:               LR x 0.1   (protect cross-modal integration)
            tokenizer:                 LR x 1.0
            layernorms:                LR x 1.0
            decorr:                    LR x 1.0   (no params by default)

        TODO (6.4 — not yet implemented here):
            skip_predictors:           LR x 2.0   (stub-C / DyDiT predictors)
            expert_B_shared_cross_attn: LR x 0.3  (Stage 3 cross-attn when built)
            router_warmup stage:       LR x 3.0   (stage-aware, needs training loop)
            diffusion_decoder:         lower LR + KL regularizer
        """
        groups = []

        # Warmup layer
        groups.append({
            "params": list(self.warm_layer.parameters()),
            "lr": base_lr * 0.3,
            "name": "warmup_cross_attn",
        })
 
        for i, block in enumerate(self.blocks):
            groups.append({
                "params": list(block.self_attn.parameters()),
                "lr": base_lr * 0.1,
                "name": f"block{i}.shared_attn",
            })
            groups.append({
                "params": list(block.router.router_A.parameters()),
                "lr": base_lr * 1.0,
                "name": f"block{i}.router_A",
            })
            groups.append({
                "params": list(block.router.router_C.parameters()),
                "lr": base_lr * 1.0,
                "name": f"block{i}.router_C",
            })
            groups.append({
                "params": list(block.router.router_B.parameters()),
                "lr": base_lr * 1.0,
                "name": f"block{i}.router_B",
            })
            groups.append({
                "params": list(block.experts.pool_A.parameters()),
                "lr": base_lr * 0.3,
                "name": f"block{i}.experts_A",
            })
            groups.append({
                "params": list(block.experts.pool_C.parameters()),
                "lr": base_lr * 0.3,
                "name": f"block{i}.experts_C",
            })
            groups.append({
                "params": list(block.experts.pool_B.pool.experts[0].parameters()),
                "lr": base_lr * 0.05,
                "name": f"block{i}.expert_B_shared",
            })
            specialist_params = []
            for e_idx in range(1, self.cfg.num_experts_B):
                specialist_params.extend(
                    list(block.experts.pool_B.pool.experts[e_idx].parameters())
                )
            groups.append({
                "params": specialist_params,
                "lr": base_lr * 0.5,
                "name": f"block{i}.experts_B_specialist",
            })
            groups.append({
                "params": list(block.router.tokenizer.parameters()),
                "lr": base_lr * 1.0,
                "name": f"block{i}.tokenizer",
            })
            groups.append({
                "params": (list(block.pre_attn_ln.parameters()) +
                           list(block.pre_exp_ln.parameters())),
                "lr": base_lr * 1.0,
                "name": f"block{i}.layernorms",
            })
            groups.append({
                "params": (
                    list(block.group_b_pipeline.stage1.parameters()) +
                    list(block.group_b_pipeline.stage2.parameters()) +
                    list(block.group_b_pipeline.stage3.parameters()) +
                    list(block.group_b_pipeline.pre_norm.parameters()) +
                    list(block.group_b_pipeline.inter_norm.parameters())
                ),
                "lr": base_lr * 0.3,
                "name": f"block{i}.group_b_pipeline_attn",
            })
            if block.group_b_pipeline.use_history_encoder:
                groups.append({
                    "params": list(block.group_b_pipeline.history_encoder.parameters()),
                    "lr": base_lr * 1.0,
                    "name": f"block{i}.history_encoder",
                })
            groups.append({
                "params": list(block.intention_heads.parameters()),
                "lr": base_lr * 1.0,
                "name": f"block{i}.intention_heads",
            })
           
            groups.append({
                "params": (
                    list(block.cross_attn_A_to_C.parameters()) +
                    list(block.cross_attn_A_to_B.parameters()) +
                    list(block.cross_attn_C_to_B.parameters())
                ),
                "lr": base_lr * 0.3,
                "name": f"block{i}.directed_cross_attn",
            })
 
        return groups


 
# Factory
 

def build_moe_backbone(
    embed_dim:    int = 256,
    num_tokens_A: int = 64,
    num_tokens_B: int = 64,
    num_tokens_C: int = 128,
    num_blocks:   int = 4,
    num_experts:  int = 4,
    **kwargs,
) -> StackedMoEBlocks:
    cfg = MoEBlockConfig(
        embed_dim=embed_dim,
        num_tokens_A=num_tokens_A,
        num_tokens_B=num_tokens_B,
        num_tokens_C=num_tokens_C,
        num_experts_A=num_experts,
        num_experts_B=num_experts,
        num_experts_C=max(num_experts, 6),
        dim_A_in=embed_dim,
        dim_B_in=embed_dim,
        dim_C_in=embed_dim,
        **{k: v for k, v in kwargs.items() if hasattr(MoEBlockConfig, k)},
    )
    return StackedMoEBlocks(cfg, num_blocks=num_blocks)


 
if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
 
    #  Shared test dimensions ─
    B  = 2
    D  = 128
    NA = 32
    NB = 16
    NC = 32
    H  = 5      # history length (short for speed)
 
    cfg = MoEBlockConfig(
        embed_dim=D,
        num_tokens_A=NA, num_tokens_B=NB, num_tokens_C=NC,
        num_experts_A=4, num_experts_B=4, num_experts_C=6,
        num_attn_heads=4,
        expert_ff_mult=2,
        dim_A_in=D, dim_B_in=D, dim_C_in=D,
        # Fire decorrelation immediately so the loss path is exercised
        step_start_A=0, step_start_B=0, step_start_C=0,
        warmup_steps_A=10, warmup_steps_B=10, warmup_steps_C=10,
        T_max=1000,
        history_len=H,
        use_history_encoder=True,
        lambda_intention=0.1,
    )
 
    #  Reusable input factory ─
    def make_inputs(t_vals=(300, 500)):
        tokens_A = torch.randn(B, NA, D)
        tokens_B = torch.randn(B, NB, D)
        tokens_C = torch.randn(B, NC, D)
        xyz      = torch.randn(B, NA, 3) * 20.0
        ttypes   = torch.randint(0, 6, (B, NC))
        t        = torch.tensor(t_vals, dtype=torch.float)
        # Group B extras
        agent_types   = torch.randint(0, 3, (B, NB))       # 0=veh,1=ped,2=cyc
        ego_mask      = torch.zeros(B, NB, dtype=torch.bool)
        ego_mask[:, 0] = True                               # agent 0 is ego
        ego_distances = torch.rand(B, NB) * 50.0
        ego_distances[:, 0] = 0.0                           # ego→ego = 0
        ego_speed     = torch.tensor([8.0, 12.0])
        history_traj  = torch.randn(B, NB, H, 4)
        history_abs   = torch.randn(B, NB, 2) * 100.0
        gps_conf      = torch.tensor([0.9, 0.7])
        intention_gt = torch.zeros(B, NB, dtype=torch.long)
        for b in range(B):
            for n in range(NB):
                if agent_types[b, n] == 1:  # pedestrian → 2 classes
                    intention_gt[b, n] = torch.randint(0, 2, (1,)).item()
                else:                        # vehicle or cyclist → 6 classes
                    intention_gt[b, n] = torch.randint(0, 6, (1,)).item()

        return dict(
            tokens_A=tokens_A, tokens_B=tokens_B, tokens_C=tokens_C,
            spatial_xyz=xyz, token_types_C=ttypes, t=t,
            agent_types=agent_types, ego_mask=ego_mask,
            ego_distances=ego_distances, ego_speed=ego_speed,
            history_traj=history_traj, history_abs=history_abs,
            gps_confidence=gps_conf, intention_gt=intention_gt,
        )
 
    PASS = "PASS"
    FAIL = "FAIL"
 
    
    # 1. Single MoEBlock — shapes
    
    print("=" * 60)
    print("1. Single MoEBlock — output shapes")
    print("=" * 60)
 
    block = MoEBlock(cfg, block_id=0, is_first_block=True)
    inp   = make_inputs()
    out   = block(**inp, step=50)
 
    print(f"  tokens_A : {out.tokens_A.shape}  (expect ({B},{NA},{D}))")
    print(f"  tokens_B : {out.tokens_B.shape}  (expect ({B},{NB},{D}))")
    print(f"  tokens_C : {out.tokens_C.shape}  (expect ({B},{NC},{D}))")
    assert out.tokens_A.shape == (B, NA, D)
    assert out.tokens_B.shape == (B, NB, D)
    assert out.tokens_C.shape == (B, NC, D)
    print(f"  Shape assertions: {PASS}")
 
    
    # 2. Single MoEBlock — no NaN in outputs or losses
    
    print("\n2. Single MoEBlock — NaN check")
    print("=" * 60)
 
    for name, tensor in [
        ("tokens_A",      out.tokens_A),
        ("tokens_B",      out.tokens_B),
        ("tokens_C",      out.tokens_C),
        ("decorr_loss",   out.decorr_loss),
        ("bias_penalty",  out.bias_penalty),
        ("ortho_loss",    out.ortho_loss),
        ("intention_loss",out.intention_loss),
    ]:
        has_nan = torch.isnan(tensor).any().item()
        status  = FAIL if has_nan else PASS
        print(f"  {name:<20} NaN={has_nan}  [{status}]")
        assert not has_nan, f"NaN found in {name}"
 
    
    # 3. Intention loss — zero at inference, > 0 with GT
    
    print("\n3. Intention loss schedule")
    print("=" * 60)
 
    # No GT (inference)
    inp_no_gt = {**make_inputs(), "intention_gt": None}
    out_no_gt = block(**inp_no_gt, step=0)
    print(f"  intention_loss (no GT)  = {out_no_gt.intention_loss.item():.6f}  (expect 0.0)")
    assert out_no_gt.intention_loss.item() == 0.0, "Loss should be 0 without GT"
    print(f"  No-GT guard: {PASS}")
 
    # With GT
    print(f"  intention_loss (with GT)= {out.intention_loss.item():.6f}  (expect > 0)")
    assert out.intention_loss.item() > 0.0, "Loss should be > 0 with GT"
    print(f"  With-GT training loss:  {PASS}")
 
    
    # 4. Gradient flow — all three token groups
    
    print("\n4. Gradient flow")
    print("=" * 60)
 
    inp_grad  = make_inputs()
    tA_g = inp_grad["tokens_A"].requires_grad_(True)
    tB_g = inp_grad["tokens_B"].requires_grad_(True)
    tC_g = inp_grad["tokens_C"].requires_grad_(True)
    inp_grad["tokens_A"] = tA_g
    inp_grad["tokens_B"] = tB_g
    inp_grad["tokens_C"] = tC_g
 
    out_g = block(**inp_grad, step=50)
    total_loss = (
        out_g.tokens_A.sum() + out_g.tokens_B.sum() + out_g.tokens_C.sum()
        + out_g.decorr_loss + out_g.bias_penalty + out_g.ortho_loss
        + out_g.intention_loss
    )
    total_loss.backward()
 
    for name, tensor in [("tokens_A", tA_g), ("tokens_B", tB_g), ("tokens_C", tC_g)]:
        has_grad = tensor.grad is not None
        has_nan  = torch.isnan(tensor.grad).any().item() if has_grad else True
        status   = PASS if (has_grad and not has_nan) else FAIL
        print(f"  {name} grad: exists={has_grad}  NaN={has_nan}  [{status}]")
        assert has_grad,   f"No grad for {name}"
        assert not has_nan, f"NaN grad for {name}"
 
    
    # 5. is_first_block wiring (3-block backbone)
    
    print("\n5. is_first_block wiring")
    print("=" * 60)
 
    backbone = build_moe_backbone(
        embed_dim=D, num_tokens_A=NA, num_tokens_B=NB, num_tokens_C=NC,
        num_blocks=3, num_experts=4, num_attn_heads=4, expert_ff_mult=2,
        step_start_A=0, step_start_B=0, step_start_C=0,
        warmup_steps_A=10, warmup_steps_B=10, warmup_steps_C=10,
        history_len=H, use_history_encoder=True,
    )
    for i, blk in enumerate(backbone.blocks):
        expected = (i == 0)
        status   = PASS if blk.is_first_block == expected else FAIL
        print(f"  block[{i}].is_first_block = {blk.is_first_block}  "
              f"(expect {expected})  [{status}]")
        assert blk.is_first_block == expected
 
    
    # 6. StackedMoEBlocks forward — shapes and loss accumulation
    
    print("\n6. StackedMoEBlocks (3 blocks) — forward pass")
    print("=" * 60)
 
    inp_s = make_inputs()
    sout  = backbone(
        **{k: v for k, v in inp_s.items()
           if k not in ("tokens_A", "tokens_B", "tokens_C",
                        "spatial_xyz", "token_types_C", "t")},
        tokens_A=inp_s["tokens_A"], tokens_B=inp_s["tokens_B"],
        tokens_C=inp_s["tokens_C"], spatial_xyz=inp_s["spatial_xyz"],
        token_types_C=inp_s["token_types_C"], t=inp_s["t"],
        step=50,
    )
    print(f"  tokens_A      : {sout.tokens_A.shape}")
    print(f"  tokens_B      : {sout.tokens_B.shape}")
    print(f"  tokens_C      : {sout.tokens_C.shape}")
    print(f"  total_decorr  : {sout.total_decorr.item():.6f}")
    print(f"  total_capacity: {sout.total_capacity.item():.6f}")
    print(f"  total_ortho   : {sout.total_ortho.item():.6f}")
    print(f"  total_intention:{sout.total_intention.item():.6f}")
    print(f"  aux_loss      : {sout.aux_loss.item():.6f}")
    print(f"  log keys      : {len(sout.log_dict)}")
 
    assert sout.tokens_A.shape == (B, NA, D)
    assert sout.tokens_B.shape == (B, NB, D)
    assert sout.tokens_C.shape == (B, NC, D)
 
    # aux_loss must equal the sum of all four components
    expected_aux = (sout.total_decorr + sout.total_capacity
                    + sout.total_ortho + sout.total_intention)
    assert torch.allclose(sout.aux_loss, expected_aux), \
        f"aux_loss mismatch: {sout.aux_loss.item()} vs {expected_aux.item()}"
    print(f"  aux_loss = decorr+capacity+ortho+intention: {PASS}")
 
    # total_intention must be non-negative and included
    assert sout.total_intention.item() >= 0.0
    print(f"  total_intention in aux_loss: {PASS}")
 
    
    # 7. Directed attention mask — directionality
    
    print("\n7. Directed attention mask")
    print("=" * 60)
 
    mask = backbone.blocks[0].router.attn_mask
    sA   = slice(0,  NA)
    sC   = slice(NA, NA + NC)
    sB   = slice(NA + NC, NA + NC + NB)
 
    checks = [
        ("A must NOT attend to C", mask[sA, sC], float("-inf"), True),
        ("A must NOT attend to B", mask[sA, sB], float("-inf"), True),
        ("C must NOT attend to B", mask[sC, sB], float("-inf"), True),
        ("B must attend to A",     mask[sB, sA], 0.0,          False),
        ("B must attend to C",     mask[sB, sC], 0.0,          False),
        ("A self-attention OK",    mask[sA, sA], 0.0,          False),
        ("C self-attention OK",    mask[sC, sC], 0.0,          False),
        ("B self-attention OK",    mask[sB, sB], 0.0,          False),
    ]
    for desc, region, expected_val, expect_neginf in checks:
        if expect_neginf:
            ok = torch.all(region == float("-inf")).item()
        else:
            ok = torch.all(region == expected_val).item()
        print(f"  {desc}: [{PASS if ok else FAIL}]")
        assert ok, f"Mask check failed: {desc}"
 
    
    # 8. Capacity factors (FIX-5)
    
    print("\n8. Capacity factor assertions")
    print("=" * 60)
 
    moe_cfg = cfg.to_moe_config()
    assert moe_cfg.capacity_factor_A == 1.5, "capacity_factor_A mismatch"
    assert moe_cfg.capacity_factor_C == 2.0, "capacity_factor_C must be 2.0"
    assert moe_cfg.capacity_factor_B == 1.5, "capacity_factor_B mismatch"
    print(f"  capacity_factor_A = {moe_cfg.capacity_factor_A}  [PASS]")
    print(f"  capacity_factor_B = {moe_cfg.capacity_factor_B}  [PASS]")
    print(f"  capacity_factor_C = {moe_cfg.capacity_factor_C}  [PASS]")
 
    
    # 9. Parameter groups — coverage, LR values, no .self attribute error
    
    print("\n9. Parameter groups")
    print("=" * 60)
 
    param_groups = backbone.get_param_groups(base_lr=1e-4)
    lr_map       = {g["name"]: g["lr"] for g in param_groups}
 
    # LR spot checks
    lr_checks = [
        ("warmup_cross_attn",          1e-4 * 0.3),
        ("block0.shared_attn",         1e-4 * 0.1),
        ("block0.router_A",            1e-4 * 1.0),
        ("block0.router_B",            1e-4 * 1.0),
        ("block0.experts_A",           1e-4 * 0.3),
        ("block0.expert_B_shared",     1e-4 * 0.05),
        ("block0.experts_B_specialist",1e-4 * 0.5),
        ("block0.intention_heads",     1e-4 * 1.0),
        ("block0.directed_cross_attn", 1e-4 * 0.3),
        ("block0.group_b_pipeline_attn", 1e-4 * 0.3),
        ("block0.history_encoder",     1e-4 * 1.0),
    ]
    for name, expected_lr in lr_checks:
        actual = lr_map.get(name)
        ok     = actual is not None and abs(actual - expected_lr) < 1e-10
        print(f"  {name:<35} lr={actual}  [{PASS if ok else FAIL}]")
        assert ok, f"LR mismatch for '{name}': got {actual}, expected {expected_lr}"
 
    # No parameter should be listed in more than one group
    all_param_ids = []
    for g in param_groups:
        for p in g["params"]:
            all_param_ids.append(id(p))
    duplicates = len(all_param_ids) - len(set(all_param_ids))
    print(f"\n  Total param groups : {len(param_groups)}")
    print(f"  Total params listed: {len(all_param_ids)}")
    print(f"  Duplicate entries  : {duplicates}")
    assert duplicates == 0, f"{duplicates} parameters appear in multiple groups"
    print(f"  No duplicate params: {PASS}")
 
    # Coverage: params in groups vs total model params
    grouped_count = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
    total_count   = sum(p.numel() for p in backbone.parameters())
    coverage_pct  = 100.0 * grouped_count / max(total_count, 1)
    print(f"  Params via groups  : {grouped_count:>10,}")
    print(f"  Total model params : {total_count:>10,}")
    print(f"  Coverage           : {coverage_pct:.1f}%")
 
    
    # 10. DyDiT skip logic — B at high-t, C at low-t
    
    print("\n10. DyDiT skip logic")
    print("=" * 60)
 
    inp_hi = make_inputs(t_vals=(900, 950))
    out_hi = block(**inp_hi, step=50)
    skip_B = out_hi.router_out.skip_B
    print(f"  t=(900,950)  skip_B={skip_B.tolist()}  (expect all True)")
    assert skip_B.all(), "Group B should skip at high-t"
    print(f"  Group B skips at high-t: {PASS}")
 
    inp_lo = make_inputs(t_vals=(50, 100))
    out_lo = block(**inp_lo, step=50)
    skip_C = out_lo.router_out.skip_C
    print(f"  t=(50,100)   skip_C={skip_C.tolist()}  (expect all True)")
    assert skip_C.all(), "Group C should skip at low-t"
    print(f"  Group C skips at low-t:  {PASS}")
 
    
    # 11. Warmup layer runs exactly once, before blocks
    
    print("\n11. Warmup layer runs once before block loop")
    print("=" * 60)
 
    call_log = []
    original_warmup = backbone.warm_layer.forward
    def _patched_warmup(tA, tB, tC):
        call_log.append("called")
        return original_warmup(tA, tB, tC)
    backbone.warm_layer.forward = _patched_warmup
 
    inp_w = make_inputs()
    backbone(
        tokens_A=inp_w["tokens_A"], tokens_B=inp_w["tokens_B"],
        tokens_C=inp_w["tokens_C"], spatial_xyz=inp_w["spatial_xyz"],
        token_types_C=inp_w["token_types_C"], t=inp_w["t"],
        agent_types=inp_w["agent_types"], ego_mask=inp_w["ego_mask"],
        ego_distances=inp_w["ego_distances"], ego_speed=inp_w["ego_speed"],
        history_traj=inp_w["history_traj"], history_abs=inp_w["history_abs"],
        gps_confidence=inp_w["gps_confidence"], intention_gt=inp_w["intention_gt"],
        step=0,
    )
    backbone.warm_layer.forward = original_warmup  # restore
 
    print(f"  Warmup called {len(call_log)} time(s)  (expect 1)")
    assert len(call_log) == 1, f"Warmup should run exactly once, ran {len(call_log)}"
    print(f"  Warmup runs exactly once: {PASS}")
 
    
    # Done
    
    print("\n" + "=" * 60)
    print("All moe_block.py __main__ checks PASSED.")
    print("=" * 60)
 