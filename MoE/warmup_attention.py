
"""
STUB-B implementation: WarmUpCrossAttentionLayer
 
Lightweight cross-attention layer that runs ONCE before the main MoEBlock
stack in StackedMoEBlocks.  Gives every downstream router a cross-modally
enriched token representation before the first routing decision fires.
 
Motivation (§3.3):
    Without this layer, the first block's Group B router gates on tokens that
    have never seen any map or sensor context.  That forces the router to make
    a blind routing decision on the very first forward pass, which:
      - increases early training instability (random expert assignment),
      - means the Group B buffer (decorrelation) fills with uninformed reps,
      - wastes the first block's shared self-attention doing work this layer
        can do more cheaply (fewer heads, no expert dispatch).
 
    Estimated cost: ~3-5% FLOPs relative to one full MoEBlock.
 
Design:
    Three directed cross-attention passes in A→C→B order:
        1. A→C  (map tokens read confirmed sensor evidence)
        2. A→B  (agent tokens read confirmed sensor velocities)
        3. C→B  (agent tokens read map context after A→C enrichment)
 
    Stop-gradients:
        A is detached as keys for A→C and A→B.
        C is detached as keys for C→B (after being enriched by A→C).
        This matches the directed-graph constraint: A and C must not be
        shaped by routing gradients flowing back from B.
 
    Pre-norm convention (matches GaussianRefinementBlock and MoEBlock):
        x = x + CrossAttn(GroupLocalLayerNorm(x), GroupLocalLayerNorm(src))
        GroupLocalLayerNorm is applied per-group, not shared.
 
    Fewer heads than main stack (warmup_num_heads from MoEConfig, default 2).
    No expert dispatch — pure attention + residual.
 
Differences from token_router.WarmUpCrossAttention:
    - Pre-norm (LN before attn) instead of post-norm (LN after residual).
    - GroupLocalLayerNorm (per-group) instead of shared LayerNorm per direction.
    - Single shared GroupLocalLayerNorm module (not three separate LNs).
    - Takes MoEBlockConfig directly (no need to unpack).
    - Cleaner separation: this file owns the full pre-stack layer;
      token_router.WarmUpCrossAttention can be deprecated once this is wired.
 
Integration in StackedMoEBlocks:
 
    __init__:
        from MoE.warmup_cross_attention import WarmUpCrossAttentionLayer
        self.warmup_layer = WarmUpCrossAttentionLayer(cfg)
 
    forward (before the block loop):
        tokens_A, tokens_B, tokens_C = self.warmup_layer(
            tokens_A, tokens_B, tokens_C
        )
 
    get_param_groups:
        groups.append({
            "params": list(self.warmup_layer.parameters()),
            "lr": base_lr * 0.3,   # same rate as directed_cross_attn in blocks
            "name": "warmup_cross_attn",
        })
"""
 
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from MoE.token_router import MoEConfig, GroupLocalLayerNorm

except ImportError:
    # fallback for standaline testing
    from token_router import MoEConfig, GroupLocalLayerNorm  # type: ignore

class _SingleDirectionCrossAttn(nn.Module):
    """
    One directed cross attention pass (internal helper)

    Lightweight version: no bias, no dropout, xavier init with small out_proj
    Pre-norm is applied by the parent (WarmupCrossAttentionLayer) before calling.

    Args:
        embed_dim: 0
        num_heads: H (typically 2, warmup_num_heads)
    
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # per-direction projections - no weight sharing
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)


        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)


        # small gain: warm layer starts as a weak perturbation
        # main block stack's shared self-attention does the heavy lifting early.
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
    
    def forward(
            self,
            query: torch.Tensor, # (B, N_q, D) -- already normed by caller
            key:   torch.Tensor, # (B, N_q, D) -- already normed, detached if needed
    )-> torch.Tensor:
        """Returns (B, N_q, D) -- cross-attn otuput only, NO residual"""
        B, N_q, D = query.shape
        N_k = key.shape[1]
        H, hd = self.num_heads, self.head_dim

        Q = self.q_proj(query).reshape(B, N_q, H, hd).transpose(1, 2) # (B, H, N_q, hd)
        K = self.k_proj(key).reshape(B, N_k, H, hd).transpose(1, 2)   # (B, H, N_k, hd)
        V = self.v_proj(key).reshape(B, N_k, H, hd).transpose(1, 2)   # (B, H, N_k, hd)


        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale #(B, H, N_q, N_k)
        weights = F.softmax(attn, dim = -1)
        out = torch.matmul(weights, V)                          # (B, H, N_q, hd)
        out = out.transpose(1, 2).reshape(B, N_q, D )

        return self.out_proj(out)   # (B, N_q, D)
    
class WarmUpCrossAttentionLayer(nn.Module):
    """
    Lightweight cross-attention layer run once before the MoEBlock stack.
 
    Enriches all three token groups with cross-modal context before any
    routing decision fires.  Runs in A→C→B order to respect the directed
    information-flow constraint.
 
    Pass order and stop-gradient semantics:
        Pass 1 — A→C:
            map tokens (C) attend to sensor tokens (A, detached).
            C gains confirmed sensor evidence before routing.
            A receives no gradient from this path.
 
        Pass 2 — A→B:
            agent tokens (B) attend to sensor tokens (A, detached).
            B gains velocity/crosswalk confirmation before routing.
            A receives no gradient from this path.
 
        Pass 3 — C→B:
            agent tokens (B) attend to map tokens (C, detached).
            C is already A-enriched from Pass 1 (residual applied before detach).
            B gains A-enriched map context.
            C receives no gradient from this path.
 
    Pre-norm per-group (GroupLocalLayerNorm):
        Each group is normalised over its own statistics independently.
        A single GroupLocalLayerNorm module is called once before all three
        passes and once more before Pass 3 to re-norm the A-enriched C tokens.
 
    Args:
        cfg: MoEConfig (uses embed_dim and warmup_num_heads)
    
    """

    def __init__(
            self,
            cfg: MoEConfig
    ):
        super().__init__()
        D = cfg.embed_dim
        nh = cfg.warmup_num_heads # default 2
        
        #grouplocallayernorm: one per normalization point.
        # ln_pre: before passes 1 and 2 (normalizes all three groups)
        # ln_C_re: re-norms C before pass 3 (C has changed from pass 1 residual)

        self.ln_pre = GroupLocalLayerNorm(D)
        self.ln_C_re = nn.LayerNorm(D) # sinlge group LN for C re-norm before pass 3

        # three independent cross-attention modules - one per direction.
        self.attn_A_to_C = _SingleDirectionCrossAttn(D, nh)
        self.attn_A_to_B = _SingleDirectionCrossAttn(D, nh)
        self.attn_C_to_B = _SingleDirectionCrossAttn(D, nh)

    def forward(
            self, 
            tokens_A: torch.Tensor, # (B, N_A, D)
            tokens_B: torch.Tensor, # (B, N_B, D)
            tokens_C: torch.Tensor, # (B, N_C, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one warm-up cross-attention pass over all three groups.
 
        Args:
            tokens_A: (B, N_A, D) — Group A (raw sensory)
            tokens_B: (B, N_B, D) — Group B (interaction / agents)
            tokens_C: (B, N_C, D) — Group C (map / context)
 
        Returns:
            tokens_A: (B, N_A, D) — UNCHANGED (no upward feedback to A)
            tokens_B: (B, N_B, D) — enriched with A and C context
            tokens_C: (B, N_C, D) — enriched with A context
        """

        # Pre-norm: normalise each group over its own statistics.
        # This is applied to ALL three groups at once.
        # A_ln is only used to form the key for A→C and A→B.
        A_ln, B_ln, C_ln = self.ln_pre(tokens_A, tokens_B, tokens_C)

        # pass 1 - A -> C
        # map tokens attend to  (detached) sensor tokens.
        # stop-graident: A must not to be shaped by C routing gradient.

        A_det = A_ln.detach()
        tokens_C = tokens_C + self.attn_A_to_C(C_ln, A_det) # (B, N_C, D)

        # pass 2 A->B
        # agent token atttent to (detachded) sensor tokens
        # stop-gradient: same reason as above.
        tokens_B = tokens_B + self.attn_A_to_B(B_ln, A_det) # (B, N_B, D)

        # pass 3 C->B
        # agent tokens attent to map tokens that are now A-enriched.
        # re-norm C before using it as keys (it has changed from pass 1).
        # Detach C so B routing gradient cannot flow back into C.
        C_enriched_ln = self.ln_C_re(tokens_C)
        C_det = C_enriched_ln.detach()
        tokens_B = tokens_B + self.attn_C_to_B( B_ln, C_det)

        # Group A is returned UNCHANGED.
        # It must not receive any gradient from the warm-up layer.
        # It already serves as a stable sensory anchor.
        return tokens_A, tokens_B, tokens_C