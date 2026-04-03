"""
DDPM/moe/moe_block.py  — corrected integration of all four stubs.

Bugs fixed in this pass (relative to the v2 upload):
  [BUG-1/10] NameError: tokens_*_ln2 used before pre_exp_ln (step 5).
             Fix: move directed cross-attention calls to AFTER step 5.
  [BUG-2]   group_b_pipeline called with exp_out_C before pool_C ran.
             Fix: pipeline runs after pool_C produces exp_out_C.
  [BUG-3]   warmup layer created twice under different names; second call
             passed DecorrConfig instead of MoEConfig.
             Fix: single self.warmup_layer; cfg.to_moe_config() only.
  [BUG-4]   block.cross_attn_A_to_B.self.parameters() — ".self" is invalid.
             Fix: block.cross_attn_A_to_B.parameters().
  [BUG-5]   ego_speed missing from MoEBlock.forward signature.
             Fix: added as required parameter.
  [BUG-6]   StackedMoEBlocks.forward didn't thread the new parameters
             (agent_types, ego_mask, ego_distances, etc.) through to block().
             Fix: forward accepts and passes all new args.
  [BUG-7]   router_out_ac.capacity_loss doesn't exist on RouterOutput.
             RouterOutput has bias_penalty. capacity_loss here should be
             cfg.capacity_penalty_coeff * bias_penalty.
             Fix: derive capacity_loss from bias_penalty.
  [BUG-8]   Stale TODO comment claiming intention_loss not yet added,
             while it was already in the dataclass.
             Fix: removed stale comment.
  [BUG-9]   Sanity check called block() with old 6-arg signature.
             Fix: updated to pass all new required args.
  [BUG-10]  (Same root as BUG-1 — confirmed resolved by same fix.)
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
from MoE.decorrelation_loss import DecorrConfig, ThreeGroupDecorrLoss, build_decorr_loss
from MoE.direct_attention import DirectedCrossAttention
from MoE.warmup_attention import WarmUpCrossAttentionLayer
from MoE.intention_heads  import IntentionHeads
from MoE.groupb_pipeline  import GroupBInternalPipeline
 


# ---------------------------------------------------------------------------
# Merged block config
# ---------------------------------------------------------------------------

@dataclass
class MoEBlockConfig:
    """All hyperparameters for one MoEBlock (and stacks of them)."""

    # Token / feature dimensions
    embed_dim:      int = 256
    num_tokens_A:   int = 64
    num_tokens_B:   int = 64
    num_tokens_C:   int = 128

    # Expert configuration
    num_experts_A:  int = 4
    num_experts_B:  int = 4
    num_experts_C:  int = 6
    top_k_A:        int = 2
    top_k_B:        int = 2
    top_k_C:        int = 1
    expert_ff_mult: int = 4
    expert_dropout: float = 0.0

    # Shared self-attention
    num_attn_heads: int = 8
    attn_dropout:   float = 0.0

    # Routing hyperparameters
    capacity_factor_A:      float = 1.5
    capacity_factor_B:      float = 1.5
    capacity_factor_C:      float = 2.0
    routing_floor:          float = 0.05
    ortho_reg_weight:       float = 1e-3
    stopgrad_C_to_B:        bool  = True
    capacity_penalty_coeff: float = 0.01
    struct_router_temp:     float = 0.1
    gate_num_heads:         int   = 4

    # DyDiT skip thresholds
    t_skip_C:          float = 0.2
    t_skip_B:          float = 0.7
    cache_interval_A:  int   = 5
    T_max:             int   = 1000

    # Input dimensions for tokeniser projections
    dim_A_in: int = 256
    dim_B_in: int = 256
    dim_C_in: int = 256

    # Decorrelation
    step_start_A:   int   = 2_000
    step_start_B:   int   = 6_000
    step_start_C:   int   = 10_000
    warmup_steps_A: int   = 2_000
    warmup_steps_B: int   = 3_000
    warmup_steps_C: int   = 2_000
    lambda_A:       float = 0.02
    lambda_B:       float = 0.03
    lambda_C:       float = 0.02
    t_decorr_B_threshold: float = 0.3
    buffer_size:    int   = 512
    min_tokens_to_update: int = 8
    sigma_C:        float = 1.0
    sigma_B:        float = 0.5
    min_expert_tokens: int = 4

    # Intention heads
    lambda_intention: float = 0.1

    # Group B pipeline
    num_attn_heads_pipeline: int  = 4
    history_len:             int  = 15
    use_history_encoder:     bool = True

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
            routing_floor=self.routing_floor,
            t_skip_C=self.t_skip_C,
            t_skip_B=self.t_skip_B,
            cache_interval_A=self.cache_interval_A,
            ortho_reg_weight=self.ortho_reg_weight,
            stopgrad_C_to_B=self.stopgrad_C_to_B,
            T_max=self.T_max,
            capacity_penalty_coeff=self.capacity_penalty_coeff,
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


# ---------------------------------------------------------------------------
# Shared self-attention with directed mask
# ---------------------------------------------------------------------------

class SharedSelfAttention(nn.Module):
    """Multi-head self-attention over the full [A | C | B] sequence.

    Pre-norm is applied by the caller (MoEBlock) before this call.
    Returns attention output only — no residual, no internal LayerNorm.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N_total, D) pre-normed.  Returns (B, N_total, D)."""
        attn_out, _ = self.attn(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return attn_out


# ---------------------------------------------------------------------------
# MoEBlock output
# ---------------------------------------------------------------------------

@dataclass
class MoEBlockOutput:
    """All outputs from a single MoEBlock forward pass."""
    tokens_A:      torch.Tensor
    tokens_B:      torch.Tensor
    tokens_C:      torch.Tensor
    decorr_loss:   torch.Tensor
    capacity_loss: torch.Tensor
    ortho_loss:    torch.Tensor
    intention_loss: torch.Tensor   # [BUG-8] stale TODO removed — field is live
    decorr_log:    Dict
    router_out:    RouterOutput


# ---------------------------------------------------------------------------
# MoE Block
# ---------------------------------------------------------------------------

class MoEBlock(nn.Module):
    """Single MoE Transformer Block.

    Forward pass order (corrected from v2):
        1.  Pre-attn group-local LN
        2.  Concat [A | C | B], shared self-attention, split + residuals
        3.  Pre-expert group-local LN  ← tokens_*_ln2 defined HERE
        4.  Directed cross-attention (A→C, A→B, C→B) using ln2 tensors  [BUG-1]
        5.  Route A + C, run A/C expert pools, apply residuals
        6.  Group B pipeline (needs exp_out_C from step 5)              [BUG-2]
        7.  Route B (cross-group conditioned on exp_out_A, exp_out_C)
        8.  Run B expert pool, apply residual
        9.  Intention heads on enriched B tokens
       10.  Decorrelation loss
    """

    def __init__(self, cfg: MoEBlockConfig, block_id: int = 0,
                 is_first_block: bool = True):
        super().__init__()
        self.cfg            = cfg
        self.block_id       = block_id
        self.is_first_block = is_first_block
        D  = cfg.embed_dim
        NA = cfg.num_tokens_A
        NB = cfg.num_tokens_B
        NC = cfg.num_tokens_C

        moe_cfg    = cfg.to_moe_config()
        decorr_cfg = cfg.to_decorr_config()

        # Normalisation
        self.pre_attn_ln = GroupLocalLayerNorm(D)
        self.pre_exp_ln  = GroupLocalLayerNorm(D)

        # Shared self-attention
        self.self_attn = SharedSelfAttention(D, cfg.num_attn_heads, cfg.attn_dropout)

        # Directed cross-attention — three independent instances
        cross_heads = max(1, cfg.num_attn_heads // 2)
        self.cross_attn_A_to_C = DirectedCrossAttention(
            embed_dim=D, num_heads=cross_heads,
            stop_grad_on_keys=True, stop_grad_on_query_enrichment=False,
            dropout=cfg.attn_dropout,
        )
        self.cross_attn_A_to_B = DirectedCrossAttention(
            embed_dim=D, num_heads=cross_heads,
            stop_grad_on_keys=True, stop_grad_on_query_enrichment=False,
            dropout=cfg.attn_dropout,
        )
        self.cross_attn_C_to_B = DirectedCrossAttention(
            embed_dim=D, num_heads=cross_heads,
            stop_grad_on_keys=False, stop_grad_on_query_enrichment=True,
            dropout=cfg.attn_dropout,
        )

        # Router
        self.router = ModalityMoERouter(
            moe_cfg,
            dim_A_in=cfg.dim_A_in,
            dim_B_in=cfg.dim_B_in,
            dim_C_in=cfg.dim_C_in,
        )

        # Expert pools
        self.experts = build_expert_pools(moe_cfg)

        # Decorrelation
        self.decorr = ThreeGroupDecorrLoss(decorr_cfg, num_experts_B=cfg.num_experts_B)

        # Group B pipeline
        self.group_b_pipeline = GroupBInternalPipeline(
            embed_dim=D,
            num_heads=cfg.num_attn_heads_pipeline,
            history_len=cfg.history_len,
            use_history_encoder=cfg.use_history_encoder,
        )

        # Intention heads
        self.intention_heads = IntentionHeads(embed_dim=D)

        # Sequence slices: layout is [A | C | B]
        self._sA = slice(0,       NA)
        self._sC = slice(NA,      NA + NC)
        self._sB = slice(NA + NC, NA + NC + NB)

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens_A:       torch.Tensor,                   # (B, N_A, D)
        tokens_B:       torch.Tensor,                   # (B, N_B, D)
        tokens_C:       torch.Tensor,                   # (B, N_C, D)
        spatial_xyz:    torch.Tensor,                   # (B, N_A, 3)
        token_types_C:  torch.Tensor,                   # (B, N_C) long
        t:              torch.Tensor,                   # (B,)
        agent_types:    torch.Tensor,                   # (B, N_B) long 0=veh,1=ped,2=cyc
        ego_mask:       torch.Tensor,                   # (B, N_B) bool
        ego_distances:  torch.Tensor,                   # (B, N_B) float metres
        ego_speed:      torch.Tensor,                   # (B,) float m/s  [BUG-5]
        history_traj:   Optional[torch.Tensor] = None,  # (B, N_B, H, 4)
        history_abs:    Optional[torch.Tensor] = None,  # (B, N_B, 2)
        gps_confidence: Optional[torch.Tensor] = None,  # (B,)
        intention_gt:   Optional[torch.Tensor] = None,  # (B, N_B) long
        step:           int = 0,
    ) -> MoEBlockOutput:

        # ── 1. Pre-attention group-local LN ──────────────────────────────
        tokens_A_ln, tokens_B_ln, tokens_C_ln = self.pre_attn_ln(
            tokens_A, tokens_B, tokens_C
        )

        # ── 2. Shared self-attention ──────────────────────────────────────
        seq_ln    = torch.cat([tokens_A_ln, tokens_C_ln, tokens_B_ln], dim=1)
        attn_mask = self.router.attn_mask
        attn_out  = self.self_attn(seq_ln, attn_mask)

        tokens_A = tokens_A + attn_out[:, self._sA, :]
        tokens_C = tokens_C + attn_out[:, self._sC, :]
        tokens_B = tokens_B + attn_out[:, self._sB, :]

        # ── 3. Pre-expert group-local LN ─────────────────────────────────
        # [BUG-1] tokens_*_ln2 must be defined BEFORE any code that uses them.
        tokens_A_ln2, tokens_B_ln2, tokens_C_ln2 = self.pre_exp_ln(
            tokens_A, tokens_B, tokens_C
        )

        # ── 4. Directed cross-attention (§3.2) ────────────────────────────
        # All three paths use the ln2 normed tensors.
        # A→C: map tokens read confirmed sensor evidence.
        tokens_C = tokens_C + self.cross_attn_A_to_C(tokens_C_ln2, tokens_A_ln2)
        # A→B: agent tokens read sensor velocities / crosswalk detections.
        tokens_B = tokens_B + self.cross_attn_A_to_B(tokens_B_ln2, tokens_A_ln2)
        # C→B: agent tokens read map context; output detached (stop-grad).
        tokens_B = tokens_B + self.cross_attn_C_to_B(tokens_B_ln2, tokens_C_ln2)

        # Re-norm after cross-attention residuals before routing.
        tokens_A_ln2, tokens_B_ln2, tokens_C_ln2 = self.pre_exp_ln(
            tokens_A, tokens_B, tokens_C
        )

        # ── 5. Route A + C, run expert pools ─────────────────────────────
        router_out_ac = self.router(
            raw_A=tokens_A_ln2,
            raw_B=tokens_B_ln2,
            raw_C=tokens_C_ln2,
            spatial_xyz=spatial_xyz,
            token_types_C=token_types_C,
            t=t,
            output_A=None,
            output_C=None,
            step=step,
            skip_tokenization=not self.is_first_block,
            route_B_only=False,
        )

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

        # ── 6. Group B internal pipeline ─────────────────────────────────
        # [BUG-2] exp_out_C is now available — pipeline receives it as map_context.
        tokens_B_ln2 = self.group_b_pipeline(
            tokens_B=tokens_B_ln2,
            ego_mask=ego_mask,
            map_context=exp_out_C,
            ego_speed=ego_speed,
            ego_distances=ego_distances,
            history_traj=history_traj,
            history_abs=history_abs,
            gps_confidence=gps_confidence,
        )

        # ── 7. Route B (cross-group conditioned) ─────────────────────────
        router_out_b = self.router(
            raw_A=tokens_A_ln2,
            raw_B=tokens_B_ln2,
            raw_C=tokens_C_ln2,
            spatial_xyz=spatial_xyz,
            token_types_C=token_types_C,
            t=t,
            output_A=exp_out_A,
            output_C=exp_out_C,
            step=step,
            skip_tokenization=True,
            route_B_only=True,
        )

        # ── 8. Run B expert pool ──────────────────────────────────────────
        exp_out_B, acts_B = self.experts.pool_B(
            tokens_B_ln2,
            router_out_b.dispatch_B,
            router_out_b.combine_B,
            router_out_b.skip_B,
            step,
        )
        tokens_B = tokens_B + exp_out_B

        # ── 9. Intention heads ────────────────────────────────────────────
        intention_for_gate, intention_loss = self.intention_heads(
            agent_repr=tokens_B_ln2,
            agent_types=agent_types,
            intention_gt=intention_gt,
        )

        # ── 10. Decorrelation loss ────────────────────────────────────────
        decorr_loss, decorr_log = self.decorr(
            acts_A=acts_A,
            acts_B=acts_B,
            acts_C=acts_C,
            step=step,
            t=t,
        )
        decorr_log = {f"block{self.block_id}/{k}": v for k, v in decorr_log.items()}

        # [BUG-7] RouterOutput has bias_penalty, not capacity_loss.
        # Derive capacity_loss from the bias_penalty scalar.
        capacity_loss = self.cfg.capacity_penalty_coeff * router_out_ac.bias_penalty
        ortho_loss    = router_out_ac.ortho_loss

        return MoEBlockOutput(
            tokens_A=tokens_A,
            tokens_B=tokens_B,
            tokens_C=tokens_C,
            decorr_loss=decorr_loss,
            capacity_loss=capacity_loss,
            ortho_loss=ortho_loss,
            intention_loss=self.cfg.lambda_intention * intention_loss,
            decorr_log=decorr_log,
            router_out=router_out_b,
        )

    def invalidate_caches(self):
        self.experts.invalidate_caches()

    def reset_decorr_buffer(self):
        self.decorr.reset_B_buffer()

    def set_stopgrad_C_to_B(self, active: bool):
        self.router.cfg.stopgrad_C_to_B = active
        for module in self.router.modules():
            if hasattr(module, "stopgrad_C_to_B"):
                module.stopgrad_C_to_B = active


# ---------------------------------------------------------------------------
# Stacked MoE Blocks
# ---------------------------------------------------------------------------

@dataclass
class StackedMoEOutput:
    tokens_A:       torch.Tensor
    tokens_B:       torch.Tensor
    tokens_C:       torch.Tensor
    total_decorr:   torch.Tensor
    total_capacity: torch.Tensor
    total_ortho:    torch.Tensor
    total_intention: torch.Tensor
    aux_loss:       torch.Tensor
    log_dict:       Dict
    last_router:    RouterOutput


class StackedMoEBlocks(nn.Module):
    """N stacked MoEBlocks — the full MoE decoder backbone.

    WarmUpCrossAttentionLayer runs once before the block loop (§3.3).
    """

    def __init__(self, cfg: MoEBlockConfig, num_blocks: int = 4):
        super().__init__()
        self.cfg        = cfg
        self.num_blocks = num_blocks

        # [BUG-3] Single warmup layer, correct config type (MoEConfig).
        self.warmup_layer = WarmUpCrossAttentionLayer(cfg.to_moe_config())

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
        agent_types:    torch.Tensor,                   # (B, N_B) long  [BUG-6]
        ego_mask:       torch.Tensor,                   # (B, N_B) bool  [BUG-6]
        ego_distances:  torch.Tensor,                   # (B, N_B) float [BUG-6]
        ego_speed:      torch.Tensor,                   # (B,) float     [BUG-6]
        history_traj:   Optional[torch.Tensor] = None,  # (B, N_B, H, 4)[BUG-6]
        history_abs:    Optional[torch.Tensor] = None,  # (B, N_B, 2)   [BUG-6]
        gps_confidence: Optional[torch.Tensor] = None,  # (B,)           [BUG-6]
        intention_gt:   Optional[torch.Tensor] = None,  # (B, N_B) long  [BUG-6]
        step:           int = 0,
    ) -> StackedMoEOutput:
        device = tokens_A.device
        total_decorr    = torch.tensor(0.0, device=device)
        total_capacity  = torch.tensor(0.0, device=device)
        total_ortho     = torch.tensor(0.0, device=device)
        total_intention = torch.tensor(0.0, device=device)
        merged_log: Dict                   = {}
        last_router: Optional[RouterOutput] = None

        # Warmup cross-attention — once before the block loop (§3.3)
        tokens_A, tokens_B, tokens_C = self.warmup_layer(tokens_A, tokens_B, tokens_C)

        for block in self.blocks:
            out: MoEBlockOutput = block(
                tokens_A=tokens_A,
                tokens_B=tokens_B,
                tokens_C=tokens_C,
                spatial_xyz=spatial_xyz,
                token_types_C=token_types_C,
                t=t,
                agent_types=agent_types,       # [BUG-6]
                ego_mask=ego_mask,             # [BUG-6]
                ego_distances=ego_distances,   # [BUG-6]
                ego_speed=ego_speed,           # [BUG-6]
                history_traj=history_traj,     # [BUG-6]
                history_abs=history_abs,       # [BUG-6]
                gps_confidence=gps_confidence, # [BUG-6]
                intention_gt=intention_gt,     # [BUG-6]
                step=step,
            )
            tokens_A = out.tokens_A
            tokens_B = out.tokens_B
            tokens_C = out.tokens_C
            total_decorr    = total_decorr    + out.decorr_loss
            total_capacity  = total_capacity  + out.capacity_loss
            total_ortho     = total_ortho     + out.ortho_loss
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
            total_intention=total_intention,
            aux_loss=total_aux,
            log_dict=merged_log,
            last_router=last_router,
        )

    def invalidate_caches(self):
        for block in self.blocks: block.invalidate_caches()

    def reset_decorr_buffers(self):
        for block in self.blocks: block.reset_decorr_buffer()

    def set_stopgrad_C_to_B(self, active: bool):
        for block in self.blocks: block.set_stopgrad_C_to_B(active)

    def get_param_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        groups = []

        # Warmup layer
        groups.append({
            "params": list(self.warmup_layer.parameters()),
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
            # [BUG-4] Fixed: .parameters() not .self.parameters()
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    torch.manual_seed(42)

    # ── Shared test dimensions ────────────────────────────────────────────
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

    # ── Reusable input factory ────────────────────────────────────────────
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
        intention_gt  = torch.randint(0, 6, (B, NB))
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
        ("capacity_loss", out.capacity_loss),
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
        + out_g.decorr_loss + out_g.capacity_loss + out_g.ortho_loss
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
    original_warmup = backbone.warmup_layer.forward
    def _patched_warmup(tA, tB, tC):
        call_log.append("called")
        return original_warmup(tA, tB, tC)
    backbone.warmup_layer.forward = _patched_warmup

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
    backbone.warmup_layer.forward = original_warmup  # restore

    print(f"  Warmup called {len(call_log)} time(s)  (expect 1)")
    assert len(call_log) == 1, f"Warmup should run exactly once, ran {len(call_log)}"
    print(f"  Warmup runs exactly once: {PASS}")

    
    # Done
    
    print("\n" + "=" * 60)
    print("All moe_block.py __main__ checks PASSED.")
    print("=" * 60)