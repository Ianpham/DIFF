"""
group_b/pipeline.py

GroupBPipeline — complete assembly of the Group B interaction pipeline.

This module wires together all sub-modules in execution order:

    Input token construction
    │
    ├─ EgoRelativeGeometry          § 2.1
    ├─ AgentHistoryEncoder          § 2.2 / §8.1  (Option B: after ego-frame)
    ├─ type embedding + group-id
    └─ input_proj  →  [B, N, D]
         │
    Stage 1: EgoCentricCrossAttention       §2.3   O(N)
         │
    Stage 2: AgentAgentAttention            §2.4   O(N·K)
         │
    Stage 3: MapContextReweighting          §2.6   (fuses C→B stop-grad context)
         │
    IntentionHeads  →  int_logits           §2.7
         │
    GroupBGateQuery → gate_q / τ(t)         §2.8
         │
    GroupBSkipPredictor → skip_mask         §4.5

Auxiliary losses collected:
    L_intention  (when intention_labels provided)

The pipeline does NOT:
  - call stop_grad on ctx_A / ctx_C (caller's responsibility in moe_block.py)
  - apply the skip mask to tokens    (caller's responsibility in expert_ffn.py)
  - dispatch tokens to experts       (caller's responsibility in token_router.py)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import (
    GroupBConfig,
    EGO_TYPE_ID,
    K_TABLE,
)
from .input_token      import EgoRelativeGeometry, AgentHistoryEncoder
from .interaction_stages import (
    AgentAgentAttention,
    EgoCentricCrossAttention,
    compute_pairwise_distances_from_states,
)
from .map_reweighting    import MapContextReweighting
from .intention_head    import IntentionHeads, intention_loss
from .gate_query         import GroupBGateQuery
from .skip_shared    import GroupBSkipPredictor, build_rare_mask
from .output            import GroupBOutput


class GroupBPipeline(nn.Module):
    """
    Full Group B (intention & interaction) pipeline.

    Args:
        cfg : GroupBConfig — all hyper-parameters in one place

    Forward inputs
    --------------
    agent_states     [B, N, 5]        world-frame  (x, y, heading, vx, vy)
    agent_histories  [B, N, T, 5]     last T steps of world-frame agent state;
                                      the pipeline converts to ego-frame internally
    agent_type_ids   [B, N]           integer type IDs (see config.py)
    ego_state        [B, 5]           current ego vehicle state, world frame
    ctx_A            [B, N, D]        A→B sensory context  (stop-grad by caller)
    ctx_C            [B, N, D]        C→B map context      (stop-grad by caller)
    t_emb            [B, D_t]         diffusion timestep embedding
    t_frac           [B]              diffusion time fraction ∈ [0, 1]
    group_id_emb     [D_id]           orthogonal Group B identity embedding
                                      (shared across all agents; 1-D, no batch dim)
    scene_class      str              "highway" / "urban" / "parking" / "default"
    intention_labels [B, N] or None   ground-truth intention class (-1 = unlabelled)

    Forward outputs
    ---------------
    GroupBOutput  (see outputs.py)
    """

    def __init__(self, cfg: GroupBConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim

        # ── Input construction 
        self.ego_rel_geom    = EgoRelativeGeometry(cfg.D_geom)
        self.history_encoder = AgentHistoryEncoder(cfg.state_dim, cfg.T_hist, cfg.D_hist)
        self.type_emb        = nn.Embedding(cfg.n_agent_types, cfg.D_type)
        self.input_proj      = nn.Linear(cfg.input_proj_in_dim, D)
        self.input_norm      = nn.LayerNorm(D)

        # Stage 1: ego-centric cross-attention 
        self.stage1 = EgoCentricCrossAttention(D, n_heads=cfg.stage1_n_heads)

        # Stage 2: agent-agent attention 
        self.stage2 = AgentAgentAttention(
            D,
            n_heads=cfg.stage2_n_heads,
            d_ref=cfg.stage2_d_ref,
            alpha=cfg.stage2_alpha,
        )

        # Stage 3: map context re-weighting 
        self.stage3 = MapContextReweighting(D)

        # Intention heads 
        self.intention_heads = IntentionHeads(D)

        # Gate query with tau(t) 
        self.gate_query = GroupBGateQuery(
            D=D,
            D_t=cfg.D_t,
            tau_min=cfg.tau_min,
            tau_max=cfg.tau_max,
        )

        # ── Learned skip predictor
        self.skip_predictor = GroupBSkipPredictor(D, floor=cfg.skip_floor)


    # Helper: convert world-frame history to ego-relative history
 

    def _history_to_ego_frame(
        self,
        agent_histories: torch.Tensor,   # [B, N, T, 5]  world frame
        ego_state:       torch.Tensor,   # [B, 5]
    ) -> torch.Tensor:                   # [B, N, T, 5]  ego frame
        """
        Converts each history timestep into the current ego frame.
        Applies the same translation + rotation as EgoRelativeGeometry,
        but outputs the full 5-dim state vector (not a learned projection).

        This gives the GRU a coordinate-system-consistent input.
        """
        B, N, T, S = agent_histories.shape

        # Ego position and heading from the CURRENT timestep
        ego_x   = ego_state[:, 0:1, None]   # [B, 1, 1]
        ego_y   = ego_state[:, 1:2, None]   # [B, 1, 1]
        ego_h   = ego_state[:, 2]           # [B]
        cos_h   = torch.cos(ego_h)          # [B]
        sin_h   = torch.sin(ego_h)          # [B]

        # Translate
        dx = agent_histories[..., 0] - ego_x.squeeze(-1)[:, :, None]   # [B, N, T]
        dy = agent_histories[..., 1] - ego_y.squeeze(-1)[:, :, None]   # [B, N, T]

        # Rotate  (broadcast cos_h / sin_h: [B] → [B, 1, 1])
        cos_b = cos_h[:, None, None]
        sin_b = sin_h[:, None, None]
        dx_r  =  cos_b * dx + sin_b * dy
        dy_r  = -sin_b * dx + cos_b * dy

        # Relative heading
        dh    = agent_histories[..., 2] - ego_state[:, 2, None, None]
        dh    = torch.atan2(torch.sin(dh), torch.cos(dh))

        # Relative velocities (no rotation here — treat as scalar magnitudes)
        dvx   = agent_histories[..., 3] - ego_state[:, 3, None, None]
        dvy   = agent_histories[..., 4] - ego_state[:, 4, None, None]

        return torch.stack([dx_r, dy_r, dh, dvx, dvy], dim=-1)  # [B, N, T, 5]


    # Forward

    def forward(
        self,
        agent_states:      torch.Tensor,              # [B, N, 5]
        agent_histories:   torch.Tensor,              # [B, N, T_hist, 5]
        agent_type_ids:    torch.Tensor,              # [B, N]
        ego_state:         torch.Tensor,              # [B, 5]
        ctx_A:             torch.Tensor,              # [B, N, D]
        ctx_C:             torch.Tensor,              # [B, N, D]
        t_emb:             torch.Tensor,              # [B, D_t]
        t_frac:            torch.Tensor,              # [B]
        group_id_emb:      torch.Tensor,              # [D_id]
        scene_class:       str = "default",
        intention_labels:  Optional[torch.Tensor] = None,   # [B, N] or None
    ) -> GroupBOutput:

        B, N, _ = agent_states.shape
        cfg = self.cfg

        # 1. Input token construction 
        # 1a. Ego-relative geometry of current timestep
        geom = self.ego_rel_geom(agent_states, ego_state)      # [B, N, D_geom]

        # 1b. History in ego frame → GRU embedding
        hist_ego = self._history_to_ego_frame(agent_histories, ego_state)
        hist     = self.history_encoder(hist_ego)              # [B, N, D_hist]

        # 1c. Agent-type embedding
        type_e = self.type_emb(agent_type_ids)                 # [B, N, D_type]

        # 1d. Group-id embedding  (same vector broadcast over all agents)
        gid_e = group_id_emb.unsqueeze(0).unsqueeze(0)         # [1, 1, D_id]
        gid_e = gid_e.expand(B, N, -1)                        # [B, N, D_id]

        # 1e. Project to hidden_dim
        raw_tokens = torch.cat([hist, geom, type_e, gid_e], dim=-1)  # [B, N, in_dim]
        tokens     = self.input_norm(self.input_proj(raw_tokens))     # [B, N, D]

        #  2. Extract ego token for Stage 1 
        ego_mask = (agent_type_ids == EGO_TYPE_ID)   # [B, N]
        # Each scene has exactly one ego token; reshape to [B, 1, D]
        ego_token = tokens[ego_mask].view(B, 1, -1)  # [B, 1, D]

        #  3. Stage 1: ego-centric cross-attention 
        repr1 = self.stage1(tokens, ego_token)        # [B, N, D]

        #  4. Stage 2: agent-agent attention 
        distances = compute_pairwise_distances_from_states(agent_states, ego_state)
        K         = K_TABLE.get(scene_class, K_TABLE["default"])
        repr2     = self.stage2(repr1, distances, K)  # [B, N, D]

        # 5. Stage 3: map context re-weighting
        # ctx_C must have stop-grad applied by the caller (moe_block.py)
        repr3 = self.stage3(repr2, ctx_C)             # [B, N, D]

        # 6. Intention heads 
        int_logits, v_mask, p_mask = self.intention_heads(repr3, agent_type_ids)
        # int_logits: [B, N, 6]

        #  7. Gate query with τ(t) 
        # ctx_A and ctx_C must have stop-grad applied by the caller
        gate_q = self.gate_query(
            repr3            = repr3,
            ctx_A            = ctx_A,
            ctx_C            = ctx_C,
            t_emb            = t_emb,
            intention_logits = int_logits,
            t_frac           = t_frac,
        )  # [B, N, D]

        # 8. Skip predictor 
        rare_mask = build_rare_mask(agent_type_ids, K_rare=cfg.K_rare)
        skip_mask, skip_scores = self.skip_predictor(repr3, t_frac, rare_mask)

        # 9. Auxiliary losses 
        aux_losses: dict = {}
        if intention_labels is not None:
            aux_losses["L_intention"] = intention_loss(
                int_logits, intention_labels, agent_type_ids
            )

        return GroupBOutput(
            tokens      = repr3,
            gate_query  = gate_q,
            skip_mask   = skip_mask,
            skip_scores = skip_scores,
            int_logits  = int_logits,
            aux_losses  = aux_losses,
        )