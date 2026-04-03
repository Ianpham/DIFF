"""
group_b/intention_heads.py

IntentionHeads module and intention_loss function   (fixes audit §2.7)

Two separate prediction heads share the same interface:
  - Vehicle head    : 6-class logits  (lane keep, LC-L, LC-R, merge, cut-in, stop)
  - Pedestrian head : 2-class logits  (crossing, not-crossing)

Pedestrian logits are zero-padded to width 6 (INTENTION_LOGIT_DIM) so that
the gate query module receives a uniform [B, N, 6] tensor for all agents.

The intention logits serve two purposes:
  1. Feed directly into the gate query (§2.8) — they inform routing decisions
     with explicit behavioural signal before any expert is selected.
  2. Supervised by L_intention (cross-entropy) against ground-truth labels
     from the dataset.  Labels of -1 are masked out (unlabelled agents).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    AGENT_TYPE_VEHICLE,
    AGENT_TYPE_PEDESTRIAN,
    INTENTION_LOGIT_DIM,
    N_VEHICLE_INTENTIONS,
    N_PEDESTRIAN_INTENTIONS,
)


# ---------------------------------------------------------------------------
# IntentionHeads
# ---------------------------------------------------------------------------

class IntentionHeads(nn.Module):
    """
    Two-head intention predictor: one head per agent class.

    Args:
        D : model hidden dimension

    Inputs:
        repr3          : [B, N, D]    — agent representations after Stage 3
        agent_type_ids : [B, N]       — integer type IDs per agent

    Outputs:
        intention_logits : [B, N, INTENTION_LOGIT_DIM]
                           vehicle agents  → 6-class logits in [:, :, :6]
                           pedestrian agents → 2-class logits in [:, :, :2],
                                               zeros in [:, :, 2:6]
                           cyclist / ego    → all zeros (no prediction)
        v_mask           : [B, N]  bool — vehicle agent positions
        p_mask           : [B, N]  bool — pedestrian agent positions
    """

    def __init__(self, D: int) -> None:
        super().__init__()

        self.vehicle_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, N_VEHICLE_INTENTIONS),   # → 6 logits
        )

        self.pedestrian_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, N_PEDESTRIAN_INTENTIONS),  # → 2 logits
        )

    def forward(
        self,
        repr3:          torch.Tensor,   # [B, N, D]
        agent_type_ids: torch.Tensor,   # [B, N]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, D = repr3.shape
        device = repr3.device

        # Output buffer — zeroed; only vehicle and pedestrian slots are filled
        intention_logits = torch.zeros(B, N, INTENTION_LOGIT_DIM, device=device)

        v_mask = (agent_type_ids == AGENT_TYPE_VEHICLE)     # [B, N]
        p_mask = (agent_type_ids == AGENT_TYPE_PEDESTRIAN)  # [B, N]

        if v_mask.any():
            v_repr   = repr3[v_mask]                         # [Nv, D]
            v_logits = self.vehicle_head(v_repr)             # [Nv, 6]
            intention_logits[v_mask] = v_logits

        if p_mask.any():
            p_repr   = repr3[p_mask]                         # [Np, D]
            p_logits = self.pedestrian_head(p_repr)          # [Np, 2]
            # Pad pedestrian logits into the first 2 slots; rest stay zero
            intention_logits[p_mask, :N_PEDESTRIAN_INTENTIONS] = p_logits

        return intention_logits, v_mask, p_mask


# ---------------------------------------------------------------------------
# intention_loss
# ---------------------------------------------------------------------------

def intention_loss(
    intention_logits: torch.Tensor,   # [B, N, INTENTION_LOGIT_DIM]
    intention_labels: torch.Tensor,   # [B, N]  ground-truth int; -1 = unlabelled
    agent_type_ids:   torch.Tensor,   # [B, N]
) -> torch.Tensor:                    # scalar
    """
    L_intention = mean of per-class cross-entropy losses over labelled agents.

    Vehicle agents    → CE over 6 classes
    Pedestrian agents → CE over 2 classes  (using only the first 2 logit slots)
    Cyclists / ego    → ignored

    Returns 0.0 (as a tensor with grad) if there are no labelled agents in
    the batch, so the caller can safely add it to the total loss.
    """
    device = intention_logits.device
    loss   = torch.zeros(1, device=device, requires_grad=False)
    n_terms = 0

    # ── Vehicle loss ──────────────────────────────────────────────────────
    v_mask = (agent_type_ids == AGENT_TYPE_VEHICLE)  # [B, N]
    if v_mask.any():
        logits = intention_logits[v_mask]              # [Nv, 6]
        labels = intention_labels[v_mask]              # [Nv]
        valid  = (labels >= 0)
        if valid.any():
            loss    = loss + F.cross_entropy(logits[valid], labels[valid].long())
            n_terms += 1

    # ── Pedestrian loss ───────────────────────────────────────────────────
    p_mask = (agent_type_ids == AGENT_TYPE_PEDESTRIAN)  # [B, N]
    if p_mask.any():
        logits = intention_logits[p_mask, :N_PEDESTRIAN_INTENTIONS]  # [Np, 2]
        labels = intention_labels[p_mask]                            # [Np]
        valid  = (labels >= 0)
        if valid.any():
            loss    = loss + F.cross_entropy(logits[valid], labels[valid].long())
            n_terms += 1

    # Normalise: average over the two loss terms (vehicle + pedestrian),
    # not over the number of agents, so the scale stays stable as N varies.
    return loss / max(n_terms, 1)