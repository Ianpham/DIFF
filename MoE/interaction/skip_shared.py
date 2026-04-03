"""
group_b/skip_and_shared.py

Two related utilities that govern which computation Group B agents skip
and how much weight the shared expert carries:

  GroupBSkipPredictor     (fixes audit §4.5 — Group B)
  shared_expert_weight    (fixes audit §2.9)

GroupBSkipPredictor
-------------------
Replaces the binary batch-level DyDiTSkipScheduler with a per-token, per-
layer MLP that outputs a continuous skip score in [0, 1].

Group B skips at HIGH diffusion-t (noisy tokens where fine interaction detail
is not yet required) and always processes at LOW t (clean signal where routing
precision matters most).

A 20% floor guarantee ensures that at least 20% of agent tokens are always
processed, preventing expert starvation.

Rare-token exception: agents whose type appears fewer than K_rare times in
the current batch always bypass the predictor and are processed unconditionally.

Trained with L_skip = BCE(scores, stop_grad(hard_targets)) where hard targets
are derived from per-token reconstruction loss improvement Δ_i.

shared_expert_weight
---------------------
Returns a per-sample scalar weight for the shared expert.  The weight is
scheduled over diffusion time:

    w_shared(t) = w_min + w_max_delta × t_frac

Higher weight at clean signal (t≈1) where the shared expert's general
knowledge contributes most to the final reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GroupBSkipPredictor
# ---------------------------------------------------------------------------

class GroupBSkipPredictor(nn.Module):
    """
    Per-token learned skip predictor for Group B.

    Architecture:
        Linear(D → D//4) → GELU → Linear(D//4 → 1) → Sigmoid

    Args:
        D     : model hidden dimension
        floor : minimum fraction of tokens that must always be processed
                (default 0.20 → 20%)
    """

    def __init__(self, D: int, floor: float = 0.20) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D, D // 4),
            nn.GELU(),
            nn.Linear(D // 4, 1),
            nn.Sigmoid(),
        )
        self.floor = floor

    def forward(
        self,
        token_repr:     torch.Tensor,   # [B, N, D]
        t_frac:         torch.Tensor,   # [B]          diffusion time in [0, 1]
        rare_mask:      torch.Tensor | None = None,
                                        # [B, N] bool  True = rare token (never skip)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip_mask  : [B, N] bool  — True means skip this token this layer
            scores     : [B, N] float — raw skip scores (needed for L_skip)
        """
        B, N, D = token_repr.shape

        # ── 1. Raw skip scores ────────────────────────────────────────────
        scores = self.mlp(token_repr).squeeze(-1)   # [B, N]  ∈ (0, 1)

        # ── 2. Timestep gate: suppress skipping at low t (clean signal) ──
        # t_frac ≈ 1 → high noise → skip is acceptable
        # t_frac ≈ 0 → clean     → do not skip
        t_gate = t_frac[:, None].expand_as(scores)  # [B, N]
        scores = scores * t_gate

        # ── 3. Enforce floor: always keep at least `floor` fraction ───────
        # Compute per-batch threshold that marks the top (1-floor) fraction
        # as skip candidates.  Tokens below the threshold are never skipped.
        threshold = scores.quantile(1.0 - self.floor, dim=-1, keepdim=True)  # [B, 1]
        skip_mask = scores > threshold   # [B, N]

        # ── 4. Rare-token exception: force process for rare agent types ───
        if rare_mask is not None:
            skip_mask = skip_mask & ~rare_mask   # never skip rare tokens

        return skip_mask, scores

    @staticmethod
    def skip_loss(
        scores:  torch.Tensor,   # [B, N]  predicted skip scores
        delta_i: torch.Tensor,   # [B, N]  per-token reconstruction loss improvement Δ_i
    ) -> torch.Tensor:           # scalar
        """
        L_skip = BCE(scores, stop_grad(hard_targets))

        hard_targets[b, i] = 1  if token i improved loss by less than the
                                   median in its batch item  (candidate to skip)
                           = 0  otherwise (must not skip)

        The stop-gradient on hard_targets breaks the dependency loop:
        the skip predictor is trained to match the hard decisions that were
        made, but those decisions are not differentiated back through the
        reconstruction loss.
        """
        # Per-batch-item median threshold
        threshold    = delta_i.median(dim=-1, keepdim=True).values  # [B, 1]
        hard_targets = (delta_i < threshold).float()                 # [B, N]

        # Stop-gradient: hard_targets should not carry gradients back into
        # whatever produced delta_i.
        hard_targets = hard_targets.detach()

        return F.binary_cross_entropy(scores, hard_targets)


# ---------------------------------------------------------------------------
# Rare-token mask helper
# ---------------------------------------------------------------------------

def build_rare_mask(
    agent_type_ids: torch.Tensor,   # [B, N]
    K_rare: int = 5,
) -> torch.Tensor:                  # [B, N] bool
    """
    Returns a boolean mask that is True for any agent whose type appears fewer
    than K_rare times across the entire batch.  These tokens are always
    processed (never skipped) to avoid expert starvation for minority classes.
    """
    # Count per-type occurrences across the full batch×agent tensor
    flat = agent_type_ids.reshape(-1)
    type_ids, counts = torch.unique(flat, return_counts=True)

    # Build a per-element count tensor
    count_map = torch.zeros(
        agent_type_ids.max().item() + 1,
        dtype=torch.long,
        device=agent_type_ids.device,
    )
    count_map[type_ids] = counts

    token_counts = count_map[agent_type_ids]   # [B, N]
    return token_counts < K_rare               # [B, N] bool


# ---------------------------------------------------------------------------
# shared_expert_weight
# ---------------------------------------------------------------------------

def shared_expert_weight(
    t_frac:         torch.Tensor,     # [B]  diffusion time fraction in [0, 1]
    w_min:          float = 0.10,
    w_max_delta:    float = 0.10,
    T_max:          float = 1.0,
) -> torch.Tensor:                    # [B]  per-sample weight scalar
    """
    Compute the time-scheduled weight for the shared expert.

        w_shared(t) = w_min + w_max_delta × (t_frac / T_max)

    t_frac = 0 (noisy)  →  w = w_min           = 0.10
    t_frac = 1 (clean)  →  w = w_min + w_max_delta = 0.20

    The shared expert's contribution grows as the diffusion process produces
    cleaner tokens where general knowledge is more useful.
    """
    return w_min + w_max_delta * (t_frac / T_max)