"""
group_b/risk_gate.py

Risk-adaptive gate confidence threshold   (resolves audit §8.3)

The gate confidence threshold is modulated by a per-agent risk score derived
from the local scene:

    threshold(b, i) = base_threshold + alpha_risk × risk_score(b, i)

High-risk situations (many nearby agents, high relative speeds) push the
threshold up, forcing agents to commit to an expert with higher confidence
before they stop routing.  This prevents premature commitment in complex
scenarios and reduces the chance of routing collapse when the scene is dense.

Two exported symbols:
  compute_risk_score         — derives [B, N] risk floats from scene geometry
  risk_adaptive_threshold    — applies them to a scalar base threshold
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# compute_risk_score
# ---------------------------------------------------------------------------

def compute_risk_score(
    distances:  torch.Tensor,    # [B, N, N]  pairwise ego-relative distances (m)
    velocities: torch.Tensor,    # [B, N, 2]  (vx, vy) per agent in ego frame
    K:          int   = 5,
    V_MAX:      float = 30.0,    # m/s — normalisation constant for velocity
) -> torch.Tensor:               # [B, N]  ∈ [0, 1]
    """
    Per-agent risk score combining spatial crowding and speed.

    proximity_risk = mean of (1 / (d + 1)) over the K nearest agents
                     (large when neighbours are very close)

    velocity_risk  = ||v_i|| / V_MAX
                     (large for fast-moving agents)

    risk_score = clamp((proximity_risk + velocity_risk) / 2, 0, 1)

    The factor of 2 in the denominator keeps the sum in a sensible range
    before clamping to [0, 1].
    """
    # ── 1. Proximity risk ─────────────────────────────────────────────────
    K_safe = min(K, distances.shape[-1])
    topk_dist, _ = distances.topk(K_safe, dim=-1, largest=False)   # [B, N, K_safe]
    proximity_risk = (1.0 / (topk_dist + 1.0)).mean(dim=-1)        # [B, N]

    # ── 2. Velocity risk ──────────────────────────────────────────────────
    speed = velocities.norm(dim=-1)                                 # [B, N]
    velocity_risk = speed / V_MAX                                   # [B, N]

    # ── 3. Combine and clamp ──────────────────────────────────────────────
    raw  = (proximity_risk + velocity_risk) / 2.0
    return raw.clamp(0.0, 1.0)                                      # [B, N]


# ---------------------------------------------------------------------------
# risk_adaptive_threshold
# ---------------------------------------------------------------------------

def risk_adaptive_threshold(
    risk_score:      torch.Tensor,   # [B, N]  ∈ [0, 1]
    base_threshold:  float = 0.40,
    alpha_risk:      float = 0.20,
) -> torch.Tensor:                   # [B, N]  per-agent gate threshold
    """
    Compute the per-agent confidence threshold for expert commitment.

        threshold(b, i) = base_threshold + alpha_risk × risk_score(b, i)

    At minimum risk (0.0):  threshold = 0.40
    At maximum risk (1.0):  threshold = 0.60  (with default alpha_risk=0.20)

    The higher the threshold, the more confident the gate logit must be
    before an agent commits to a single expert.  In dense / high-speed
    scenes the router is forced to be more selective, distributing load
    more evenly across experts.
    """
    return base_threshold + alpha_risk * risk_score