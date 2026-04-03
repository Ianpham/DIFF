"""
group_b/gate_query.py

GroupBGateQuery   (fixes audit §2.8)

Assembles the gate query vector from five sources and applies a diffusion-
timestep-conditioned temperature schedule τ(t).

Sources concatenated:
    repr3            [D]   — agent representation after Stage 3 (map-reweighted)
    ctx_A            [D]   — A→B sensory context  (stop-grad applied by caller)
    ctx_C            [D]   — C→B map context      (stop-grad applied by caller)
    t_emb            [D_t] — diffusion timestep embedding
    intention_logits [6]   — pre-gate behavioural signal from IntentionHeads

Temperature schedule τ(t):
    τ(t) = τ_max − (τ_max − τ_min) · t_frac

    t_frac = 0  →  most noisy (early diffusion)  →  τ = τ_max  →  soft routing
    t_frac = 1  →  clean signal (late diffusion)  →  τ = τ_min  →  sharp routing

The output is divided by τ(t) before being handed to the MoE router.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import INTENTION_LOGIT_DIM


class GroupBGateQuery(nn.Module):
    """
    Gate query formation for Group B agents.

    Args:
        D       : model hidden dimension  (repr3, ctx_A, ctx_C each have this width)
        D_t     : diffusion timestep embedding dimension
        tau_min : temperature lower bound  (sharp routing, low-noise regime)
        tau_max : temperature upper bound  (soft routing, high-noise regime)
    """

    def __init__(
        self,
        D:       int,
        D_t:     int,
        tau_min: float = 0.3,
        tau_max: float = 1.5,
    ) -> None:
        super().__init__()

        # Total input width of the concatenated query vector
        D_q = D + D + D + D_t + INTENTION_LOGIT_DIM   # 3D + D_t + 6

        self.proj    = nn.Linear(D_q, D)
        self.norm    = nn.LayerNorm(D)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.D       = D
        self.D_t     = D_t

    # ------------------------------------------------------------------
    # Temperature schedule
    # ------------------------------------------------------------------

    def compute_tau(self, t_frac: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample (and per-agent, once broadcast) temperature.

        Args:
            t_frac : [B] or [B, N, 1]  — diffusion time fraction in [0, 1]

        Returns:
            tau : same shape as t_frac
        """
        return self.tau_max - (self.tau_max - self.tau_min) * t_frac

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        repr3:             torch.Tensor,   # [B, N, D]
        ctx_A:             torch.Tensor,   # [B, N, D]   stop-grad applied upstream
        ctx_C:             torch.Tensor,   # [B, N, D]   stop-grad applied upstream
        t_emb:             torch.Tensor,   # [B, D_t]
        intention_logits:  torch.Tensor,   # [B, N, 6]
        t_frac:            torch.Tensor,   # [B]          diffusion time in [0, 1]
    ) -> torch.Tensor:                     # [B, N, D]    temperature-scaled gate query

        B, N, D = repr3.shape

        # ── 1. Broadcast t_emb and t_frac to per-agent shape ─────────────
        t_emb_exp  = t_emb[:, None, :].expand(B, N, self.D_t)   # [B, N, D_t]
        t_frac_exp = t_frac[:, None, None].expand(B, N, 1)       # [B, N, 1]

        # ── 2. Concatenate all five sources ───────────────────────────────
        gate_in = torch.cat(
            [repr3, ctx_A, ctx_C, t_emb_exp, intention_logits],
            dim=-1,
        )  # [B, N, 3D + D_t + 6]

        # ── 3. Linear projection + LayerNorm ──────────────────────────────
        q = self.norm(self.proj(gate_in))   # [B, N, D]

        # ── 4. Temperature scaling ────────────────────────────────────────
        tau = self.compute_tau(t_frac_exp)  # [B, N, 1]
        q   = q / tau                        # [B, N, D]

        return q   # feeds directly into the MoE router as the gating signal