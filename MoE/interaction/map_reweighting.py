"""
group_b/map_reweighting.py

Stage 3 — MapContextReweighting   (fixes audit §2.6)

After the directed cross-group attention modules (A→B sensory confirmation,
C→B map context) have produced context vectors for each agent, this module
learns a per-agent scalar gate that decides how much map context each agent
should absorb into its representation.

Agents near complex intersections or merges will learn high gate values;
agents on open highways or in clear lanes will learn low gate values.

The stop-gradient on ctx_C must be applied BEFORE this module is called
(i.e. in moe_block.py when the C→B cross-attention output is passed in).
This module itself does not call .detach() — it trusts the caller.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MapContextReweighting(nn.Module):
    """
    Per-agent sigmoid gate for fusing map context into agent representations.

    Formulation:
        gate_in_i = concat[repr2_i, ctx_C_i]          [B, N, 2D]
        g_i       = σ( W_gate · gate_in_i )            [B, N, 1]
        repr3_i   = g_i · ctx_C_i + (1 − g_i) · repr2_i
        output    = LayerNorm(repr3)

    Args:
        D : model hidden dimension
    """

    def __init__(self, D: int) -> None:
        super().__init__()
        # Single linear → scalar gate per agent
        self.gate_proj = nn.Linear(2 * D, 1)
        self.norm      = nn.LayerNorm(D)

    def forward(
        self,
        repr2: torch.Tensor,   # [B, N, D]  — agent repr after Stage 2
        ctx_C: torch.Tensor,   # [B, N, D]  — map context from C→B xattn
                               #               stop-grad applied by caller
    ) -> torch.Tensor:         # [B, N, D]  — repr3: map-reweighted representation

        gate_in = torch.cat([repr2, ctx_C], dim=-1)     # [B, N, 2D]
        g       = torch.sigmoid(self.gate_proj(gate_in)) # [B, N, 1]

        repr3 = g * ctx_C + (1.0 - g) * repr2           # [B, N, D]
        return self.norm(repr3)