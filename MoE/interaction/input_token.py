"""
group_b/input_tokens.py

Input token construction for Group B agents.

Two sub-modules are defined here:
  EgoRelativeGeometry   — converts world-frame agent states into ego-frame features
  AgentHistoryEncoder   — GRU over per-agent trajectory history (in ego frame)

Design note (resolves audit §2.2, §8.1):
  History is encoded AFTER ego-relative geometry is computed (Option B).
  This means the GRU always operates on ego-frame coordinates, so the resulting
  embedding is directly comparable across agents without any further rotation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EgoRelativeGeometry
# ---------------------------------------------------------------------------

class EgoRelativeGeometry(nn.Module):
    """
    Converts world-frame agent states into a learned ego-relative geometry
    embedding.

    Raw feature vector (7 dims) per agent:
        dx_rot   — longitudinal offset in ego heading frame  (metres)
        dy_rot   — lateral offset in ego heading frame       (metres)
        dh       — relative heading, wrapped to [-π, π]      (radians)
        dvx      — relative longitudinal velocity            (m/s)
        dvy      — relative lateral velocity                 (m/s)
        dist     — Euclidean distance from ego               (metres)
        bearing  — bearing angle from ego                    (radians)

    Args:
        D_g: output embedding dimension (default 64)

    Inputs:
        agent_states : [B, N, 5]  (x, y, heading, vx, vy) in world frame
        ego_state    : [B, 5]     ego vehicle state in world frame

    Output:
        rel_geom : [B, N, D_g]
    """

    def __init__(self, D_g: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(7, D_g)
        self.norm = nn.LayerNorm(D_g)

    def forward(
        self,
        agent_states: torch.Tensor,   # [B, N, 5]
        ego_state:    torch.Tensor,   # [B, 5]
    ) -> torch.Tensor:                # [B, N, D_g]

        # ── 1. Translate: move agents to ego-centred origin ──────────────
        dx = agent_states[..., 0] - ego_state[:, None, 0]  # [B, N]
        dy = agent_states[..., 1] - ego_state[:, None, 1]  # [B, N]

        # ── 2. Rotate: align axes with ego heading ────────────────────────
        # ego_h shape: [B, 1] → broadcasts over N agents
        ego_h = ego_state[:, 2:3]                           # [B, 1]
        cos_h = torch.cos(ego_h)
        sin_h = torch.sin(ego_h)

        dx_rot =  cos_h * dx + sin_h * dy                  # [B, N]
        dy_rot = -sin_h * dx + cos_h * dy                  # [B, N]

        # ── 3. Relative heading (wrapped) ─────────────────────────────────
        dh = agent_states[..., 2] - ego_state[:, None, 2]  # [B, N]
        dh = torch.atan2(torch.sin(dh), torch.cos(dh))     # wrap to [-π, π]

        # ── 4. Relative velocity in ego frame ─────────────────────────────
        dvx = agent_states[..., 3] - ego_state[:, None, 3]
        dvy = agent_states[..., 4] - ego_state[:, None, 4]

        # ── 5. Distance and bearing (useful spatial priors) ───────────────
        dist    = torch.sqrt(dx_rot ** 2 + dy_rot ** 2 + 1e-6)  # [B, N]
        bearing = torch.atan2(dy_rot, dx_rot)                    # [B, N]

        # ── 6. Stack, project, normalise ─────────────────────────────────
        raw = torch.stack(
            [dx_rot, dy_rot, dh, dvx, dvy, dist, bearing], dim=-1
        )  # [B, N, 7]

        return self.norm(self.proj(raw))   # [B, N, D_g]


# ---------------------------------------------------------------------------
# AgentHistoryEncoder
# ---------------------------------------------------------------------------

class AgentHistoryEncoder(nn.Module):
    """
    Encodes the last T_hist timesteps of an agent's state trajectory into a
    fixed-size embedding using a single-layer GRU.

    Important: the history tensor passed in here must already be expressed in
    the ego-relative coordinate frame (computed by EgoRelativeGeometry or an
    equivalent transform).  This is Option B from audit §8.1.

    Args:
        state_dim : width of each timestep's state vector
        T_hist    : number of history timesteps (sequence length for GRU)
        D_h       : GRU hidden size / output embedding dimension

    Inputs:
        history_in_ego_frame : [B, N, T_hist, state_dim]

    Output:
        hist_emb : [B, N, D_h]  — final GRU hidden state per agent
    """

    def __init__(self, state_dim: int, T_hist: int, D_h: int) -> None:
        super().__init__()
        self.gru     = nn.GRU(state_dim, D_h, batch_first=True)
        self.T_hist  = T_hist
        self.D_h     = D_h

    def forward(
        self,
        history_in_ego_frame: torch.Tensor,  # [B, N, T_hist, state_dim]
    ) -> torch.Tensor:                        # [B, N, D_h]

        B, N, T, D = history_in_ego_frame.shape

        # Merge batch and agent dims so the GRU sees (B*N) independent sequences
        x = history_in_ego_frame.view(B * N, T, D)   # [B*N, T, state_dim]

        _, h = self.gru(x)          # h: [1, B*N, D_h]  (1 layer, 1 direction)
        h = h.squeeze(0)            # [B*N, D_h]

        return h.view(B, N, self.D_h)   # [B, N, D_h]