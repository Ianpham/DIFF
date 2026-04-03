
"""
group_b/interaction_stages.py
 
Stage 1 — EgoCentricCrossAttention   (fixes audit §2.3)
Stage 2 — AgentAgentAttention        (fixes audit §2.4)
 
Helper — compute_pairwise_distances  (shared utility)
 
Design principles:
  Stage 1  O(N)  : each agent attends ONLY to the ego token.
                   No agent ever reads another agent's raw features directly.
  Stage 2  O(N·K): each agent attends to its K nearest neighbours, ranked by
                   ego-relative distance.  A distance-bias term on the logits
                   encodes the physical prior that closer agents matter more.
"""
 
from __future__ import annotations
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from .config import K_TABLE
 

# helper

def compute_pairwise_distances(
        ego_rel_geom: torch.Tensor, # [B, N, D_g] - ego-relative geometry embeddings
)-> torch.Tensor:                   # [B, N, N] - symmetric distance matrix
    """
    Approximate pairwise distance between agents using the first two dim of
    the ego-relative geometry vector (dx_rot, dy_rot before projection)

    Note: callers that have access to the raw dx_rot / rot values should
    pass those directly instead. This function is a fallback that works  on
    the projected embeddings via L2 Norms: it is scale-invarient but not metrics in metres. 
    For K-NN mask it gives the right ordering
    """

    # use L2 distance the embedding space as a proxy for spatial proximity
    # [B, N, 1, D_g] - [B, 1, N, D_g] -> [B, N, N]
    diff = ego_rel_geom.unsqueeze(2) - ego_rel_geom.unsqueeze(1)
    return diff.pow(2).sum(dim = -1).sqrt()     # [B, N, N]

def compute_pairwise_distances_from_states(
        agent_states: torch.Tensor, # [B, N, 5] world frame (x, y, ...)
        ego_state:    torch.Tensor, # [B, 5]
)-> torch.Tensor:                   # [B, N, N] distance in metres
    """
    Preferred version: computes Euclidean distances in metres directly from
    the raw world-frame positions.  Use this when world-frame states are
    available (e.g. inside GroupBPipeline.forward).
    """
    #  positions in world frame
    xy = agent_states[..., :2]           # [B, N, 2]
    diff = xy.unsqueeze(2) - xy.unsqueeze(1)  # [B, N, N, 2]
    return diff.pow(2).sum(dim=-1).sqrt()     # [B, N, N]

# stage 1  - ego-centric cross-attention

class EgoCentricCrossAttention(nn.Module):
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

    def __init__(self, D_g: int = 64)-> None:
        super().__init__()
        self.proj = nn.Linear( 8, D_g)
        self.norm = nn.LayerNorm(D_g)

    def forward(
            self,
            agent_states: torch.Tensor,     # [B, N, 5]
            ego_state: torch.Tensor,        # [B, 5]
    )-> torch.Tensor:
        # 1. Translate: move agents to ego-centerd origin
        dx = agent_states[..., 0] - ego_state[:, None, 0] # [B, N]
        dy = agent_states[..., 1] - ego_state[:, None, 1] # [B, N]

        # 2.Rotate: align axes with ego heading
        # ego_h shape: [B, 1] -> broadcasts over N agents
        ego_h = ego_state[:, 2:3]       # [B, 1]
        cos_h = torch.cos(ego_h)
        sin_h = torch.sin(ego_h)

        dx_rot = cos_h * dx + sin_h * dy        # [B, N]
        dy_rot = -sin_h * dx + cos_h * dy       # [B, N]

        # 3. Relative heading (wrapped)
        dh = agent_states[..., 2] - ego_state[:, None, 2] # [B, N]
        dh = torch.arctan(torch.sin(dh), torch.cos(dh))   # wrap to [-pi, pi]

        # 4. relative velocity in ego frame
        dvx = agent_states[..., 3] - ego_state[:, None, 3]
        dvy = agent_states[..., 4] - ego_state[:, None, 4]

        # 5. distance and bearing (useful spatial priors)
        dist    = torch.sqrt(dx_rot ** 2 + dy_rot ** 2 + 1e-6)  # [B, N]
        bearing = torch.atan2(dy_rot, dx_rot)                   # [B, N]

        # 6. stack jproject normalize
        raw = torch.stack(
            [dx_rot, dy_rot, dh, dvx, dvy, dist, bearing], dim = -1
        )

        return self.norm(self.proj(raw))    # [N, N, D_g]
    
# Stage 2 — Agent–Agent Attention
class AgentAgentAttention(nn.Module):
    """
    Each agent attends to its K nearest neighbours (by ego-relative distance).
    Attention logits are additively biased by  -alpha · dist(i, j) / d_ref,
    reinforcing the physical prior that closer agents are more relevant.
 
    K is chosen per scene from K_TABLE (imported from config).
 
    Args:
        D       : model hidden dimension
        n_heads : number of attention heads
        d_ref   : reference distance for logit normalisation (metres, default 50)
        alpha   : distance-bias strength (default 1.0)
    """
 
    def __init__(
        self,
        D:       int,
        n_heads: int   = 4,
        d_ref:   float = 50.0,
        alpha:   float = 1.0,
    ) -> None:
        super().__init__()
        self.attn   = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.norm   = nn.LayerNorm(D)
        self.d_ref  = d_ref
        self.alpha  = alpha
 
    def forward(
        self,
        repr1:     torch.Tensor,   # [B, N, D]     ego-conditioned from Stage 1
        distances: torch.Tensor,   # [B, N, N]     pairwise distances in metres
        K:         int,            # neighbourhood size from K_TABLE
    ) -> torch.Tensor:             # [B, N, D]
 
        B, N, D = repr1.shape
 
        # 1. Top-K neighbour mask 
        # For each agent i, keep only the K closest agents (excluding itself
        # via the distance matrix diagonal being 0 — agent i is always in
        # its own top-1, so effective social neighbourhood = K-1 peers + self).
        K_safe = min(K, N)
        _, topk_idx = distances.topk(K_safe, dim=-1, largest=False)  # [B, N, K_safe]
 
        # mask[b, i, j] = True  →  agent i must NOT attend to agent j
        mask = torch.ones(B, N, N, dtype=torch.bool, device=repr1.device)
        mask.scatter_(-1, topk_idx, False)   # False = this pair IS allowed
 
        # 2. Distance-bias on logits
        # Positions outside the top-K mask are already set to −∞ via the mask,
        # so the bias only matters for the K retained pairs.
        dist_bias = -self.alpha * distances / self.d_ref   # [B, N, N]
        # For masked-out positions, set to −∞ (softmax will zero them out)
        dist_bias = dist_bias.masked_fill(mask, float("-inf"))
 
        # 3. Expand bias for multi-head attention 
        # nn.MultiheadAttention expects attn_mask of shape
        # [B * n_heads, N, N] or [N, N] when batch_first=True.
        n_heads = self.attn.num_heads
        # [B, 1, N, N] → [B, n_heads, N, N] → [B*n_heads, N, N]
        dist_bias_mh = (
            dist_bias.unsqueeze(1)
                      .expand(-1, n_heads, -1, -1)
                      .reshape(B * n_heads, N, N)
        )
 
        # 4. Multi-head attention + residual 
        out, _ = self.attn(repr1, repr1, repr1, attn_mask=dist_bias_mh)
        return self.norm(repr1 + out)

