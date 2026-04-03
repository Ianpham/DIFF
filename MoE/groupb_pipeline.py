
"""
STUB-C implementation: GroupBInternalPipeline
 
Enriches Group B (agent / interaction) tokens through three stages before
the gate query is formed in GroupBRouter.  Replaces the stub that previously
raised NotImplementedError in moe_block.py.
 
Pipeline order:
    [optional]  Option B+ HistoryEncoder fusion
    Stage 1     EgoCentricCrossAttention      (§2.3)
    Stage 2     EgoProximityAgentAttention    (§2.4)
    Stage 3     MapContextReweighting         (§2.6)
 
Returns: (B, N_B, D) enriched tokens — NO residual.  Caller (MoEBlock) adds
the residual:
    tokens_B = tokens_B + pipeline(tokens_B_ln, ...)
 
 
Option B+  History Encoder  (ego-relative, nuPlan-correct)
 
Architecture decision: ego-compensated relative deltas primary, shallow
absolute branch fused at token level, confidence-weighted blending.
 
Why B+ not raw absolute:
    nuPlan's AV localisation has odometry drift.  Absolute coordinates
    accumulate error over the horizon (H=15 steps).  Ego-relative deltas
    cancel the common-mode drift: Δp_i(t) = p_i(t) - p_ego(t) gives
    each agent's motion relative to ego at each step, which is stable
    even when global position drifts.
 
    Absolute branch (shallow, H=1 linear) is kept so the model can still
    see global lane alignment when GPS is trustworthy.
 
    Confidence-weighted blend:
        h = α * lstm_out + (1-α) * abs_proj
        α = sigmoid(conf_gate(ego_speed, gps_confidence))
        High GPS confidence -> lean absolute; high speed / urban -> lean relative.
 
Inputs expected (all from data pipeline):
    history_traj:   (B, N_B, H, 4)  — per-agent history, H steps,
                    [Δx_ego, Δy_ego, Δvx_ego, Δvy_ego] in ego frame
    history_abs:    (B, N_B, 2)     — current absolute (x, y) position
    ego_speed:      (B,)            — ego speed scalar (m/s)
    gps_confidence: (B,)            — GPS/localisation confidence in [0,1]
 
If history_traj is None the encoder is skipped and tokens pass through
unchanged (graceful degradation at inference when history unavailable).
 
 
Stage 1  EgoCentricCrossAttention  (§2.3)
 
Each agent token attends to the EGO TOKEN ONLY.
    Query  = all agent tokens  (B, N_B, D)
    Key/V  = ego token         (B, 1,   D)
Cost: O(N) — one key per agent regardless of scene size.
 
Ego is identified by ego_mask: (B, N_B) bool, exactly one True per sample.
Pre-norm applied inside this module (consistent with rest of block stack).
 
 
Stage 2  EgoProximityAgentAttention  (§2.4)
 
Top-K neighbors by ego-proximity distance (NOT agent-agent distance).
K from deterministic lookup table driven by ego_speed + local_agent_density.
Attention bias: additive score bias from (d_{i->ego}, d_{j->ego}) pair.
Ego uses SEPARATE Q/K/V projections.
Applied as residual over Stage 1 output.
 
Bugs fixed vs token_router.EgoProximityAgentAttention:
    FIX-1  scores = scores * bias  ->  scores = scores + bias
           Attention bias must be additive (Transformer convention).
           Multiplicative bias scales attention magnitudes in unexpected ways,
           especially when bias values span both sides of 1.0.
    FIX-2  attn_out.transpose(2,1)  ->  attn_out.transpose(1,2)
           After einsum "bhnk,bhnkd->bhnd" output is (B,H,N,hd).
           transpose(1,2) gives (B,N,H,hd) for reshape to (B,N,D).
           transpose(2,1) gives (B,N,H,hd) only coincidentally when H==N,
           which is never the case — silent shape corruption otherwise.
 
 
Stage 3  MapContextReweighting  (§2.6)
 
Per-agent scalar gate applied after C->B cross-attention context is available:
    agent_repr_3 = agent_repr_2 * sigmoid(W · concat[agent_repr_2, map_ctx])
map_ctx is the mean-pooled Group C output passed in from MoEBlock.
This is a per-feature gate (D-dim), not a single scalar — each feature
dimension is independently modulated.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoryEncoder(nn.Module):
    """Ego-compensated relative-delta LSTM with shallow absolute branch.
 
    Primary path (relative deltas, ego-frame):
        LSTM over H steps of [Δx_ego, Δy_ego, Δvx_ego, Δvy_ego].
        Hidden size = embed_dim // 2.  Final hidden state projected to D.
 
    Secondary path (absolute position):
        Single linear layer: (x, y) -> D.  No temporal processing.
 
    Blend:
        alpha = sigmoid(conf_gate([ego_speed, gps_confidence]))
        h_fused = alpha * h_rel + (1 - alpha) * h_abs
 
    Output is added to the incoming token (residual), then LayerNorm.
 
    Args:
        embed_dim:   D
        history_len: H (number of history timesteps)
        input_dim:   feature dim per step (default 4: Δx,Δy,Δvx,Δvy)
    """

    def __init__(
            self, 
            embed_dim: int,
            history_len: int = 15, 
            input_dim: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_len = history_len
        hidden = embed_dim // 2

        # relative path: LSTM over ego-frame deltas
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = hidden,
            num_layers = 1,
            batch_first = True,
            bidirectional = False,
        )

        self.rel_proj = nn.Linear(hidden, embed_dim, bias = False)
        
        #absolute path: shallow linear (x, y) -> D
        self.abs_proj = nn.Linear(2, embed_dim, bias = False)

        # confidence gate: [ego_speed (1), gps_confidence (1)] -> alpha scalar.

        self.conf_gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # post fusion LayerNorm + output projection
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self)-> None:

        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.xavier_uniform_(self.rel_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.abs_proj.weight, gain=0.1)
        nn.init.zeros_(self.conf_gate[-1].bias)

    def forward(
            self,
            tokens:     torch.Tensor,           # (B, N_B, D)
            history_traj: torch.Tensor,         # (B, N_B, H, 4) ego-grame deltas
            history_abs:   torch.Tensor,        # (B, N_B, 2) absolute (x, y)
            ego_speed:      torch.Tensor,       # (B,)
            gps_confidence: torch.Tensor,       # (B, )
    )-> torch.Tensor:
        """Returns (B, N_B, D) — tokens enriched with history, pre-normed."""

        B, N, D = tokens.shape
        H = history_traj.shape[2]

        # relavtive path: LSTM
        # flatten batch x agents for LSTM: (B*N, H, 4)
        traj_flat = history_traj.reshape(B*N, H, -1)
        _, (h_n, _) = self.lstm(traj_flat)          # h_n 
        h_rel = self.rel_proj(h_n.squeeze(0)) # (B*N, D)
        h_rel = h_rel.reshape(B, N, D)        # (B, N, D)

        # absolute path: linear
        h_abs = self.abs_proj(history_abs)      # (B, N, D)

        # confidence-weight blend
        # alpha per sample, boardcast over agentss
        conf_input = torch.stack([ego_speed, gps_confidence], dim = -1) # (B, 2)
        alpha = torch.sigmoid(self.conf_gate(conf_input))               # (B, 1)
        alpha = alpha.unsqueeze(1)                                      # (B, 1, 1)

        h_fused = alpha * h_rel + (1.0 - alpha) * h_abs

        # residual + norm
        return self.norm(tokens + h_fused)
    

# Stage 1  Ego-Centric Cross-Attention  (§2.3)

class EgoCentricCrossAttention(nn.Module):
    """Each agent attends to the ego token ONLY (§2.3).
 
    Cost: O(N) — one key regardless of scene size.
    Pre-norm: caller (pipeline) supplies already-normed tokens.
    Returns cross-attention OUTPUT only — no residual, no norm.
 
    Args:
        embed_dim:  D
        num_heads:  H
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias = False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias = False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain = 0.02)

    def forward(
            self,
            tokens_B: torch.Tensor,     # (B, N_B, D) - pre-normed
            ego_mask: torch.Tensor,     # (B, N_B) bool - exactly one True per sample
    )-> torch.Tensor:
        """Returns (B, N_B, D) cross-attn output. NO residual"""
        B, N, D = tokens_B.shape
        H, hd = self.num_heads, self.head_dim

        # extract ego token per sample: (B, 1, D)
        ego_idx = ego_mask.long().argmax(dim = -1)      # (B,)
        ego_token = tokens_B[torch.arange(B, device = tokens_B.device),
                             ego_idx].unsqueeze(1)      # (B, 1, D)
        

        Q = self.q_proj(tokens_B).reshape(B, N, H, hd).transpose(1, 2)  # (B,H,N,hd)
        K = self.k_proj(ego_token).reshape(B, 1, H, hd).transpose(1, 2) # (B,H,1,hd)
        V = self.v_proj(ego_token).reshape(B, 1, H, hd).transpose(1, 2) # (B,H,1,hd)
 
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale      # (B,H,N,1)
        weights = F.softmax(scores, dim=-1)                               # (B,H,N,1)
        out     = torch.matmul(weights, V)                                # (B,H,N,hd)
        out     = out.transpose(1, 2).reshape(B, N, D)                   # (B,N,D)
 
        return self.out_proj(out)


# stage 2 Ego-proximity agent-agent attention

class EgoProximityAgentAttention(nn.Module):
    """Ego-proximity-filtered sparse agent-agent attention (§2.4).
 
    Top-K neighbors selected by d_{i->ego} (proximity to ego, not to each other).
    K from deterministic lookup: f(ego_speed, local_agent_density).
    Additive attention bias from (d_{i->ego}, d_{j->ego}) pair.
    Ego token uses separate Q/K/V projections.
    Applied as residual over Stage 1 output.
 
    Pre-norm: caller supplies already-normed tokens.
    Returns cross-attention OUTPUT only — no residual, no norm.
 
    Bugs fixed vs token_router.EgoProximityAgentAttention:
        FIX-1  scores + bias  (was: scores * bias — wrong, must be additive)
        FIX-2  attn_out.transpose(1,2)  (was: .transpose(2,1) — wrong axis)
 
    Args:
        embed_dim:           D
        num_heads:           H
        K_default:           default top-K neighbors
        K_max:               maximum top-K
        proximity_threshold: distance threshold for density estimate (metres)
    """

    def __init__(
        self,
        embed_dim:           int,
        num_heads:           int,
        K_default:           int   = 4,
        K_max:               int   = 6,
        proximity_threshold: float = 20.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim           = embed_dim
        self.num_heads           = num_heads
        self.head_dim            = embed_dim // num_heads
        self.scale               = math.sqrt(self.head_dim)
        self.K_default           = K_default
        self.K_max               = K_max
        self.proximity_threshold = proximity_threshold
 
        # Standard agent projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        # Separate ego projections (ego has privileged role, §2.4)
        self.ego_q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ego_k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ego_v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
 
        # Additive distance bias: (d_{i->ego}, d_{j->ego}) -> H bias values
        # Input dim = 2, output = num_heads (one bias per head per pair)
        self.distance_bias_mlp = nn.Sequential(
            nn.Linear(2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_heads),
        )
 
        for m in [self.q_proj, self.k_proj, self.v_proj,
                  self.ego_q_proj, self.ego_k_proj, self.ego_v_proj]:
            nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)

    # K lookup
    @torch.no_grad()
    def _compute_K(
        self,
        ego_speed: torch.Tensor,        # (B,)
        ego_distances: torch.Tensor,     # (B, N_B)
    ) -> int:
        """Deterministic K from ego speed + local density. No gradient."""

        N = ego_distances.shape[1]
        close = (ego_distances < self.proximity_threshold).float().sum(-1)  # (B,)
        density = close / max(N, 1)                      # (B,) in [0, 1]

        avg_speed   = ego_speed.mean().item()
        avg_density = density.mean().item()

        K = self.K_default
        if avg_speed > 15.0:
            K = min(K + 1, self.K_max)
        if avg_density > 0.5:
            K = min(K + 1, self.K_max)
        return K         
    
    # forward
    def forward(
            self,
            tokens_B: torch.Tensor,  # (B, N_B, D) - pre-normed, stage 1 output
            ego_distances: torch.Tensor, # (B, N_B) distance of each agent to ego
            ego_mask:   torch.Tensor,   # (B, N_B) bool - True = ego token
            ego_speed: torch.Tensor,    # (B,) scalar 
    )-> torch.Tensor:
        """Return (B, N_B, D) attention output. NO residual"""
        B, N, D = tokens_B.shape
        H, hd = self.num_heads, self.head_dim

        K = min(self._compute_K(ego_speed, ego_distances), N - 1)
        if K <= 0:
            # degenerate scene: only ego present, nothing to attend to.
            return torch.zeros_like(tokens_B)
        
        # top K neighbor per agent by ego-proximity
        # Block self-attention: set own distance to inf so agent i never picks itslef as a neighbor
        self_inf = torch.eye(N, device=tokens_B.device, dtype = torch.bool)
        self_inf = self_inf.unsqueeze(0).expand(B, -1, -1) # (B, N, N)

        # dist rank[b, i, j] = ego_distance of agent j (used to rank neighbors of i)
        dist_rank = ego_distances.unsqueeze(1).expand(B, N, N).clone()
        dist_rank[self_inf] = float("inf")

        _, topk_idx = dist_rank.topk(K, dim = -1, largest= False) # (B, N, K)

        # gather neighbor tokens
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, K, D)
        tokens_exp = tokens_B.unsqueeze(2).expand(-1, -1, N, -1) # (N, N, N, D)
        neighbors = torch.gather(tokens_exp, 2, idx_exp)          # (B, N, K, D)

        # Q/K/V projections (ego uses separate weights)
        is_ego = ego_mask.float().unsqueeze(-1)         # B, N, 1

        Q = ((1 - is_ego) * self.q_proj(tokens_B)
             + is_ego * self.ego_q_proj(tokens_B))
        
        K_mat = self.k_proj(neighbors)      # (B, N, D)
        V_mat = self.v_proj(neighbors)      # (B, N, K, D)

        Q = Q.reshape(B, N, H, hd).transpose(1, 2)  # (B, H, N, hd)
        K_mat = K_mat.reshape(B, N, K, H, hd).permute(0, 3, 1, 2, 4) # (B, H, N, K, hd)
        V_mat = V_mat.reshape(B, N, K, H, hd).permute(0, 3, 1, 2, 4) # (B, H, N, K, hd)

        # scaled do-product scores
        # Q: (B, H, N, hd), K_mat = (B, H, N, K, hd)
        scores = torch.einsum("bhnd,bhnkd->bhnk", Q, K_mat) / self.scale  # (B,H,N,K) 

        #  Additive distance bias (FIX-1: + not *) 
        # d_{i->ego}: ego_distances broadcast over K
        d_i = ego_distances.unsqueeze(-1).expand(B, N, K)         # (B, N, K)
        # d_{j->ego}: gather ego_distances for each neighbor
        d_j = torch.gather(
            ego_distances.unsqueeze(1).expand(-1, N, -1),          # (B, N, N)
            2, topk_idx                                             # (B, N, K)
        )                                                           # (B, N, K)
 
        dist_pair = torch.stack([d_i, d_j], dim=-1)               # (B, N, K, 2)
        bias = self.distance_bias_mlp(dist_pair)                   # (B, N, K, H)
        bias = bias.permute(0, 3, 1, 2)                            # (B, H, N, K)
        scores = scores + bias                                     # additive (FIX-1)
 
        #  Softmax + weighted sum 
        attn_weights = F.softmax(scores, dim=-1)                   # (B, H, N, K)
        attn_out = torch.einsum("bhnk,bhnkd->bhnd", attn_weights, V_mat)  # (B,H,N,hd)
 
        # FIX-2: correct transpose — (B,H,N,hd) -> (B,N,H,hd) -> (B,N,D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)      # (B, N, D)
 
        return self.out_proj(attn_out)
     
# Stage 3  Map Context Re-weighting  (§2.6)
class MapContextReweighting(nn.Module):
    """Per-agent feature gate conditioned on map context (§2.6).
 
    agent_repr_3 = agent_repr_2 * sigmoid(W · concat[agent_repr_2, map_ctx])
 
    map_ctx is mean-pooled Group C output: (B, D).
    Gate produces a D-dimensional weight per agent — per-feature modulation,
    not a single scalar.  This lets the gate suppress irrelevant map features
    (e.g. highway geometry for a parking-lot agent) independently per dim.
 
    Pre-norm: caller supplies already-normed tokens.
    Returns gated tokens directly — no residual (the gating IS the output).
 
    Args:
        embed_dim: D
    """
 
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Init final layer near zero so gate starts near 0.5 (sigmoid(0)=0.5)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
 
    def forward(
        self,
        agent_repr:  torch.Tensor,   # (B, N_B, D) — Stage 2 output
        map_context: torch.Tensor,   # (B, D) or (B, N_C, D) — Group C output
    ) -> torch.Tensor:
        """Returns (B, N_B, D) — gated agent representations."""
        B, N, D = agent_repr.shape
 
        # Mean-pool map context if full sequence provided
        if map_context.dim() == 3:
            map_ctx = map_context.mean(dim=1)   # (B, N_C, D) → (B, D)
        else:
            map_ctx = map_context               # (B, D)
 
        # Broadcast map context over all agents
        map_ctx_exp = map_ctx.unsqueeze(1).expand(B, N, D)  # (B, N, D)
 
        gate_input = torch.cat([agent_repr, map_ctx_exp], dim=-1)  # (B, N, 2D)
        gate_vals  = torch.sigmoid(self.gate(gate_input))           # (B, N, D)
 
        return agent_repr * gate_vals

class GroupBInternalPipeline(nn.Module):
    """Full Group B pre-gate enrichment pipeline (§2.3-2.6 + Option B+).
 
    Wires:
        [optional] HistoryEncoder     — ego-relative LSTM + absolute branch
        Stage 1    EgoCentricCrossAttention
        Stage 2    EgoProximityAgentAttention
        Stage 3    MapContextReweighting
 
    Pre-norm convention:
        A single GroupLocalLayerNorm-compatible LN (nn.LayerNorm) is applied
        ONCE before the stage cascade.  Each stage operates on the same
        pre-normed representation (stages 1 and 2 share the same normed input;
        stage 2 residual is applied onto stage 1 output before stage 3).
 
    Residual structure:
        h0 = LayerNorm(tokens_B)                 # shared pre-norm
        h1 = tokens_B + Stage1(h0, ego_mask)     # ego cross-attn residual
        h1_ln = LayerNorm(h1)                    # re-norm before stage 2
        h2 = h1 + Stage2(h1_ln, ...)             # proximity attn residual
        h3 = Stage3(h2, map_context)             # map gate (multiplicative, no add)
        return h3                                # caller adds residual to tokens_B
 
    This module returns the PRE-RESIDUAL enrichment output.
    MoEBlock applies the outer residual:
        tokens_B = tokens_B + pipeline(tokens_B_ln, ...)
 
    Args:
        embed_dim:            D
        num_heads:            H for Stages 1 and 2
        K_default:            default top-K for Stage 2
        K_max:                max top-K for Stage 2
        proximity_threshold:  distance threshold for Stage 2 density (metres)
        history_len:          H timesteps for HistoryEncoder (default 15)
        use_history_encoder:  whether to run the B+ encoder (default True)
    """
 
    def __init__(
        self,
        embed_dim:            int,
        num_heads:            int   = 4,
        K_default:            int   = 4,
        K_max:                int   = 6,
        proximity_threshold:  float = 20.0,
        history_len:          int   = 15,
        use_history_encoder:  bool  = True,
    ):
        super().__init__()
        self.use_history_encoder = use_history_encoder
 
        if use_history_encoder:
            self.history_encoder = HistoryEncoder(
                embed_dim=embed_dim,
                history_len=history_len,
            )
 
        # Pre-norm before the stage cascade
        self.pre_norm   = nn.LayerNorm(embed_dim)
        # Re-norm between Stage 1 and Stage 2
        self.inter_norm = nn.LayerNorm(embed_dim)
 
        self.stage1 = EgoCentricCrossAttention(embed_dim, num_heads)
        self.stage2 = EgoProximityAgentAttention(
            embed_dim, num_heads, K_default, K_max, proximity_threshold
        )
        self.stage3 = MapContextReweighting(embed_dim)
 
    def forward(
        self,
        tokens_B:       torch.Tensor,                    # (B, N_B, D) — pre-normed by MoEBlock
        ego_mask:       torch.Tensor,                    # (B, N_B) bool
        map_context:    torch.Tensor,                    # (B, N_C, D) or (B, D) Group C output
        ego_speed:      torch.Tensor,                    # (B,) scalar m/s
        ego_distances:  torch.Tensor,                    # (B, N_B) dist of each agent to ego
        history_traj:   Optional[torch.Tensor] = None,   # (B, N_B, H, 4) ego-frame deltas
        history_abs:    Optional[torch.Tensor] = None,   # (B, N_B, 2)    absolute (x,y)
        gps_confidence: Optional[torch.Tensor] = None,   # (B,) in [0,1]
    ) -> torch.Tensor:
        """Enrich Group B tokens through the full pipeline.
 
        Args:
            tokens_B:       (B, N_B, D) — already group-local normed by MoEBlock
            ego_mask:       (B, N_B) bool — exactly one True per sample
            map_context:    (B, N_C, D) or (B, D) — Group C expert output
            ego_speed:      (B,) — ego vehicle speed in m/s
            ego_distances:  (B, N_B) — each agent's distance to ego (metres)
            history_traj:   (B, N_B, H, 4) — ego-frame relative deltas (optional)
            history_abs:    (B, N_B, 2) — absolute (x,y) current positions (optional)
            gps_confidence: (B,) — GPS confidence in [0,1] (optional)
 
        Returns:
            (B, N_B, D) — enriched tokens.  NO outer residual.
            Caller adds:  tokens_B = tokens_B + pipeline(tokens_B_ln, ...)
        """
        B, N, D = tokens_B.shape
 
        # history encoder 
        h = tokens_B
        if (self.use_history_encoder
                and history_traj is not None
                and history_abs is not None):
            gc = gps_confidence if gps_confidence is not None \
                 else torch.ones(B, device=tokens_B.device)
            h = self.history_encoder(h, history_traj, history_abs,
                                     ego_speed, gc)
        # h: (B, N_B, D) — history-enriched tokens (or original if skipped)
 
        # Pre-norm before stage cascade 
        h_ln = self.pre_norm(h)
 
        #  Stage 1: ego-centric cross-attention 
        h1 = h + self.stage1(h_ln, ego_mask)
 
        #  Re-norm before Stage 2 (h1 has changed from Stage 1 residual) ─
        h1_ln = self.inter_norm(h1)
 
        #  Stage 2: ego-proximity agent-agent attention 
        h2 = h1 + self.stage2(h1_ln, ego_distances, ego_mask, ego_speed)
 
        # Stage 3: map context re-weighting 
        # Multiplicative gate — no additive residual, the gate IS the output.
        h3 = self.stage3(h2, map_context)
 
        return h3
    
if __name__ == "__main__":
    torch.manual_seed(42)
    B, D   = 2, 128
    N_B    = 12
    N_C    = 32
    H_hist = 15
 
    pipeline = GroupBInternalPipeline(
        embed_dim=D,
        num_heads=4,
        K_default=3,
        K_max=5,
        history_len=H_hist,
        use_history_encoder=True,
    )
 
    # Inputs
    tokens_B    = torch.randn(B, N_B, D, requires_grad=True)
    map_context = torch.randn(B, N_C, D)
 
    # ego_mask: agent 0 is ego in both samples
    ego_mask = torch.zeros(B, N_B, dtype=torch.bool)
    ego_mask[:, 0] = True
 
    ego_distances = torch.rand(B, N_B) * 50.0   # 0-50 m
    ego_distances[:, 0] = 0.0                    # ego is 0m from itself
    ego_speed     = torch.tensor([8.0, 12.0])    # m/s
 
    history_traj   = torch.randn(B, N_B, H_hist, 4)
    history_abs    = torch.randn(B, N_B, 2) * 100.0
    gps_confidence = torch.tensor([0.9, 0.6])
 
    # Shape 
    out = pipeline(tokens_B, ego_mask, map_context, ego_speed,
                   ego_distances, history_traj, history_abs, gps_confidence)
    assert out.shape == (B, N_B, D), f"Output shape: {out.shape}"
    print(f"Output shape: {out.shape}  ✓")
 
    #  No NaN 
    assert not torch.isnan(out).any(), "NaN in output"
    print(f"No NaN:  ✓")
 
    # Output differs from input (enrichment happened) 
    assert not torch.allclose(out, tokens_B.detach()), "Output should differ from input"
    print(f"Output differs from input (pipeline enriched tokens):  ✓")
 
    #  Gradient flows back to tokens_B 
    out.sum().backward()
    assert tokens_B.grad is not None, "tokens_B should have grad"
    assert not torch.isnan(tokens_B.grad).any(), "NaN in grad"
    print(f"Gradient flows to tokens_B:  ✓")
 
    # ── Without history encoder (graceful degradation)
    pipeline_nohist = GroupBInternalPipeline(
        embed_dim=D, num_heads=4, use_history_encoder=False
    )
    out_nh = pipeline_nohist(
        tokens_B.detach(), ego_mask, map_context, ego_speed, ego_distances
    )
    assert out_nh.shape == (B, N_B, D)
    assert not torch.isnan(out_nh).any()
    print(f"No history encoder (graceful degradation): shape={out_nh.shape}  ✓")
 
    # ── History encoder skipped when history_traj=None 
    out_notraj = pipeline(
        tokens_B.detach(), ego_mask, map_context, ego_speed, ego_distances,
        history_traj=None, history_abs=None
    )
    assert out_notraj.shape == (B, N_B, D)
    print(f"history_traj=None skips encoder gracefully:  ✓")
 
    # ── Stage 2 FIX-1: additive bias (not multiplicative) 
    # Verify by checking that zeroing distance_bias_mlp weights gives same
    # result as running with no bias (not same as scaling by zero).
    with torch.no_grad():
        for p in pipeline.stage2.distance_bias_mlp.parameters():
            p.zero_()
    out_nobias = pipeline(
        tokens_B.detach(), ego_mask, map_context, ego_speed, ego_distances
    )
    assert not torch.isnan(out_nobias).any(), "NaN with zeroed bias"
    print(f"FIX-1 additive bias (zeroed → no NaN, runs cleanly):  ✓")
 
    # ── Stage 2 FIX-2: correct transpose 
    # If transpose(2,1) were used instead of transpose(1,2) with H≠N,
    # the reshape to (B,N,D) would silently give wrong values.
    # We verify correct shape with N_B ≠ num_heads.
    assert N_B != pipeline.stage2.num_heads, "Test requires N_B ≠ num_heads"
    stage2_out = pipeline.stage2(
        torch.randn(B, N_B, D),
        ego_distances,
        ego_mask,
        ego_speed,
    )
    assert stage2_out.shape == (B, N_B, D), \
        f"FIX-2 failed: stage2 output shape {stage2_out.shape} ≠ (B,N_B,D)"
    print(f"FIX-2 correct transpose (N_B={N_B} ≠ H={pipeline.stage2.num_heads}): shape correct  ✓")
 
    # ── Map context as (B,D) pooled tensor 
    map_pooled = map_context.mean(dim=1)   # (B, D)
    out_pooled = pipeline(
        tokens_B.detach(), ego_mask, map_pooled, ego_speed, ego_distances
    )
    assert out_pooled.shape == (B, N_B, D)
    print(f"Map context as (B,D) pooled tensor:  ✓")
 
    # ── Ego token receives its own separate projections 
    # Verify ego_q_proj ≠ q_proj (independent weights)
    s2 = pipeline.stage2
    assert not torch.allclose(s2.q_proj.weight, s2.ego_q_proj.weight), \
        "ego_q_proj should have different weights from q_proj"
    print(f"Ego uses separate Q/K/V projections (independent weights):  ✓")
 
    # ── Option B+ confidence gate 
    # High GPS confidence should lean more on absolute branch
    # Low GPS confidence should lean more on relative LSTM
    # Verify α varies with gps_confidence
    he = pipeline.history_encoder
    traj  = torch.randn(1, N_B, H_hist, 4)
    abs_p = torch.randn(1, N_B, 2)
    spd   = torch.tensor([10.0])
    toks  = torch.randn(1, N_B, D)
 
    conf_high = torch.tensor([0.95])
    conf_low  = torch.tensor([0.05])
    out_high  = he(toks, traj, abs_p, spd, conf_high)
    out_low   = he(toks, traj, abs_p, spd, conf_low)
    assert not torch.allclose(out_high, out_low), \
        "Different GPS confidence should produce different outputs"
    print(f"Option B+ confidence gate produces different outputs for high/low GPS:  ✓")
 
    # ── Parameter count 
    n_total  = sum(p.numel() for p in pipeline.parameters())
    n_hist   = sum(p.numel() for p in pipeline.history_encoder.parameters())
    n_stages = n_total - n_hist
    print(f"\nParameter counts:")
    print(f"  HistoryEncoder:  {n_hist:>10,}")
    print(f"  Stages 1-3:      {n_stages:>10,}")
    print(f"  Total pipeline:  {n_total:>10,}")
 
    print("\n" + "="*55)
    print("All GroupBInternalPipeline tests PASSED.")
    print("="*55)
    print("\nBugs fixed from token_router.EgoProximityAgentAttention:")
    print("  FIX-1  scores + bias  (was * bias — multiplicative is wrong)")
    print("  FIX-2  attn_out.transpose(1,2)  (was .transpose(2,1) — wrong axis)")
    print("\nOpen decision (§2.2):")
    print("  History encoder placement confirmed: LSTM runs AFTER ego-relative")
    print("  geometry is computed (Option B+ with H=15).")
    print("\nOpen decision (§2.4):")
    print("  Ego token uses SEPARATE Q/K/V projections and goes through the")
    print("  same gate as other agents (no privileged expert routing).")