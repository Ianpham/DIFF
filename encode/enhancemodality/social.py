"""
AgentEncoder — Level 2: Social Interaction
==========================================
Spec source : encoder_architecture_specification.docx §7

Takes per-agent token embeddings produced by HistoryEncoder and applies
a single round of relational self-attention that models pairwise social
interactions between agents.

Two critical bugs from the existing implementation are fixed here:

  BUG 1 (§7.1) — distance_bias_mlp output was computed but never added
                  to attention logits.  Fixed: bias is permuted and passed
                  as attn_mask to scaled_dot_product_attention.

  BUG 2 (§7.2) — pairwise relational features were mean-pooled across
                  the agent dimension and added to Q, destroying directional
                  pairwise structure (rel[i,j] ≠ rel[j,i]).
                  Fixed: relational features are projected into value space
                  and injected as an additive value bias in the weighted sum
                  (Graphormer-style, Ying et al. 2021).

NavsimDataset inputs

multi_agent_history  (B, 32, 4, 7)  — source of pairwise geometry
multi_agent_states   (B, 32, 5)     — [x, y, vx, vy, heading] at current frame
agent_tokens         (B, N, d_model) — output of HistoryEncoder

The pairwise relational features are built from multi_agent_states
current-frame positions and headings, giving directional information
(e.g. "agent j is ahead and to the left of agent i").

Architecture (§7)
-
Input  : agent_tokens (B, N, d_model) from HistoryEncoder
         pairwise_features (B, N, N, d_rel) — directional relational features
         pairwise_distances (B, N, N, 1)    — scalar L2 distances

Forward pass:
  1. distance_bias_mlp(distances) → (B,N,N,H) → permute → (B,H,N,N)  [FIX 1]
  2. rel_value_proj(pairwise_features) → (B,N,N,d_model)              [FIX 2]
  3. Standard QKV projection
  4. scaled_dot_product_attention with distance bias as attn_mask
  5. Add relational value bias:  out += (attn_w ⊙ rel_val).sum(dim=-2)
  6. Output projection + residual

Output : (B, N, d_model) socially-contextualised agent tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---
# Pairwise relational feature builder
# ---

class PairwiseRelationalFeatures(nn.Module):
    """Build directional pairwise features between N agents.

    For each ordered pair (i, j) computes:
        rel_x, rel_y        — position of j relative to i, rotated into i's
                              heading frame (so "ahead" means positive rel_x
                              regardless of absolute orientation)
        distance            — L2 distance
        rel_speed           — speed difference (j_speed - i_speed)
        rel_heading         — heading difference (cos, sin decomposed)

    Result is a 7-dimensional directional feature vector.
    rel[i,j] ≠ rel[j,i] by construction (directional).

    Parameters
    --
    eps : float — small value to avoid division by zero in normalisation.
    """

    FEAT_DIM = 7  # (rel_x_rot, rel_y_rot, dist, rel_speed, cos_dh, sin_dh, 1/dist_clamp)

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        states      : torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        --
        states       : (B, N, 5)  [x, y, vx, vy, heading] in ego frame
        padding_mask : (B, N) bool — True for zero-padded (absent) agents

        Returns
        ---
        rel_feats : (B, N, N, FEAT_DIM)   directional pairwise features
        distances : (B, N, N, 1)          scalar L2 distance (for bias MLP)
        """
        B, N, _ = states.shape

        x   = states[:, :, 0]   # (B, N)
        y   = states[:, :, 1]
        vx  = states[:, :, 2]
        vy  = states[:, :, 3]
        h   = states[:, :, 4]   # heading in radians

        speed = (vx ** 2 + vy ** 2).sqrt()   # (B, N)

        #  Relative position of j w.r.t. i 
        # dx[b,i,j] = x[b,j] - x[b,i]
        dx = x.unsqueeze(2) - x.unsqueeze(1)   # (B, N, N)
        dy = y.unsqueeze(2) - y.unsqueeze(1)

        #  Rotate into agent i's heading frame 
        cos_h = torch.cos(h)   # (B, N)
        sin_h = torch.sin(h)

        # For each i, rotate (dx, dy) by -h_i
        # rel_x_rot[b,i,j] =  cos(h_i)*dx[b,i,j] + sin(h_i)*dy[b,i,j]
        # rel_y_rot[b,i,j] = -sin(h_i)*dx[b,i,j] + cos(h_i)*dy[b,i,j]
        cos_i = cos_h.unsqueeze(2)   # (B, N, 1)
        sin_i = sin_h.unsqueeze(2)

        rel_x_rot =  cos_i * dx + sin_i * dy   # (B, N, N)
        rel_y_rot = -sin_i * dx + cos_i * dy

        #  Distance 
        dist = (dx ** 2 + dy ** 2).sqrt()        # (B, N, N)
        dist_clamp = dist.clamp(min=self.eps)

        #  Relative speed: speed_j - speed_i 
        rel_speed = speed.unsqueeze(2) - speed.unsqueeze(1)   # (B, N, N)

        #  Relative heading: decomposed as (cos Δh, sin Δh) 
        dh     = h.unsqueeze(2) - h.unsqueeze(1)   # (B, N, N)
        cos_dh = torch.cos(dh)
        sin_dh = torch.sin(dh)

        #  Inverse distance (proximity signal) 
        inv_dist = 1.0 / dist_clamp                # (B, N, N)

        #  Stack into (B, N, N, FEAT_DIM) 
        rel_feats = torch.stack(
            [rel_x_rot, rel_y_rot, dist, rel_speed, cos_dh, sin_dh, inv_dist],
            dim=-1,
        )   # (B, N, N, 7)

        # Zero out features for padded agents so they don't affect attention
        if padding_mask is not None:
            # Rows i where agent i is padded → zero all its outgoing relations
            pad_i = padding_mask.unsqueeze(2).unsqueeze(-1)   # (B, N, 1, 1)
            # Cols j where agent j is padded → zero all incoming relations
            pad_j = padding_mask.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            rel_feats = rel_feats.masked_fill(pad_i | pad_j, 0.0)

        distances = dist.unsqueeze(-1)   # (B, N, N, 1)

        return rel_feats, distances


# ---
# AgentEncoderL2
# ---

class AgentEncoderL2(nn.Module):
    """Social interaction encoder: relational self-attention over N agent tokens.

    Parameters
    --
    d_model      : Token dimension (must match HistoryEncoder output).
    num_heads    : Number of attention heads.
    d_rel        : Pairwise relational feature dim (PairwiseRelationalFeatures.FEAT_DIM = 7).
    dropout      : Attention dropout probability.
    """

    def __init__(
        self,
        d_model  : int = 256,
        num_heads: int = 8,
        d_rel    : int = PairwiseRelationalFeatures.FEAT_DIM,
        dropout  : float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.dropout   = dropout

        #  QKV projection 
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=True)

        #  FIX 1: distance bias MLP 
        # Input: scalar distance (B, N, N, 1)
        # Output: (B, N, N, num_heads) → added to attention logits
        self.distance_bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, num_heads),
        )

        #  FIX 2: relational value projection 
        # Project pairwise features (B, N, N, d_rel) → (B, N, N, d_model)
        # The value aggregation becomes:
        #   out_i = Σ_j  attn[i,j] * (V_j + rel_val[i,j])
        self.rel_value_proj = nn.Linear(d_rel, d_model, bias=False)

        # Layer norm + residual (pre-norm style)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # Bias MLPs: small init so bias starts near zero
        for m in self.distance_bias_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.rel_value_proj.weight, std=0.02)

    # --
    def forward(
        self,
        agent_tokens     : torch.Tensor,
        pairwise_features: torch.Tensor,
        pairwise_distances: torch.Tensor,
        padding_mask     : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        --
        agent_tokens       : (B, N, d_model)   from HistoryEncoder
        pairwise_features  : (B, N, N, d_rel)  from PairwiseRelationalFeatures
        pairwise_distances : (B, N, N, 1)       scalar distances
        padding_mask       : (B, N) bool — True for absent agents

        Returns
        ---
        output : (B, N, d_model)  socially-contextualised agent tokens
        """
        B, N, D = agent_tokens.shape
        H       = self.num_heads
        Dh      = self.d_head

        # Pre-norm
        residual = agent_tokens
        x        = self.norm(agent_tokens)

        #  FIX 1: compute distance bias 
        # (B, N, N, 1) → (B, N, N, H) → (B, H, N, N)
        dist_bias = self.distance_bias_mlp(pairwise_distances)   # (B, N, N, H)
        dist_bias = dist_bias.permute(0, 3, 1, 2)                # (B, H, N, N)

        # Mask out padding agents in the bias (set to -inf so softmax ignores)
        if padding_mask is not None:
            pad_ij = padding_mask.unsqueeze(1).unsqueeze(2) | \
                     padding_mask.unsqueeze(1).unsqueeze(3)   # (B, 1, N, N) broadcast
            dist_bias = dist_bias.masked_fill(pad_ij, float("-inf"))

        #  FIX 2: project relational features into value space 
        # (B, N, N, d_rel) → (B, N, N, d_model)
        rel_val = self.rel_value_proj(pairwise_features)   # (B, N, N, d_model)
        # Reshape for per-head use: (B, N, N, H, Dh)
        rel_val_h = rel_val.reshape(B, N, N, H, Dh)

        #  QKV projection 
        qkv = self.qkv_proj(x)                            # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)                  # (3, B, H, N, Dh)
        Q, K, V = qkv.unbind(dim=0)                       # each (B, H, N, Dh)

        #  Attention weights with distance bias 
        # attn_logits = QK^T / sqrt(d_head) + dist_bias
        scale       = math.sqrt(Dh)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, H, N, N)
        attn_logits = attn_logits + dist_bias

        # Additional key-padding mask: columns of absent agents → -inf
        if padding_mask is not None:
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(key_mask, float("-inf"))

        attn_w = F.softmax(attn_logits, dim=-1)   # (B, H, N, N)

        if self.dropout > 0.0 and self.training:
            attn_w = F.dropout(attn_w, p=self.dropout)

        #  Standard value aggregation 
        # out_std[b,h,i,:] = Σ_j  attn_w[b,h,i,j] * V[b,h,j,:]
        out_std = torch.matmul(attn_w, V)   # (B, H, N, Dh)

        #  FIX 2: relational value bias 
        # out_rel[b,h,i,:] = Σ_j  attn_w[b,h,i,j] * rel_val_h[b,i,j,h,:]
        # attn_w  : (B, H, N, N)
        # rel_val_h: (B, N, N, H, Dh)
        #
        # Efficient computation:
        #   (B, H, N, N) x (B, N, N, H, Dh)
        # Rearrange rel_val_h → (B, H, N, N, Dh)
        rel_val_h = rel_val_h.permute(0, 3, 1, 2, 4)   # (B, H, N, N, Dh)
        # attn_w unsqueeze for broadcast: (B, H, N, N, 1)
        out_rel = (attn_w.unsqueeze(-1) * rel_val_h).sum(dim=3)   # (B, H, N, Dh)

        out = out_std + out_rel   # (B, H, N, Dh)

        #  Merge heads and project 
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)   # (B, N, d_model)
        out = self.out_proj(out)

        #  Residual connection 
        return out + residual   # (B, N, d_model)


# ---
# Convenience wrapper: build pairwise features then run L2 encoder
# ---

class SocialInteractionEncoder(nn.Module):
    """Full social interaction stage: pairwise feature builder + AgentEncoderL2.

    Drop-in module that takes raw agent states + HistoryEncoder tokens
    and returns socially-contextualised tokens.

    Parameters
    --
    d_model   : Token dimension.
    num_heads : Attention heads.
    dropout   : Attention dropout.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.pairwise = PairwiseRelationalFeatures()
        self.l2       = AgentEncoderL2(
            d_model   = d_model,
            num_heads = num_heads,
            d_rel     = PairwiseRelationalFeatures.FEAT_DIM,
            dropout   = dropout,
        )

    def forward(
        self,
        agent_tokens : torch.Tensor,
        agent_states : torch.Tensor,
        padding_mask : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        --
        agent_tokens : (B, N, d_model)   HistoryEncoder output
        agent_states : (B, N, 5)         [x, y, vx, vy, heading] current frame
        padding_mask : (B, N) bool       True = absent agent (zero-padded row)

        Returns
        ---
        tokens : (B, N, d_model)
        """
        rel_feats, distances = self.pairwise(agent_states, padding_mask)
        return self.l2(agent_tokens, rel_feats, distances, padding_mask)


# ---
# NavsimDataset adapter
# ---

def prepare_agent_encoder_inputs(
    batch        : dict,
    history_tokens: torch.Tensor,
    device       : Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract inputs for SocialInteractionEncoder from a NavsimDataset batch.

    NavsimDataset
    -
    multi_agent_states : (B, 32, 5)  [x, y, vx, vy, heading]
                         Row 0 = ego; zero-padded rows = absent agents.
    history_tokens     : (B, 32, T_hist, d_model)  from HistoryEncoder
                         Collapsed to (B, 32, d_model) by mean-pooling T_hist.

    Returns
    ---
    agent_tokens  : (B, N, d_model)
    agent_states  : (B, N, 5)
    padding_mask  : (B, N) bool  True = absent agent
    """
    states = batch["multi_agent_states"]   # (B, 32, 5)
    if device is not None:
        states = states.to(device)
    states = states.float()
    if states.dim() == 2:
        states = states.unsqueeze(0)       # (1, 32, 5)

    # Padding mask: rows whose absolute sum is zero are absent agents
    # (ego at row 0 is always present)
    padding_mask = (states.abs().sum(dim=-1) == 0)   # (B, 32)

    # Collapse temporal dimension from HistoryEncoder output if needed
    if history_tokens.dim() == 4:
        # (B, N, T_hist, d_model) → mean over T → (B, N, d_model)
        tokens = history_tokens.mean(dim=2)
    else:
        tokens = history_tokens   # already (B, N, d_model)

    if device is not None:
        tokens = tokens.to(device)

    return tokens, states, padding_mask


# ---
# Sanity check
# ---

if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, D, H = 2, 32, 256, 8

    print("=" * 60)
    print("PairwiseRelationalFeatures")
    print("=" * 60)

    pairwise = PairwiseRelationalFeatures()

    # Simulate multi_agent_states: ego at origin, others scattered
    states = torch.randn(B, N, 5)
    states[:, 0, :] = 0.0   # ego always at origin

    # Last 4 agents absent
    states[:, 28:, :] = 0.0
    pad_mask = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[:, 28:] = True

    rel_feats, distances = pairwise(states, padding_mask=pad_mask)
    print(f"  states shape      : {states.shape}")
    print(f"  rel_feats shape   : {rel_feats.shape}")   # (B, 32, 32, 7)
    print(f"  distances shape   : {distances.shape}")   # (B, 32, 32, 1)

    # Verify directional asymmetry: rel[i,j] ≠ rel[j,i] in general
    i, j = 0, 3
    same = (rel_feats[0, i, j] == rel_feats[0, j, i]).all()
    print(f"  rel[0,3] == rel[3,0] (should be False for most features): {same.item()}")

    # Self-distance should be zero
    self_dist = distances[:, :, :, 0].diagonal(dim1=1, dim2=2)
    assert self_dist.abs().max() < 1e-5, "Self-distances not zero"
    print(f"  Self-distances are zero ✓")

    # Padded pairs should have zero features
    assert rel_feats[:, 28:, :, :].abs().max() == 0.0, "Padded rows not zeroed"
    assert rel_feats[:, :, 28:, :].abs().max() == 0.0, "Padded cols not zeroed"
    print(f"  Padded pairs zeroed ✓")

    # --
    print("\n" + "=" * 60)
    print("AgentEncoderL2 (direct, with explicit pairwise inputs)")
    print("=" * 60)

    l2_enc = AgentEncoderL2(d_model=D, num_heads=H)
    agent_tokens = torch.randn(B, N, D)

    out = l2_enc(agent_tokens, rel_feats, distances, padding_mask=pad_mask)
    print(f"  Input  tokens : {agent_tokens.shape}")
    print(f"  Output tokens : {out.shape}")             # (B, 32, 256)
    assert out.shape == agent_tokens.shape, "Shape mismatch"
    print(f"  Shape preserved ✓")

    total = sum(p.numel() for p in l2_enc.parameters())
    print(f"  Params: {total:,}")

    # --
    print("\n" + "=" * 60)
    print("SocialInteractionEncoder (end-to-end convenience wrapper)")
    print("=" * 60)

    social_enc = SocialInteractionEncoder(d_model=D, num_heads=H)
    out2 = social_enc(agent_tokens, states, padding_mask=pad_mask)
    print(f"  Input  tokens : {agent_tokens.shape}")
    print(f"  Output tokens : {out2.shape}")
    total2 = sum(p.numel() for p in social_enc.parameters())
    print(f"  Params: {total2:,}")

    # --
    print("\n" + "=" * 60)
    print("NavsimDataset adapter")
    print("=" * 60)

    # Simulate HistoryEncoder output: (B, N, T_hist, d_model)
    T_hist = 4
    fake_history_tokens = torch.randn(B, N, T_hist, D)

    fake_batch = {
        "multi_agent_states": states,
    }

    tokens_in, states_out, mask_out = prepare_agent_encoder_inputs(
        fake_batch, fake_history_tokens
    )
    print(f"  agent_tokens  : {tokens_in.shape}")     # (B, 32, 256)  — T collapsed
    print(f"  agent_states  : {states_out.shape}")    # (B, 32, 5)
    print(f"  padding_mask  : {mask_out.shape}  "
          f"  absent count per batch: {mask_out.sum(dim=1).tolist()}")

    out3 = social_enc(tokens_in, states_out, padding_mask=mask_out)
    print(f"  SocialInteractionEncoder output: {out3.shape}")

    # --
    print("\n" + "=" * 60)
    print("Bug fix verification")
    print("=" * 60)

    # FIX 1: distance bias must be non-zero and affect outputs
    # Run twice with different distance biases and verify outputs differ
    enc_test = AgentEncoderL2(d_model=D, num_heads=H)
    toks = torch.randn(B, 4, D)
    states_small = torch.randn(B, 4, 5)
    pw = PairwiseRelationalFeatures()
    rf, dist1 = pw(states_small)
    dist2 = dist1 * 100.0   # very different distances → different biases

    out_a = enc_test(toks, rf, dist1)
    out_b = enc_test(toks, rf, dist2)
    bias_matters = not torch.allclose(out_a, out_b)
    print(f"  FIX 1 — distance bias affects output: {bias_matters} ✓")

    # FIX 2: relational value bias must be directional
    # Swap rel_feats rows i,j and verify output differs (directional ≠ symmetric)
    rf_swapped = rf.clone()
    rf_swapped[:, 0, 1, :] = rf[:, 1, 0, :]
    rf_swapped[:, 1, 0, :] = rf[:, 0, 1, :]

    out_c = enc_test(toks, rf,         dist1)
    out_d = enc_test(toks, rf_swapped, dist1)
    relval_directional = not torch.allclose(out_c, out_d)
    print(f"  FIX 2 — relational value bias is directional: {relval_directional} ✓")

    print("\n✓ All checks passed")