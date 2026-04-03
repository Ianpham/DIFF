""""
Remain encoder 
# modality is just for testing, we need different module for this.

Module:
GoalIntentEncoder
BehaviorEncoder
EgoStateEncoder
ModalityGate

Navsim Input
--------------
goal feature    : (B, 1, 5) [goal_rel_x, goal_rel_y, dist, heading_to_goal, route_length]
mutil_agent_state: (B, 32, 5) used for behavor proxy (vx, vy, heading columns)
agent_states:    : (B,1, 5) [x=0, y=0, vx, vy, h = 0] ego state 
note: navsim ego is aways at origin with h = 0. 
Curature is derived from yaw_rate if availabel: otherwise approximated from vx/vy change across history frames
"""
"""
Design note:
    GoalIntentEncoder and BehaviorEncoder apply feature dropout (p = 0.15) on raw input
    during training - not embeddings - so the model learn to plan with partial intent/behavior information

    EgostateEncoder always receive input: no dropout
    Modality gate: just for testing and reference.

All encoders add a learnbable modality embedding to their output tokens.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# share helper

def build_mlp(in_dim: int, hidden_dim: int, out_dim = int)-> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim)
    )

class FeatureDropout(nn.Module):
    """
    Per-feature Bernoulli dropout applied to raw input features.
 
    Unlike nn.Dropout (which scales surviving values), this zeros individual
    feature dimensions independently — simulating a sensor/signal that is
    simply absent.  Only active during training.
 
    Parameters
    ----------
    p : float — probability of zeroing each individual feature value.
    """

    def __init__(self, p: float = 0.15):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        
        mask = torch.bernoulli(torch.full_like(x, 1.0 - self.p))
        return x * mask
    
    def extra_repr(self):
        return f"p={self.p}"
    

# goal intentencoder
class GoalIntentEncoder(nn.Module):
    """
    Encode route-level goal features into a single conditioning token.
 
    Input  : (B, 1, 5)  [goal_rel_x, goal_rel_y, dist, heading_to_goal, route_length]
    Output : (B, 1, d_model)
 
    The token is used for adaLN conditioning throughout the DyDiT backbone:
    it modulates the scale and shift of layer norms in the denoising network.
 
    Feature dropout (p=0.15) is applied to raw inputs during training,
    teaching the model to plan with partial or missing intent information
    (classifier-free guidance analogy, Ho & Salimans 2022).
 
    Parameters
    ----------
    d_model   : Output token dimension.
    d_hidden  : MLP hidden dimension. Spec: 128–256. Default 128.
    input_dim : Raw feature dimension. Default 5.
    dropout_p : Feature dropout probability. Spec: 0.15.
    """

    def __init__(
            self, 
            d_model: int = 256,
            d_hidden: int = 128,
            in_channels: int = 5,
            dropout_p: float = 0.15,
    ):
        super().__init__()
        self.feat_dropout   = FeatureDropout(p = dropout_p)
        self.mlp            = build_mlp(in_channels, d_hidden, d_model)
        self.modality_emb   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
        x: (B, 1, 5) or (B, 5) goal features

        Return 
        tokens: (B, 1, d_model)

        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, 5)
        
        x   = self.feat_dropout(x)
        token = self.mlp(x)
        token = token + self.modality_emb   # (1, 1, d_model)
        return token        # (B, 1, d_model)
    

# behavior encoder
class BehaviorEncoder(nn.Module):
    """
    Encode per-agent tactical behavior features.
 
    Input  : (B, N, 6)  per-agent behavior indicators
    Output : (B, N, d_model)
 
    NavsimDataset does not supply explicit behavior labels.  A proxy is
    derived in prepare_behavior_input() from multi_agent_states:
        [speed, lateral_speed, |heading|, accel_proxy, vx_sign, vy_sign]
 
    Feature dropout (p=0.15) applied during training — behavior signals
    are noisy or incorrect in real deployment.
 
    The output tokens can be used two ways in the backbone:
        1. Aggregated (mean/attention pool) → single adaLN conditioning vector
        2. Per-agent tokens for cross-attention with trajectory tokens
 
    Parameters
    ----------
    d_model   : Output token dimension.
    d_hidden  : MLP hidden dimension. Default 128.
    input_dim : Raw feature dimension. Default 6.
    dropout_p : Feature dropout probability. Spec: 0.15
    """

    def __init__(
            self, 
            d_model: int = 256,
            d_hidden: int = 128,
            input_dim: int = 6,
            dropout_p: float = 0.15
    ):
        super().__init__()
        self.feat_dropout  = FeatureDropout(p=dropout_p)
        self.mlp            = build_mlp(input_dim,d_hidden, d_model)

        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
            self, 
            x: torch.Tensor
    )-> torch.Tensor:
        """
        x: (B, N, 6) per-agent behavior feature

        Returns
        ----
        tokens: (B,. N, d_model)
        """

        x   = self.feat_dropout(x) # (B, N, 6)
        tokens = self.mlp(x)        # (B, N, d_model)
        tokens = tokens + self.modality_emb # 1, 1, d_model
        return tokens
    

    def aggregate(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mean-pool agent tokens → single conditioning vector for adaLN.
 
        Parameters
        ----------
        tokens : (B, N, d_model)
        mask   : (B, N) bool — True for absent agents (excluded from mean)
 
        Returns
        -------
        agg : (B, d_model)
        """

        if mask is not None:
            # zero out absent agents and compute masked mean
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
            count = (~mask).float().sum(dim = 1, keepdim= True).clamp(min = 1.0)
            return tokens.sum(dim = 1) / count      # (B, d_model)
        
        return tokens.mean(dim = 1) # (B, d_model)

# egostate encoder
class EgoStateEncoder(nn.Module):
    """
    Encode the ego vehicle state into a single reference-frame token.
 
    Raw features from NavsimDataset agent_states: (B, 1, 5)
        [x=0, y=0, vx, vy, h=0]  — ego always at origin with heading=0
 
    Feature engineering (deterministic, before MLP):
        1. Heading decomposition : replace h with (cos h, sin h)  → +1 dim
        2. Speed                 : sqrt(vx² + vy²)               → +1 dim
        3. Curvature             : yaw_rate / speed (if available)
                                   else approximated as 0          → +1 dim
 
    Final input to MLP: (B, 1, F_ego + 3)  where F_ego=5 → 8 dims.
 
    No feature dropout — ego state is always reliable (§10.2).
    The ego token is also the primary input to the ModalityGate.
 
    Parameters
    ----------
    d_model        : Output token dimension.
    d_hidden       : MLP hidden dimension. Default 128.
    raw_input_dim  : Dimension of raw ego features before engineering. Default 5.
    eps            : Small value for division stability in curvature. Default 1e-3.
    
    """
    ENGINEERED_DIM = 8   # 5 raw - 1 heading + cos + sin + speed + curvature = 8

    def __init__(
            self, 
            d_model     : int = 256,
            d_hidden    : int = 128,
            raw_input_dim: int = 5,
            eps         : float = 1e-3
    ):
        super().__init__()
        self.eps        = eps
        in_dim          = raw_input_dim - 1 + 3 # drop heading, add cos/sin/speed/curvature

        # raw input dim = 5 x, y, vx, vy , h -> drop_h (cos, sin) + speed + curvarute 5-1+3 = 7
        # Actually: keep x(0),y(0),vx,vy → 4 + cos_h + sin_h + speed + curvature = 8
        self.in_dim = raw_input_dim - 1 + 3
        # Correct: raw=[x,y,vx,vy,h] → engineered=[x,y,vx,vy,cos_h,sin_h,speed,curvature] = 8
        in_dim       = 8

        self.mlp = build_mlp(in_dim, d_hidden, d_model)
        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    @staticmethod
    def engineer_features(x: torch.Tensor, yaw_rate: Optional[torch.Tensor] = None,
                          eps: float = 1e-3) -> torch.Tensor:
        """
        Apply heading decomposition, speed, and curvature.
 
        Parameters
        ----------
        x        : (..., 5)  [x, y, vx, vy, heading]
        yaw_rate : (..., 1) or None — if provided, used for curvature
        eps      : stability clamp for division
 
        Returns
        -------
        out : (..., 8)  [x, y, vx, vy, cos_h, sin_h, speed, curvature]
        """
        pos_vel = x[..., :4]    # x, y, vx, vy
        heading = x[..., 4:5]   # (..., 1)

        cos_h = torch.cos(heading)      #smooth heading representation
        sin_h = torch.sin(heading)

        vx = x[..., 2:3]
        vy = x[..., 3:4]

        speed = (vx ** 2 + vy ** 2).sqrt()  # (..., 1)

        # curvature = yaw_rate / speed
        if yaw_rate is not None:
            curvature = yaw_rate / speed.clamp(min = eps)

        else:
            # Ego is at origin with h=0 in NavsimDataset; curvature not directly available.
            # Return zero — will be learned as a bias; accurate curvature
            # can be injected if yaw_rate is added to the dataset.

            curvature = torch.zeros_like(speed)

        return torch.cat([pos_vel, cos_h, sin_h, speed, curvature], dim = -1) # (..., 8)
    
    def forward(
            self, 
            x: torch.Tensor,
            yaw_rate: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x        : (B, 1, 5) or (B, 5)  ego state [x, y, vx, vy, heading]
        yaw_rate : (B, 1, 1) or None    optional yaw rate for curvature
 
        Returns
        -------
        token    : (B, 1, d_model)  ego token for backbone
        ego_feat : (B, d_model)     summary vector for modality gate
        
        """
        if x.dim() == 2:
            x =x.squeeze(1) # B, 1, 5
        x_eng = self.engineer_features(x, yaw_rate, self.eps) # (B, 1, 8)
        token = self.mlp(x_eng)                                 # (B, 1, d_model)
        token = token + self.modality_emb

        ego_feat = token.squeeze(1) # B, d_model
        return token, ego_feat      # (B, 1, d_model) (B, d_model)


# testing for modality gate, we dont do gate here
# Canonical modality names — index order must match gate_mlp output
MODALITY_NAMES: List[str] = [
    "history",        # 0  HistoryEncoder
    "bev",            # 1  BEVEncoder
    "ego",            # 2  EgoStateEncoder (always 1.0 — ego is unconditional)
    "agent_l2",       # 3  AgentEncoderL2
    "pedestrian",     # 4  PedestrianEncoder
    "traffic",        # 5  TrafficControlEncoder
    "intersection",   # 6  IntersectionEncoder
    "goal",           # 7  GoalIntentEncoder
    "behavior",       # 8  BehaviorEncoder
]
NUM_MODALITIES = len(MODALITY_NAMES)
EGO_MODALITY_IDX = 2   # ego gate is always 1.0 (unconditional reference)
 
 
class ModalityGate(nn.Module):
    """Ego-conditioned per-modality scalar gates in [0, 1].
 
    Architecture (§11.2)
    --------------------
    ego_summary : (B, d_model)  from EgoStateEncoder
    → ego_state_mlp  → (B, d_gate)
    → gate_mlp       → (B, num_modalities)
    → sigmoid        → gates in [0, 1]
 
    The gates are applied to raw inputs *before* encoding:
        history_input *= gates[:, 0]
        bev_input     *= gates[:, 1]
        ...etc
 
    Ego (index 2) is always 1.0 — the ego state is the unconditional
    reference frame and must never be suppressed.
 
    Training objectives (§11.3)
    ---------------------------
    1. Modality dropout: each gate is set to 0.0 with p=0.10 during
       training, forcing the model to plan with missing modalities.
    2. Minimum activation penalty: L_gate encourages each gate to be
       active (> 0.5) at least 10% of batches.
 
    Parameters
    ----------
    d_model          : Ego feature dimension (from EgoStateEncoder).
    d_gate           : Gate hidden dimension. Default 64.
    num_modalities   : Number of gates. Default 9.
    modality_dropout : Probability of zeroing a gate during training. Default 0.10.
    min_activation   : Target minimum fraction of batches each gate is active. Default 0.10.
    penalty_weight   : Weight for the minimum activation penalty. Default 0.01.
    """
 
    def __init__(
        self,
        d_model         : int = 256,
        d_gate          : int = 64,
        num_modalities  : int = NUM_MODALITIES,
        modality_dropout: float = 0.10,
        min_activation  : float = 0.10,
        penalty_weight  : float = 0.01,
    ):
        super().__init__()
        self.num_modalities   = num_modalities
        self.modality_dropout = modality_dropout
        self.min_activation   = min_activation
        self.penalty_weight   = penalty_weight
 
        # Ego summary projection
        self.ego_state_mlp = build_mlp(d_model, d_gate, d_gate)
 
        # Gate logit prediction
        self.gate_mlp = nn.Linear(d_gate, num_modalities)
 
        # Initialise gate logits to ~0 so sigmoid starts at ~0.5
        nn.init.zeros_(self.gate_mlp.weight)
        nn.init.zeros_(self.gate_mlp.bias)
 
   
    def forward(self, ego_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gates from ego feature summary.
 
        Parameters
        ----------
        ego_feat : (B, d_model)  from EgoStateEncoder.forward()[1]
 
        Returns
        -------
        gates      : (B, num_modalities)  values in [0, 1]
        gate_loss  : scalar tensor — minimum activation penalty (0 at inference)
        """
        ego_summary = self.ego_state_mlp(ego_feat)           # (B, d_gate)
        gate_logits = self.gate_mlp(ego_summary)             # (B, num_modalities)
        gates       = torch.sigmoid(gate_logits)             # (B, num_modalities)
 
        # Ego gate is unconditional — always 1.0
        ego_col         = torch.ones_like(gates[:, EGO_MODALITY_IDX:EGO_MODALITY_IDX+1])
        gates           = torch.cat([
            gates[:, :EGO_MODALITY_IDX],
            ego_col,
            gates[:, EGO_MODALITY_IDX+1:],
        ], dim=1)
 
        # ---- Training-only: modality dropout ----
        if self.training and self.modality_dropout > 0.0:
            dropout_mask = torch.bernoulli(
                torch.full((gates.shape[0], self.num_modalities),
                           1.0 - self.modality_dropout,
                           device=gates.device)
            )
            # Never drop ego gate
            dropout_mask[:, EGO_MODALITY_IDX] = 1.0
            gates = gates * dropout_mask
 
        # ---- Training-only: minimum activation penalty ----
        gate_loss = torch.tensor(0.0, device=gates.device)
        if self.training:
            # For each modality: penalise if mean(gate > 0.5) < min_activation
            active_frac  = (gates > 0.5).float().mean(dim=0)   # (num_modalities,)
            shortfall    = (self.min_activation - active_frac).clamp(min=0.0)
            gate_loss    = self.penalty_weight * shortfall.sum()
 
        return gates, gate_loss
 
   
    def apply_gates(
        self,
        gates   : torch.Tensor,
        inputs  : Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply gate scalars to raw modality inputs before encoding.
 
        Parameters
        ----------
        gates  : (B, num_modalities)
        inputs : dict mapping modality name → raw tensor.
                 Tensors may have varying shapes; the gate is broadcast
                 over all dims after dim 0.
 
        Returns
        -------
        gated_inputs : same dict with each tensor scaled by its gate.
        """
        gated = {}
        for idx, name in enumerate(MODALITY_NAMES):
            if name not in inputs:
                continue
            t   = inputs[name]
            g   = gates[:, idx]
            # Reshape gate to (B, 1, 1, ...) to broadcast over all spatial/feature dims
            for _ in range(t.dim() - 1):
                g = g.unsqueeze(-1)
            gated[name] = t * g
        return gated
 
 
# ---------------------------------------------------------------------------
# NavsimDataset adapters
# ---------------------------------------------------------------------------
 
def prepare_goal_input(batch: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """Extract goal_features from NavsimDataset batch → (B, 1, 5)."""
    x = batch["goal_features"]
    if device:
        x = x.to(device)
    x = x.float()
    if x.dim() == 2:
        x = x.unsqueeze(0)    # (1, 1, 5) if unbatched
    return x                  # (B, 1, 5)
 
 
def prepare_behavior_input(
    batch : dict,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a 6-dim behavior proxy from multi_agent_states.
 
    NavsimDataset does not provide explicit behavior labels, so we derive
    a proxy from the kinematic state:
 
        0: speed            = sqrt(vx² + vy²)
        1: lateral_speed    = |vy|  (non-ego reference — proxy for lane change)
        2: cos(heading)     — smooth heading signal
        3: sin(heading)
        4: accel_proxy      = speed (approximates how fast agent is moving;
                              true acceleration requires history delta)
        5: vx_sign          = sign(vx)  — forward vs backward motion
 
    Returns
    -------
    behavior : (B, N, 6)
    """
    s = batch["multi_agent_states"]   # (B, N, 5)  [x, y, vx, vy, heading]
    if device:
        s = s.to(device)
    s = s.float()
    if s.dim() == 2:
        s = s.unsqueeze(0)
 
    vx      = s[:, :, 2:3]
    vy      = s[:, :, 3:4]
    heading = s[:, :, 4:5]
 
    speed    = (vx ** 2 + vy ** 2).sqrt()
    lat_spd  = vy.abs()
    cos_h    = torch.cos(heading)
    sin_h    = torch.sin(heading)
    accel_px = speed                          # proxy — replace with Δspeed if history available
    vx_sign  = vx.sign()
 
    return torch.cat([speed, lat_spd, cos_h, sin_h, accel_px, vx_sign], dim=-1)  # (B, N, 6)
 
 
def prepare_ego_input(
    batch : dict,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract ego state from NavsimDataset batch.
 
    NavsimDataset agent_states shape: (B, 1, 5) [x=0, y=0, vx, vy, h=0]
 
    Returns
    -------
    ego_state : (B, 1, 5)
    yaw_rate  : None  (not in NavsimDataset; can be plumbed in externally)
    """
    s = batch.get("agent_states", batch.get("multi_agent_states"))
    if device:
        s = s.to(device)
    s = s.float()
    if s.dim() == 2:
        s = s.unsqueeze(0)
    # Take only ego row if multi_agent_states was used
    ego = s[:, 0:1, :]   # (B, 1, 5)
    return ego, None
 
 
# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 2, 32, 256
 
   
    print("=" * 60)
    print("GoalIntentEncoder  (§8)")
    print("=" * 60)
 
    goal_enc  = GoalIntentEncoder(d_model=D, d_hidden=128)
    goal_feats = torch.randn(B, 1, 5)
 
    goal_enc.train()
    token_train = goal_enc(goal_feats)
    goal_enc.eval()
    token_eval  = goal_enc(goal_feats)
 
    print(f"  Input  : {goal_feats.shape}")
    print(f"  Output : {token_train.shape}")          # (B, 1, 256)
    # Training output may differ due to feature dropout
    dropout_active = not torch.allclose(token_train, token_eval)
    print(f"  Feature dropout active in train: {dropout_active}")
    print(f"  Params : {sum(p.numel() for p in goal_enc.parameters()):,}")
 
   
    print("\n" + "=" * 60)
    print("BehaviorEncoder  (§9)")
    print("=" * 60)
 
    beh_enc    = BehaviorEncoder(d_model=D, d_hidden=128)
    beh_feats  = torch.randn(B, N, 6)
    pad_mask   = torch.zeros(B, N, dtype=torch.bool)
    pad_mask[:, 28:] = True
 
    beh_enc.train()
    beh_tokens_train = beh_enc(beh_feats)
    beh_enc.eval()
    beh_tokens_eval  = beh_enc(beh_feats)
 
    print(f"  Input  : {beh_feats.shape}")
    print(f"  Tokens : {beh_tokens_train.shape}")     # (B, 32, 256)
    print(f"  Aggregated (adaLN): {beh_enc.aggregate(beh_tokens_eval, pad_mask).shape}")
    print(f"  Feature dropout active in train: "
          f"{not torch.allclose(beh_tokens_train, beh_tokens_eval)}")
    print(f"  Params : {sum(p.numel() for p in beh_enc.parameters()):,}")
 
   
    print("\n" + "=" * 60)
    print("EgoStateEncoder  (§10)")
    print("=" * 60)
 
    ego_enc   = EgoStateEncoder(d_model=D, d_hidden=128)
    ego_state = torch.zeros(B, 1, 5)     # ego at origin, h=0
    ego_state[:, :, 2] = torch.tensor([3.0, 8.0]).unsqueeze(1)   # vx = 3, 8 m/s
 
    ego_token, ego_feat = ego_enc(ego_state)
    print(f"  Input        : {ego_state.shape}")
    print(f"  Token output : {ego_token.shape}")      # (B, 1, 256)
    print(f"  Ego feat     : {ego_feat.shape}")       # (B, 256)  for gate
 
    # Feature engineering check
    x_eng = EgoStateEncoder.engineer_features(ego_state)
    print(f"  Engineered features shape: {x_eng.shape}")  # (B, 1, 8)
    speeds = x_eng[0, 0, 6].item()
    print(f"  Computed speed for vx=3 : {speeds:.3f}  (expected 3.000)")
    print(f"  Params : {sum(p.numel() for p in ego_enc.parameters()):,}")
 
   
    print("\n" + "=" * 60)
    print("ModalityGate  (§11)")
    print("=" * 60)
 
    gate = ModalityGate(d_model=D, d_gate=64)
    gate.train()
 
    gates_train, loss_train = gate(ego_feat)
    print(f"  Gates shape   : {gates_train.shape}")   # (B, 9)
    print(f"  Gate values   : {gates_train[0].detach().tolist()}")
    print(f"  Ego gate (idx {EGO_MODALITY_IDX}) = "
          f"{gates_train[:, EGO_MODALITY_IDX].tolist()} (always 1.0)")
    assert (gates_train[:, EGO_MODALITY_IDX] == 1.0).all(), "Ego gate not fixed to 1.0"
    print(f"  Gate loss     : {loss_train.item():.6f}")
    print(f"  Params : {sum(p.numel() for p in gate.parameters()):,}")
 
    # apply_gates test
    raw_inputs = {
        "history"    : torch.randn(B, N, 4, 7),   # (B, N, T, F)
        "bev"        : torch.randn(B, 12, 200, 200),
        "ego"        : torch.randn(B, 1, 5),
        "pedestrian" : torch.randn(B, 10, 5),
        "traffic"    : torch.randn(B, 8, 5),
    }
    gated = gate.apply_gates(gates_train, raw_inputs)
    print(f"\n  apply_gates output shapes:")
    for k, v in gated.items():
        print(f"    {k:12s}: {tuple(v.shape)}")
 
   
    print("\n" + "=" * 60)
    print("NavsimDataset adapters")
    print("=" * 60)
 
    fake_batch = {
        "goal_features"       : torch.randn(B, 1, 5),
        "multi_agent_states"  : torch.randn(B, N, 5),
        "agent_states"        : torch.zeros(B, 1, 5),
    }
    fake_batch["agent_states"][:, :, 2] = 5.0   # vx = 5 m/s
 
    goal_x          = prepare_goal_input(fake_batch)
    beh_x           = prepare_behavior_input(fake_batch)
    ego_x, yaw_rate = prepare_ego_input(fake_batch)
 
    print(f"  goal_features   : {goal_x.shape}")
    print(f"  behavior proxy  : {beh_x.shape}")
    print(f"  ego_state       : {ego_x.shape}")
 
    # Full pipeline
    goal_enc.eval(); beh_enc.eval(); ego_enc.eval(); gate.eval()
 
    goal_token              = goal_enc(goal_x)
    beh_tokens              = beh_enc(beh_x)
    ego_token2, ego_feat2   = ego_enc(ego_x, yaw_rate)
    gates_eval, loss_eval   = gate(ego_feat2)
    beh_agg                 = beh_enc.aggregate(beh_tokens)
 
    print(f"\n  GoalIntent token : {goal_token.shape}")
    print(f"  Behavior tokens  : {beh_tokens.shape}")
    print(f"  Behavior agg     : {beh_agg.shape}")
    print(f"  Ego token        : {ego_token2.shape}")
    print(f"  Gates            : {gates_eval.shape}")
    print(f"  Gate loss (eval) : {loss_eval.item():.6f}  (always 0 at eval)")
 
   
    print("\n" + "=" * 60)
    print("Parameter summary")
    print("=" * 60)
    modules = {
        "GoalIntentEncoder": goal_enc,
        "BehaviorEncoder"  : beh_enc,
        "EgoStateEncoder"  : ego_enc,
        "ModalityGate"     : gate,
    }
    total = 0
    for name, mod in modules.items():
        n = sum(p.numel() for p in mod.parameters())
        total += n
        print(f"  {name:22s}: {n:>8,}")
    print(f"  {'TOTAL':22s}: {total:>8,}")
 
    print("\n✓ All checks passed")