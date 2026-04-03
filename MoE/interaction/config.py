"""
group_b/config.py

Configuration dataclass for the Group B (Intention & Interaction) pipeline.
All architectural hyper-parameters live here so callers never pass magic numbers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Agent-type integer IDs
# ---------------------------------------------------------------------------
AGENT_TYPE_VEHICLE    = 0
AGENT_TYPE_PEDESTRIAN = 1
AGENT_TYPE_CYCLIST    = 2
EGO_TYPE_ID           = 3   # ego vehicle is its own type so it is always distinguishable

# Number of distinct agent types (including ego)
N_AGENT_TYPES = 4

# ---------------------------------------------------------------------------
# Intention class registries
# ---------------------------------------------------------------------------
VEHICLE_INTENTIONS: Dict[str, int] = {
    "lane_keep":          0,
    "lane_change_left":   1,
    "lane_change_right":  2,
    "merge":              3,
    "cut_in":             4,
    "stop":               5,
}
N_VEHICLE_INTENTIONS = len(VEHICLE_INTENTIONS)   # 6

PEDESTRIAN_INTENTIONS: Dict[str, int] = {
    "crossing":     0,
    "not_crossing": 1,
}
N_PEDESTRIAN_INTENTIONS = len(PEDESTRIAN_INTENTIONS)   # 2

# Unified logit width: pedestrian logits are zero-padded to this size
INTENTION_LOGIT_DIM = N_VEHICLE_INTENTIONS   # 6

# ---------------------------------------------------------------------------
# Scene-density K lookup table (neighbourhood size for Stage 2)
# ---------------------------------------------------------------------------
K_TABLE: Dict[str, int] = {
    "highway": 6,    # fewer, faster agents
    "urban":   12,   # dense, mixed traffic
    "parking": 8,    # slow but spatially dense
    "default": 10,
}

# ---------------------------------------------------------------------------
# GroupBConfig
# ---------------------------------------------------------------------------
@dataclass
class GroupBConfig:
    """
    All hyper-parameters for the Group B pipeline.

    Dimension naming convention
    ---------------------------
    hidden_dim  (D)     : main model width, shared across all sub-modules
    D_geom              : output dim of EgoRelativeGeometry projection
    D_hist              : output dim of AgentHistoryEncoder GRU
    D_type              : agent-type embedding dim
    D_id                : group-identity embedding dim (set externally, matched here)
    D_t                 : diffusion timestep embedding dim (from transdiffuser)
    state_dim           : raw per-agent state vector width (x, y, heading, vx, vy)
    T_hist              : number of history timesteps fed to the GRU
    """

    # ---- Core width --------------------------------------------------------
    hidden_dim: int = 256       # D

    # ---- Sub-module dims ---------------------------------------------------
    D_geom: int = 64            # ego-relative geometry projection output
    D_hist: int = 128           # GRU hidden size / history embedding
    D_type: int = 32            # agent-type embedding
    D_id:   int = 32            # group-identity embedding (must match token_router.py)
    D_t:    int = 256           # timestep embedding (must match transdiffuser.py)

    # ---- Input / history ---------------------------------------------------
    state_dim: int = 5          # (x, y, heading, vx, vy)
    T_hist:    int = 10         # history window in timesteps
    n_agent_types: int = N_AGENT_TYPES

    # ---- Stage 1 (ego-centric cross-attention) -----------------------------
    stage1_n_heads: int = 4

    # ---- Stage 2 (agent-agent attention) -----------------------------------
    stage2_n_heads: int = 4
    stage2_d_ref:   float = 50.0   # reference distance for bias normalisation (metres)
    stage2_alpha:   float = 1.0    # distance-bias strength

    # ---- Gate query / temperature schedule --------------------------------
    tau_min: float = 0.3    # temperature floor  (sharp routing at low noise)
    tau_max: float = 1.5    # temperature ceiling (soft  routing at high noise)

    # ---- Skip predictor ----------------------------------------------------
    skip_floor: float = 0.20    # minimum fraction of tokens always processed
    K_rare:     int   = 5       # rare-type threshold: below this → never skip

    # ---- Risk-adaptive gate ------------------------------------------------
    alpha_risk:      float = 0.2
    base_threshold:  float = 0.4
    V_MAX:           float = 30.0   # m/s, used to normalise velocity risk

    # ---- Shared expert weight schedule ------------------------------------
    w_shared_min:       float = 0.10
    w_shared_max_delta: float = 0.10

    # ---- Loss weights (owned here for completeness, consumed in trainer) ---
    lambda_intention: float = 0.5
    lambda_skip:      float = 0.1

    # ---- Scene class (set per-batch by the dataloader) --------------------
    # Not a tensor — passed as a plain str to forward() or resolved externally.
    default_scene_class: str = "default"

    # ---- Derived / read-only properties -----------------------------------
    @property
    def input_proj_in_dim(self) -> int:
        """Total dim of the concatenated input token before linear projection."""
        return self.D_hist + self.D_geom + self.D_type + self.D_id