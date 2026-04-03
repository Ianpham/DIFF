"""
group_b/__init__.py

Public API for the Group B intention & interaction package.

Typical import pattern in moe_block.py / token_router.py:

    from group_b import GroupBPipeline, GroupBConfig, GroupBOutput
    from group_b import apply_privileged_ego_routing
    from group_b import shared_expert_weight
    from group_b import compute_risk_score, risk_adaptive_threshold
    from group_b.skip_and_shared import GroupBSkipPredictor
"""

from .config import (
    GroupBConfig,
    # Agent type IDs
    AGENT_TYPE_VEHICLE,
    AGENT_TYPE_PEDESTRIAN,
    AGENT_TYPE_CYCLIST,
    EGO_TYPE_ID,
    N_AGENT_TYPES,
    # Intention registries
    VEHICLE_INTENTIONS,
    PEDESTRIAN_INTENTIONS,
    INTENTION_LOGIT_DIM,
    # K lookup table
    K_TABLE,
)

from .input_token import (
    EgoRelativeGeometry,
    AgentHistoryEncoder,
)

from .interaction_stages import (
    EgoCentricCrossAttention,
    AgentAgentAttention,
    compute_pairwise_distances,
    compute_pairwise_distances_from_states,
)

from .map_reweighting import MapContextReweighting

from .intention_head import (
    IntentionHeads,
    intention_loss,
)

from .gate_query import GroupBGateQuery

from .skip_shared import (
    GroupBSkipPredictor,
    build_rare_mask,
    shared_expert_weight,
)

from .risk_gate import (
    compute_risk_score,
    risk_adaptive_threshold,
)

from .output import (
    GroupBOutput,
    apply_privileged_ego_routing,
)

from .pipeline import GroupBPipeline

__all__ = [
    # Config & constants
    "GroupBConfig",
    "AGENT_TYPE_VEHICLE",
    "AGENT_TYPE_PEDESTRIAN",
    "AGENT_TYPE_CYCLIST",
    "EGO_TYPE_ID",
    "N_AGENT_TYPES",
    "VEHICLE_INTENTIONS",
    "PEDESTRIAN_INTENTIONS",
    "INTENTION_LOGIT_DIM",
    "K_TABLE",
    # Sub-modules (exposed for unit testing)
    "EgoRelativeGeometry",
    "AgentHistoryEncoder",
    "EgoCentricCrossAttention",
    "AgentAgentAttention",
    "MapContextReweighting",
    "IntentionHeads",
    "GroupBGateQuery",
    "GroupBSkipPredictor",
    # Helpers
    "compute_pairwise_distances",
    "compute_pairwise_distances_from_states",
    "build_rare_mask",
    "shared_expert_weight",
    "compute_risk_score",
    "risk_adaptive_threshold",
    "intention_loss",
    # Output types
    "GroupBOutput",
    "apply_privileged_ego_routing",
    # Top-level pipeline
    "GroupBPipeline",
]