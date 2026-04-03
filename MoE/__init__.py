"""
DDPM/moe/__init__.py

Public API for the three-group Modality-level MoE module.

Import surface kept intentionally narrow: callers should only need
the config, the backbone builder, and the output types.

Usage:
    from models.moe import (
        MoEBlockConfig,
        StackedMoEBlocks,
        build_moe_backbone,
        StackedMoEOutput,
        TTYPE_PHANTOM,
        TTYPE_VECTORMAP,
        TTYPE_TRAFFIC_LT,
        TTYPE_INTERSECTION,
        TTYPE_BEV_STRUCT,
        TTYPE_UNKNOWN,
    )
"""

from MoE.token_router import (
    MoEConfig,
    ModalityMoERouter,
    RouterOutput,
    GroupIdentityEmbedding,
    GroupLocalLayerNorm,
    DirectedAttentionMask,
    TimestepEmbedding,
    DyDiTSkipScheduler,
    GroupTokenizer,
    build_moe_router,
    # Group constants
    GROUP_A,
    GROUP_B,
    GROUP_C,
    # Token type constants (for Group C structural router)
    TTYPE_PHANTOM,
    TTYPE_VECTORMAP,
    TTYPE_TRAFFIC_LT,
    TTYPE_INTERSECTION,
    TTYPE_BEV_STRUCT,
    TTYPE_UNKNOWN,
    NUM_C_TYPES,
)

from MoE.expert_ffn import (
    GatedExpertFFN,
    ExpertPool,
    GroupAExpertPool,
    GroupBExpertPool,
    GroupCExpertPool,
    AllGroupsExpertRunner,
    ExpertOutputs,
    build_expert_pools,
    count_expert_params,
)

from MoE.decorrelation_loss import (
    DecorrConfig,
    CosinePairwiseLoss,
    RBFKernelMMDLoss,
    AnchorBuffer,
    GroupADecorrLoss,
    GroupBDecorrLoss,
    GroupCDecorrLoss,
    ThreeGroupDecorrLoss,
    PhaseTracker,
    build_decorr_loss,
)

from MoE.moeblock import (
    MoEBlockConfig,
    SharedSelfAttention,
    MoEBlock,
    MoEBlockOutput,
    StackedMoEBlocks,
    StackedMoEOutput,
    build_moe_backbone,
)

__all__ = [
    # Configs
    "MoEConfig",
    "MoEBlockConfig",
    "DecorrConfig",
    # Top-level modules
    "ModalityMoERouter",
    "StackedMoEBlocks",
    "MoEBlock",
    "ThreeGroupDecorrLoss",
    "AllGroupsExpertRunner",
    # Builders
    "build_moe_backbone",
    "build_moe_router",
    "build_expert_pools",
    "build_decorr_loss",
    # Output types
    "RouterOutput",
    "MoEBlockOutput",
    "StackedMoEOutput",
    "ExpertOutputs",
    # Group constants
    "GROUP_A", "GROUP_B", "GROUP_C",
    # Token type IDs for Group C
    "TTYPE_PHANTOM", "TTYPE_VECTORMAP", "TTYPE_TRAFFIC_LT",
    "TTYPE_INTERSECTION", "TTYPE_BEV_STRUCT", "TTYPE_UNKNOWN",
    "NUM_C_TYPES",
    # Utilities
    "PhaseTracker",
    "count_expert_params",
    "DyDiTSkipScheduler",
]