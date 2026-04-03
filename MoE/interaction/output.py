"""
group_b/outputs.py

GroupBOutput  — named output container for GroupBPipeline.forward()
EgoRoutingMixin — mixin logic for privileged ego-token routing (audit §8.2)

Keeping the output in a dataclass (rather than a plain tuple) means callers
can access fields by name, which makes the call sites in moe_block.py and
token_router.py self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


# ---------------------------------------------------------------------------
# GroupBOutput
# ---------------------------------------------------------------------------

@dataclass
class GroupBOutput:
    """
    Container returned by GroupBPipeline.forward().

    Fields
    ------
    tokens          : [B, N, D]     final agent representations (repr3)
    gate_query      : [B, N, D]     temperature-scaled query for MoE router
    skip_mask       : [B, N] bool   True = skip this token at this layer
    skip_scores     : [B, N] float  raw skip MLP scores (needed for L_skip)
    int_logits      : [B, N, 6]     intention logits  (for L_intention + inference)
    aux_losses      : dict          named scalar losses collected during forward
                                    keys: "L_intention"  (present when labels given)
    """
    tokens:      torch.Tensor
    gate_query:  torch.Tensor
    skip_mask:   torch.Tensor
    skip_scores: torch.Tensor
    int_logits:  torch.Tensor
    aux_losses:  Dict[str, torch.Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# apply_privileged_ego_routing  (resolves audit §8.2)
# ---------------------------------------------------------------------------

def apply_privileged_ego_routing(
    dispatch_mask:    torch.Tensor,    # [B, N, E]  bool — which expert each token goes to
    combine_weights:  torch.Tensor,    # [B, N, E]  float — combine coefficients
    agent_type_ids:   torch.Tensor,    # [B, N]
    ego_type_id:      int,
    ego_expert_idx:   int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Force the ego token to always use expert `ego_expert_idx` (index 0 by
    convention) and give that expert double weight in the combine step.

    The ego agent is privileged because:
      - It drives the scene; its routing decision anchors all interaction
        tokens.
      - It must not be lost in the routing competition (top-K sampling
        could, by chance, route it to any expert).

    Args:
        dispatch_mask   : modified in-place; also returned for clarity
        combine_weights : modified in-place; also returned for clarity
        agent_type_ids  : [B, N]
        ego_type_id     : integer ID for the ego agent type
        ego_expert_idx  : which expert index the ego token is always sent to

    Returns:
        (dispatch_mask, combine_weights)  — same tensors after modification
    """
    ego_mask = (agent_type_ids == ego_type_id)   # [B, N]

    # Dispatch: send ego exclusively to the privileged expert
    dispatch_mask[ego_mask, :]             = False
    dispatch_mask[ego_mask, ego_expert_idx] = True

    # Combine: double weight on the privileged expert output for ego tokens
    combine_weights[ego_mask, ego_expert_idx] *= 2.0
    # Renormalise so the combine weights still sum to 1
    row_sum = combine_weights[ego_mask].sum(dim=-1, keepdim=True).clamp(min=1e-6)
    combine_weights[ego_mask] = combine_weights[ego_mask] / row_sum

    return dispatch_mask, combine_weights