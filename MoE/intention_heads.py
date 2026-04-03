"""
STUB-D implementation: IntentionHeads

Pre-gate intention prediction heads for Group B tokens (§2.7).

Motivation:
    The Group B gate (GroupBRouter) implicitly learns to route agents by
    behavioral mode.  Making this explicit via supervised intention heads:
      - Gives the gate a structured prior (6 vehicle classes, 2 ped classes)
        rather than learning the decomposition from routing signal alone.
      - Decouples "what is this agent likely to do" from "which expert
        should process it" — the gate can then specialise experts around
        known archetypes instead of discovering them from scratch.
      - Provides a loss signal (L_intention) that is faster to converge
        than the MoE routing objective alone.

Intention taxonomy (§2.7):
    Vehicle (and cyclist — see note below):
        6 classes = Lateral × Longitudinal
            Lateral:      Left-change / Straight / Right-change   (3)
            Longitudinal: Accelerating / Constant / Decelerating  (3)
        Label matrix:
            0: Left  + Accel    3: Straight + Accel
            1: Left  + Const    4: Straight + Const   ← most common
            2: Left  + Decel    5: Straight + Decel
            (Right analogues wrap: 6-class total with right = mirrored)
        Actual 6-class layout used here:
            0: LL (Left-change + Decel)
            1: LS (Left-change + Straight-speed)
            2: LA (Left-change + Accel)
            3: KK (Keep-lane + any — rolled into Straight bucket)
            4: RA (Right-change + Accel)
            5: RD (Right-change + Decel)
        The exact label convention must match the data pipeline's
        intention label extraction.  The head is label-convention-agnostic
        — it outputs 6 logits regardless of how they're named.

    Pedestrian:
        2 classes: not-crossing (0) / crossing (1)
        SEPARATE head — pedestrians do not have a meaningful
        Lateral × Longitudinal decomposition.

    Cyclist:
        Treated as vehicle (6-class head) — cyclists exhibit the same
        lateral/longitudinal intent structure as vehicles in nuPlan.
        ⚠ DECISION POINT: if your dataset has a dedicated cyclist
        intention taxonomy, add a third head and handle agent_type == 2
        separately.  Current choice: cyclist → vehicle head.

Agent type codes (consistent with moe_block.py STUB-D docstring):
    0 = vehicle
    1 = pedestrian
    2 = cyclist  (routed to vehicle head)

Output for gate query (§2.8):
    intention_for_gate: (B, N_B, output_dim)
        output_dim = max(vehicle_classes, ped_classes) = 6
        For pedestrians: 2-class logits are zero-padded to 6 dims.
        For vehicles/cyclists: 6-class logits used directly.
    This tensor is concatenated into the GroupBRouter gate_input.

Loss (L_intention, §2.7):
    Cross-entropy, computed separately per agent type.
    Cyclists use the vehicle head loss.
    Total: L_intention = mean(CE_vehicle) + mean(CE_pedestrian)
           weighted by the number of valid tokens of each type.
    Returns 0.0 scalar when no GT labels are provided (inference).

Risk-adaptive threshold (§2.7 open decision):
    DEFERRED — fixed threshold of 0.4 for all tokens per current plan.
    When implemented, high-risk agents (close, converging, high speed)
    should route with lower fallback to shared expert.
    Hook: risk_scores (B, N_B) float can be passed to forward; currently
    ignored but accepted so the interface doesn't change when this is added.

MoEBlockOutput extension:
    Once this stub is integrated, add to MoEBlockOutput:
        intention_loss: torch.Tensor   # scalar, λ_intention * L_intention
    And in StackedMoEBlocks.forward, accumulate:
        total_intention = total_intention + out.intention_loss
    And add to total_aux:
        total_aux = total_decorr + total_capacity + total_ortho + total_intention

Integration in MoEBlock.__init__:
    from MoE.intention_heads import IntentionHeads
    self.intention_heads = IntentionHeads(cfg.embed_dim)

Integration in MoEBlock.forward (replace [STUB-D] TODO):
    # agent_types: (B, N_B) long, 0=vehicle, 1=pedestrian, 2=cyclist
    # intention_gt: (B, N_B) long or None (None at inference)
    intention_for_gate, intention_loss = self.intention_heads(
        agent_repr=tokens_B_after_stage2,   # (B, N_B, D) Stage 2 output
        agent_types=agent_types,             # (B, N_B) long
        intention_gt=intention_gt,           # (B, N_B) long or None
    )
    # intention_for_gate feeds into GroupBRouter gate query (§2.8)
    # intention_loss contributes to total auxiliary loss

Integration in GroupBRouter (token_router.py):
    Replace IntentionPredictionhead with IntentionHeads.
    Change forward signature to accept agent_types (long) instead of
    is_pedestrian (bool).  intention_for_gate shape stays (B, N_B, 6).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Agent type codes — must match data pipeline convention.
AGENT_VEHICLE    = 0
AGENT_PEDESTRIAN = 1
AGENT_CYCLIST    = 2   # treated as vehicle for intention purposes

# Intention class counts
NUM_VEHICLE_CLASSES    = 6
NUM_PEDESTRIAN_CLASSES = 2
GATE_OUTPUT_DIM        = max(NUM_VEHICLE_CLASSES, NUM_PEDESTRIAN_CLASSES)  # = 6


class IntentionHeads(nn.Module):
    """Vehicle 6-class + pedestrian 2-class pre-gate intention heads (§2.7).

    Two completely separate MLP heads — no weight sharing between vehicle
    and pedestrian.  Cyclists are routed through the vehicle head.

    Architecture (per head):
        Linear(D → D//2) → ReLU → Linear(D//2 → num_classes)

    The output for gate query is always (B, N_B, 6):
        - Vehicles/cyclists: 6-class logits directly.
        - Pedestrians: 2-class logits, zero-padded to 6 dims.
        Zero-padding is intentional — the gate MLP learns to read the
        pedestrian signal from dims 0-1 and ignore dims 2-5 for peds.

    Loss:
        CE_vehicle:    cross-entropy on vehicle + cyclist tokens.
        CE_pedestrian: cross-entropy on pedestrian tokens.
        L_intention = (sum_vehicle_CE + sum_ped_CE) / (N_vehicle + N_ped)
        Returns 0.0 when intention_gt is None (inference) or when a type
        has zero valid tokens.

    Args:
        embed_dim:          D — token feature dimension
        vehicle_classes:    number of vehicle intention classes (default 6)
        pedestrian_classes: number of pedestrian intention classes (default 2)
    """

    def __init__(
        self,
        embed_dim:          int = 256,
        vehicle_classes:    int = NUM_VEHICLE_CLASSES,
        pedestrian_classes: int = NUM_PEDESTRIAN_CLASSES,
    ):
        super().__init__()
        self.vehicle_classes    = vehicle_classes
        self.pedestrian_classes = pedestrian_classes
        self.gate_output_dim    = max(vehicle_classes, pedestrian_classes)
        hidden = embed_dim // 2

        # Vehicle head: D → D//2 → vehicle_classes
        self.vehicle_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, vehicle_classes),
        )

        # Pedestrian head: D → D//2 → pedestrian_classes (SEPARATE weights)
        self.pedestrian_head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pedestrian_classes),
        )

        self._init_weights()

     
    def _init_weights(self) -> None:
        """Small output layer init — heads start near-uniform."""
        for head in (self.vehicle_head, self.pedestrian_head):
            final_linear = head[-1]
            nn.init.xavier_uniform_(final_linear.weight, gain=0.1)
            nn.init.zeros_(final_linear.bias)

     
    def forward(
        self,
        agent_repr:    torch.Tensor,                    # (B, N_B, D)
        agent_types:   torch.Tensor,                    # (B, N_B) long: 0=veh,1=ped,2=cyc
        intention_gt:  Optional[torch.Tensor] = None,   # (B, N_B) long or None
        risk_scores:   Optional[torch.Tensor] = None,   # (B, N_B) float — reserved
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intention logits for gate and (optionally) the CE loss.

        Args:
            agent_repr:   (B, N_B, D) — Stage 2 agent representations
            agent_types:  (B, N_B) long — 0=vehicle, 1=pedestrian, 2=cyclist
            intention_gt: (B, N_B) long — GT intention class indices, or None
                          at inference.  Cyclists use vehicle-class GT labels.
            risk_scores:  (B, N_B) float — reserved for risk-adaptive threshold
                          (§2.7 open decision, currently ignored).

        Returns:
            intention_for_gate: (B, N_B, 6) — padded logits for gate query
            intention_loss:     scalar tensor — 0.0 at inference or if no GT
        """
        B, N, D = agent_repr.shape

         
        # Run both heads over ALL tokens.
        # Masking happens after — avoids Python loops over the batch.
         
        veh_logits = self.vehicle_head(agent_repr)      # (B, N, 6)
        ped_logits = self.pedestrian_head(agent_repr)   # (B, N, 2)

         
        # Build gate output: (B, N, 6)
        #   pedestrians → ped_logits zero-padded to 6 dims
        #   vehicles / cyclists → veh_logits directly
         
        is_ped = (agent_types == AGENT_PEDESTRIAN).unsqueeze(-1).float()  # (B,N,1)

        # Pad ped logits to 6 dims
        pad_size = self.gate_output_dim - self.pedestrian_classes         # = 4
        ped_padded = F.pad(ped_logits, (0, pad_size))                     # (B, N, 6)

        # Soft blend: pedestrian tokens get padded ped logits,
        # vehicle/cyclist tokens get vehicle logits.
        intention_for_gate = (1.0 - is_ped) * veh_logits + is_ped * ped_padded  # (B, N, 6)

         
        # Intention loss (only during training, only when GT provided).
         
        intention_loss = self._compute_loss(
            veh_logits, ped_logits, agent_types, intention_gt
        )

        return intention_for_gate, intention_loss

     
    def _compute_loss(
        self,
        veh_logits:   torch.Tensor,                  # (B, N, 6)
        ped_logits:   torch.Tensor,                  # (B, N, 2)
        agent_types:  torch.Tensor,                  # (B, N) long
        intention_gt: Optional[torch.Tensor],        # (B, N) long or None
    ) -> torch.Tensor:
        """Cross-entropy loss over valid (non-padding) tokens.

        Returns scalar 0.0 if intention_gt is None or a type has no tokens.
        Cyclists use the vehicle head labels.
        """
        device = veh_logits.device

        if intention_gt is None:
            return torch.tensor(0.0, device=device)

        #  Vehicle + cyclist tokens 
        veh_mask = (agent_types == AGENT_VEHICLE) | (agent_types == AGENT_CYCLIST)
        # (B, N) bool
        loss_veh = torch.tensor(0.0, device=device)
        n_veh = veh_mask.sum().item()

        if n_veh > 0:
            veh_logits_flat = veh_logits[veh_mask]          # (N_veh, 6)
            veh_gt_flat     = intention_gt[veh_mask]        # (N_veh,)
            loss_veh = F.cross_entropy(veh_logits_flat, veh_gt_flat)

        #  Pedestrian tokens 
        ped_mask = (agent_types == AGENT_PEDESTRIAN)
        loss_ped = torch.tensor(0.0, device=device)
        n_ped = ped_mask.sum().item()

        if n_ped > 0:
            ped_logits_flat = ped_logits[ped_mask]          # (N_ped, 2)
            ped_gt_flat     = intention_gt[ped_mask]        # (N_ped,)
            # GT labels for pedestrians must be in {0, 1}
            loss_ped = F.cross_entropy(ped_logits_flat, ped_gt_flat)

        #  Weighted mean (by token count, not equal weight per type) 
        total_n = n_veh + n_ped
        if total_n == 0:
            return torch.tensor(0.0, device=device)

        loss = (n_veh * loss_veh + n_ped * loss_ped) / total_n
        return loss


    @property
    def output_dim(self) -> int:
        """Dimension of intention_for_gate — for gate MLP input sizing."""
        return self.gate_output_dim



# Sanity check


if __name__ == "__main__":
    torch.manual_seed(42)
    B, D = 2, 128
    N_B  = 12   # mixed: 5 vehicles, 4 pedestrians, 3 cyclists

    heads = IntentionHeads(embed_dim=D)

    # Build mixed agent_types: [veh×5, ped×4, cyc×3]
    agent_types = torch.tensor(
        [[0, 0, 0, 0, 0,   # 5 vehicles
          1, 1, 1, 1,       # 4 pedestrians
          2, 2, 2],         # 3 cyclists
         [0, 0, 0, 1, 1,
          1, 2, 2, 2,
          0, 0, 1]],
        dtype=torch.long
    )  # (2, 12)
    assert agent_types.shape == (B, N_B)

    agent_repr = torch.randn(B, N_B, D, requires_grad=True)

    #  Inference (no GT) 
    gate_out, loss_inf = heads(agent_repr, agent_types, intention_gt=None)

    assert gate_out.shape == (B, N_B, 6), f"gate_out shape: {gate_out.shape}"
    assert loss_inf.item() == 0.0, "Loss should be 0 at inference"
    assert not torch.isnan(gate_out).any(), "NaN in gate output"
    print(f"Inference — gate_out: {gate_out.shape}  loss: {loss_inf.item():.4f}  ✓")

    #  Training (with GT)
    # Vehicle GT: classes 0-5. Ped GT: classes 0-1. Cyclist GT: classes 0-5.
    intention_gt = torch.zeros(B, N_B, dtype=torch.long)
    # Assign valid GT per agent type per sample
    for b in range(B):
        for n in range(N_B):
            if agent_types[b, n] == AGENT_PEDESTRIAN:
                intention_gt[b, n] = torch.randint(0, 2, (1,)).item()
            else:  # vehicle or cyclist
                intention_gt[b, n] = torch.randint(0, 6, (1,)).item()

    gate_out_tr, loss_tr = heads(agent_repr, agent_types, intention_gt=intention_gt)
    assert gate_out_tr.shape == (B, N_B, 6)
    assert loss_tr.item() > 0.0, "Training loss should be > 0"
    assert not torch.isnan(loss_tr), "NaN in training loss"
    print(f"Training  — gate_out: {gate_out_tr.shape}  loss: {loss_tr.item():.4f}  ✓")

    # Gradient flow 
    loss_tr.backward()
    assert agent_repr.grad is not None, "agent_repr should have grad"
    assert not torch.isnan(agent_repr.grad).any(), "NaN in grad"
    print(f"Gradient flow through CE loss:  ✓")

    # Pedestrian uses its own head (different params than vehicle) 
    # Zero the vehicle head, verify gate output for pedestrian tokens differs
    # from vehicle tokens in a controlled way.
    agent_repr2 = torch.randn(B, N_B, D)
    with torch.no_grad():
        for p in heads.vehicle_head.parameters():
            p.zero_()
    gate_zeroed, _ = heads(agent_repr2, agent_types, intention_gt=None)
    # Vehicle tokens should all output zero (head zeroed)
    veh_mask_b0 = agent_types[0] != AGENT_PEDESTRIAN   # (N_B,) bool
    veh_gate = gate_zeroed[0][veh_mask_b0]
    ped_gate  = gate_zeroed[0][agent_types[0] == AGENT_PEDESTRIAN]
    assert torch.allclose(veh_gate, torch.zeros_like(veh_gate)), \
        "Vehicle head zeroed → vehicle gate output should be zero"
    assert not torch.allclose(ped_gate, torch.zeros_like(ped_gate)), \
        "Pedestrian head still active → ped gate output should be non-zero"
    print(f"Separate vehicle/pedestrian heads (independent weights):  ✓")

    #  Cyclist routed to vehicle head
    # With vehicle head zeroed, cyclists should also output zeros (same head).
    cyc_mask_b0 = agent_types[0] == AGENT_CYCLIST
    cyc_gate = gate_zeroed[0][cyc_mask_b0]
    assert torch.allclose(cyc_gate, torch.zeros_like(cyc_gate)), \
        "Cyclists use vehicle head → should also be zero when vehicle head zeroed"
    print(f"Cyclists routed through vehicle head:  ✓")

    #  Pedestrian zero-padding 
    # Restore vehicle head, verify ped gate dims 2-5 are always zero
    # when ped_logits is computed (2-class padded to 6).
    heads2 = IntentionHeads(embed_dim=D)
    # Force ped head to output a known value so we can check padding
    with torch.no_grad():
        for p in heads2.pedestrian_head.parameters():
            p.zero_()   # zero head → ped logits all zero → padded also zero
    pure_ped_types = torch.ones(1, 4, dtype=torch.long)   # all pedestrians
    pure_ped_repr  = torch.randn(1, 4, D)
    gate_ped, _ = heads2(pure_ped_repr, pure_ped_types)
    # dims 2-5 should be zero (padding)
    assert torch.allclose(gate_ped[0, :, 2:], torch.zeros(4, 4)), \
        "Ped gate dims 2-5 should be zero (padding)"
    print(f"Pedestrian zero-padding to 6 dims (dims 2-5 = 0):  ✓")

    # ── All-vehicle batch 
    heads3 = IntentionHeads(embed_dim=D)
    all_veh_types = torch.zeros(B, N_B, dtype=torch.long)
    all_veh_gt    = torch.randint(0, 6, (B, N_B))
    gate_v, loss_v = heads3(agent_repr2.requires_grad_(True),
                             all_veh_types, all_veh_gt)
    assert gate_v.shape == (B, N_B, 6)
    assert loss_v.item() > 0.0
    print(f"All-vehicle batch:  loss={loss_v.item():.4f}  ✓")

    # ── All-pedestrian batch
    all_ped_types = torch.ones(B, N_B, dtype=torch.long)
    all_ped_gt    = torch.randint(0, 2, (B, N_B))
    gate_p, loss_p = heads3(agent_repr2.detach().requires_grad_(True),
                             all_ped_types, all_ped_gt)
    assert gate_p.shape == (B, N_B, 6)
    assert loss_p.item() > 0.0
    # dims 2-5 should carry zero contribution for all-ped (soft-blended to ped path)
    print(f"All-pedestrian batch: loss={loss_p.item():.4f}  ✓")

    #  output_dim property 
    assert heads3.output_dim == 6
    print(f"output_dim == 6:  ✓")

    #  No GT, no loss (inference guard) 
    _, loss_none = heads3(agent_repr2, all_veh_types, intention_gt=None)
    assert loss_none.item() == 0.0
    print(f"No GT → loss == 0.0:  ✓")

    #  Parameter count 
    n_params = sum(p.numel() for p in IntentionHeads(D).parameters())
    print(f"Parameter count: {n_params:,}  ✓")

    print("\n" + "="*55)
    print("All IntentionHeads tests PASSED.")
    print("="*55)
    print("\nOpen decision (§2.7):")
    print("  Risk-adaptive gate threshold (high-risk → lower shared-expert")
    print("  fallback) is DEFERRED. risk_scores kwarg accepted but ignored.")
    print("  Wire it in when the risk scoring module is ready.")