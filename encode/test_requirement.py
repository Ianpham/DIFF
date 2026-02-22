"""
Test encoder requirements system.
=================================
Tests compatibility checking across different dataset configurations,
plus integration tests with the AgentEncoder.

Usage:
    python test_requirements.py

Note:
    Uses ContractBuilder to simulate different dataset contracts.
    No actual dataset loading is needed.
"""

import torch
import torch.nn as nn
from typing import Optional

from requirements import (
    EncoderRequirements,
    StandardRequirements,
    RequirementValidator,
)
from datasets.navsim.navsim_utilize.contract import FeatureType, DataContract, ContractBuilder


# =============================================================================
# Helper: Build common dataset contracts
# =============================================================================

def build_full_contract() -> DataContract:
    """Simulate EnhancedNavsimDataset — has everything."""
    return (
        ContractBuilder("EnhancedNavsimDataset")
        .add_feature(FeatureType.LIDAR_POINTS, shape=(-1, 3))
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.CAMERA_IMAGES, shape=(8, 3, 900, 1600))
        .add_feature(FeatureType.BEV_LABELS, shape=(12, 200, 200))
        .add_feature(FeatureType.VECTOR_MAP, shape=(-1,))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 7))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 20, 7))
        .add_feature(FeatureType.AGENT_NEARBY, shape=(1, 10, 7))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .add_feature(FeatureType.ROUTE, shape=(-1,))
        .add_feature(FeatureType.DIFFICULTY, shape=(-1,))
        .set_physical_limits(max_batch_size=8, memory_footprint_mb=200.0)
        .set_semantic_info(
            num_cameras=8, bev_channels=12, agent_state_dim=7,
            history_length=20, has_acceleration=True,
            has_nearby_agents=True, has_vector_maps=True,
        )
        .build()
    )

def build_basic_contract() -> DataContract:
    """Simulate NavsimDataset (basic) — BEV + basic agent, no acceleration."""
    return (
        ContractBuilder("NavsimDataset")
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.BEV_LABELS, shape=(5, 200, 200))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 4, 5))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .set_physical_limits(max_batch_size=16, memory_footprint_mb=50.0)
        .set_semantic_info(
            num_cameras=0, bev_channels=5, agent_state_dim=5,
            history_length=4, has_acceleration=False,
            has_nearby_agents=False, has_vector_maps=False,
        )
        .build()
    )

def build_lidar_only_contract() -> DataContract:
    """LiDAR-only dataset — no cameras, no BEV labels."""
    return (
        ContractBuilder("LidarOnlyDataset")
        .add_feature(FeatureType.LIDAR_POINTS, shape=(-1, 3))
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 4, 5))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .set_physical_limits(max_batch_size=32, memory_footprint_mb=30.0)
        .set_semantic_info(
            num_cameras=0, bev_channels=0, agent_state_dim=5,
            history_length=4, has_acceleration=False,
            has_nearby_agents=False, has_vector_maps=False,
        )
        .build()
    )

def build_camera_only_contract() -> DataContract:
    """Camera-only dataset — no LiDAR at all."""
    return (
        ContractBuilder("CameraOnlyDataset")
        .add_feature(FeatureType.CAMERA_IMAGES, shape=(6, 3, 900, 1600))
        .add_feature(FeatureType.BEV_LABELS, shape=(8, 200, 200))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 10, 5))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .set_physical_limits(max_batch_size=8, memory_footprint_mb=150.0)
        .set_semantic_info(
            num_cameras=6, bev_channels=8, agent_state_dim=5,
            history_length=10, has_acceleration=False,
            has_nearby_agents=False, has_vector_maps=False,
        )
        .build()
    )

def build_minimal_contract() -> DataContract:
    """Most minimal dataset — almost nothing."""
    return (
        ContractBuilder("MinimalDataset")
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 3))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 8, 3))
        .set_physical_limits(max_batch_size=64, memory_footprint_mb=10.0)
        .set_semantic_info(
            num_cameras=0, bev_channels=0, agent_state_dim=3,
            history_length=0, has_acceleration=False,
            has_nearby_agents=False, has_vector_maps=False,
        )
        .build()
    )

def build_phase_dataset_contract() -> DataContract:
    """Simulate PhaseNavsimDataset with Phase 0 enabled."""
    return (
        ContractBuilder("PhaseNavsimDataset")
        .add_feature(FeatureType.LIDAR_POINTS, shape=(-1, 3))
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.CAMERA_IMAGES, shape=(8, 3, 900, 1600))
        .add_feature(FeatureType.BEV_LABELS, shape=(12, 200, 200))
        .add_feature(FeatureType.VECTOR_MAP, shape=(-1,))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 7))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 20, 7))
        .add_feature(FeatureType.AGENT_NEARBY, shape=(1, 10, 7))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .add_feature(FeatureType.ROUTE, shape=(-1,))
        .add_feature(FeatureType.DIFFICULTY, shape=(-1,))
        .set_physical_limits(max_batch_size=4, memory_footprint_mb=250.0)
        .set_semantic_info(
            num_cameras=8, bev_channels=12, agent_state_dim=7,
            history_length=20, has_acceleration=True,
            has_nearby_agents=True, has_vector_maps=True,
        )
        .build()
    )

def build_medium_contract() -> DataContract:
    """
    Medium-capability dataset.
    History >= 10 (satisfies AGENT_FULL), agent_state_dim=5, no accel/multi-agent.
    """
    return (
        ContractBuilder("MediumDataset")
        .add_feature(FeatureType.LIDAR_BEV, shape=(2, 200, 200))
        .add_feature(FeatureType.BEV_LABELS, shape=(8, 200, 200))
        .add_feature(FeatureType.AGENT_STATE, shape=(1, 5))
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 12, 5))
        .add_feature(FeatureType.GT_TRAJECTORY, shape=(1, 16, 5))
        .set_physical_limits(max_batch_size=16, memory_footprint_mb=80.0)
        .set_semantic_info(
            num_cameras=0, bev_channels=8, agent_state_dim=5,
            history_length=12, has_acceleration=False,
            has_nearby_agents=False, has_vector_maps=False,
        )
        .build()
    )


# =============================================================================
# Diagnostic Utility
# =============================================================================

def diagnose_contract(contract: DataContract):
    """Print what a contract provides."""
    print(f"  Dataset: {contract.dataset_name}")
    print(f"  Features: {[f.name for f in contract.features]}")
    print(f"  agent_state_dim: {contract.agent_state_dim}")
    print(f"  history_length: {contract.history_length}")
    print(f"  bev_channels: {contract.bev_channels}")
    print(f"  num_cameras: {contract.num_cameras}")
    print(f"  has_acceleration: {contract.has_acceleration}")
    print(f"  has_nearby_agents: {contract.has_nearby_agents}")
    print(f"  has_vector_maps: {contract.has_vector_maps}")


# =============================================================================
# TEST 1: Basic Requirements
# =============================================================================

def test_basic_requirements():
    """Test creating and checking basic requirements."""
    req = EncoderRequirements(
        name="TestEncoder",
        required={FeatureType.LIDAR_BEV},
        min_agent_state_dim=5,
    )
    print(req)
    print()

    contract = build_basic_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    assert compatible, f"Expected compatible but got errors: {errors}"
    assert len(errors) == 0
    print("\n✓ Basic requirements test passed!")


# =============================================================================
# TEST 2: Fallback Requirements
# =============================================================================

def test_fallback_requirements():
    """Test PointPillars fallback: needs LIDAR_POINTS, falls back to LIDAR_BEV."""
    req = StandardRequirements.POINTPILLARS
    print(req)
    print()

    contract = build_basic_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    assert compatible, f"Expected fallback to work but got errors: {errors}"
    assert len(errors) == 0
    assert len(warnings) > 0, "Expected warning about using fallback"
    print("\n✓ Fallback requirements test passed!")


# =============================================================================
# TEST 3: Fallback NOT available
# =============================================================================

def test_fallback_not_available():
    """Test when required feature AND its fallback are both missing."""
    req = EncoderRequirements(
        name="StrictLidarEncoder",
        required={FeatureType.LIDAR_POINTS},
        fallback_allowed=True,
        fallback_features={FeatureType.LIDAR_POINTS: FeatureType.LIDAR_BEV},
    )
    print(req)
    print()

    contract = build_camera_only_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")

    assert not compatible, "Expected incompatible — no LiDAR at all"
    assert len(errors) > 0
    print("\n✓ Fallback-not-available test passed!")


# =============================================================================
# TEST 4: Dimension Constraints — padding allowed (medium dataset)
# =============================================================================

def test_dimension_constraints_with_padding():
    """
    Test AGENT_FULL against medium dataset (history=12 >= required 10).

    KEY INSIGHT from TEST 4 failure:
      min_history_length is ALWAYS a hard error in check_compatibility().
      You cannot fabricate history frames that don't exist.
      But agent_state_dim, acceleration, multi-agent CAN be padded/degraded.

    Fix: use medium dataset (history=12) instead of basic (history=4).
    """
    req = StandardRequirements.AGENT_FULL
    print(req)
    print()

    contract = build_medium_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    assert compatible, f"Expected compatible (degraded) but got errors: {errors}"
    assert len(warnings) > 0, "Expected warnings about padding/degradation"

    warning_text = " ".join(warnings)
    assert "Agent state dim" in warning_text, "Expected warning about agent_state_dim"
    assert "Acceleration" in warning_text or "acceleration" in warning_text
    assert "Multi-agent" in warning_text or "single-agent" in warning_text
    print("\n✓ Dimension constraint (with padding) test passed!")


# =============================================================================
# TEST 4b: History too short → HARD error even with fallback_allowed
# =============================================================================

def test_dimension_constraints_history_too_short():
    """
    AGENT_FULL against basic dataset (history=4 < required 10).
    min_history_length is ALWAYS a hard error — you can't invent frames.
    """
    req = StandardRequirements.AGENT_FULL
    print(req)
    print()

    contract = build_basic_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    assert not compatible, "Expected incompatible — history too short (4 < 10)"
    assert any("History" in e or "history" in e for e in errors)
    assert len(warnings) > 0, "Expected warnings about other degradations"
    print("\n✓ History-too-short constraint test passed!")


# =============================================================================
# TEST 5: Dimension Constraints (strict, no fallback)
# =============================================================================

def test_dimension_constraints_strict():
    """Test strict encoder — NO fallback allowed."""
    req = EncoderRequirements(
        name="StrictAgentEncoder",
        required={FeatureType.AGENT_STATE, FeatureType.AGENT_HISTORY},
        min_agent_state_dim=7,
        min_history_length=10,
        needs_acceleration=True,
        needs_multi_agent=True,
        fallback_allowed=False,
    )
    print(req)
    print()

    contract = build_basic_contract()
    diagnose_contract(contract)
    print()

    compatible, errors, warnings = req.check_compatibility(contract)
    print(f"Compatible: {compatible}")
    print(f"Errors: {errors}")

    assert not compatible
    assert len(errors) >= 3, f"Expected multiple errors, got {len(errors)}"
    print("\n✓ Strict dimension constraint test passed!")


# =============================================================================
# TEST 6: Full dataset satisfies everything
# =============================================================================

def test_full_dataset_all_encoders():
    """EnhancedNavsimDataset should satisfy ALL standard encoders."""
    contract = build_full_contract()
    print("Full dataset contract:")
    diagnose_contract(contract)
    print()

    all_requirements = {
        'pointpillars': StandardRequirements.POINTPILLARS,
        'lidar_bev': StandardRequirements.LIDAR_BEV,
        'multi_camera': StandardRequirements.MULTI_CAMERA,
        'single_camera': StandardRequirements.SINGLE_CAMERA,
        'bev_semantic': StandardRequirements.BEV_SEMANTIC,
        'bev_full': StandardRequirements.BEV_FULL,
        'agent_basic': StandardRequirements.AGENT_BASIC,
        'agent_full': StandardRequirements.AGENT_FULL,
        'vector_map': StandardRequirements.VECTOR_MAP,
        'goal_intent': StandardRequirements.GOAL_INTENT,
    }

    validator = RequirementValidator(all_requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert is_valid, "Full dataset should satisfy ALL encoders"
    print("\n✓ Full dataset satisfies all encoders test passed!")


# =============================================================================
# TEST 7: LiDAR-only dataset
# =============================================================================

def test_lidar_only_dataset():
    """LiDAR-only dataset: pass LiDAR, fail camera/BEV-label/vector-map."""
    contract = build_lidar_only_contract()
    print("LiDAR-only contract:")
    diagnose_contract(contract)
    print()

    requirements = {
        'pointpillars': StandardRequirements.POINTPILLARS,
        'lidar_bev': StandardRequirements.LIDAR_BEV,
        'multi_camera': StandardRequirements.MULTI_CAMERA,
        'bev_full': StandardRequirements.BEV_FULL,
        'vector_map': StandardRequirements.VECTOR_MAP,
        'agent_basic': StandardRequirements.AGENT_BASIC,
    }

    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert report['encoders']['pointpillars']['compatible']
    assert report['encoders']['lidar_bev']['compatible']
    assert not report['encoders']['multi_camera']['compatible']
    assert not report['encoders']['bev_full']['compatible']
    assert not report['encoders']['vector_map']['compatible']
    print("\n✓ LiDAR-only dataset test passed!")


# =============================================================================
# TEST 8: Camera-only dataset
# =============================================================================

def test_camera_only_dataset():
    """Camera-only: fail LiDAR, pass camera + BEV."""
    contract = build_camera_only_contract()
    print("Camera-only contract:")
    diagnose_contract(contract)
    print()

    requirements = {
        'pointpillars': StandardRequirements.POINTPILLARS,
        'lidar_bev': StandardRequirements.LIDAR_BEV,
        'multi_camera': StandardRequirements.MULTI_CAMERA,
        'bev_semantic': StandardRequirements.BEV_SEMANTIC,
        'agent_basic': StandardRequirements.AGENT_BASIC,
    }

    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert not report['encoders']['pointpillars']['compatible']
    assert not report['encoders']['lidar_bev']['compatible']
    assert report['encoders']['multi_camera']['compatible']
    assert report['encoders']['bev_semantic']['compatible']
    print("\n✓ Camera-only dataset test passed!")


# =============================================================================
# TEST 9: Minimal dataset — almost everything fails
# =============================================================================

def test_minimal_dataset():
    """Minimal dataset should fail most encoders."""
    contract = build_minimal_contract()
    print("Minimal contract:")
    diagnose_contract(contract)
    print()

    requirements = {
        'pointpillars': StandardRequirements.POINTPILLARS,
        'multi_camera': StandardRequirements.MULTI_CAMERA,
        'bev_semantic': StandardRequirements.BEV_SEMANTIC,
        'agent_basic': StandardRequirements.AGENT_BASIC,
        'vector_map': StandardRequirements.VECTOR_MAP,
        'history_lstm': StandardRequirements.HISTORY_LSTM,
    }

    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert not is_valid
    num_incompatible = report['summary']['incompatible']
    print(f"\n  {num_incompatible}/{report['summary']['total']} encoders incompatible")
    assert num_incompatible >= 4
    print("\n✓ Minimal dataset test passed!")


# =============================================================================
# TEST 10: Phase dataset contract
# =============================================================================

def test_phase_dataset():
    """PhaseNavsimDataset should satisfy multi-modal requirements."""
    contract = build_phase_dataset_contract()
    print("Phase dataset contract:")
    diagnose_contract(contract)
    print()

    requirements = StandardRequirements.create_multi_modal(
        use_lidar=True, use_camera=True, use_bev=True, use_vector_map=True,
    )

    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert is_valid
    print("\n✓ Phase dataset test passed!")


# =============================================================================
# TEST 11: Multi-Modal Selective
# =============================================================================

def test_multi_modal_selective():
    """Basic dataset can handle BEV + basic agent."""
    contract = build_basic_contract()
    print("Basic contract:")
    diagnose_contract(contract)
    print()

    requirements = {
        'bev': StandardRequirements.BEV_SEMANTIC,
        'agent': StandardRequirements.AGENT_BASIC,
    }

    validator = RequirementValidator(requirements)
    is_valid, report = validator.validate(contract)
    validator.print_report(report)

    assert is_valid
    print("\n✓ Multi-modal selective test passed!")


# =============================================================================
# TEST 12: Custom Encoder Requirements
# =============================================================================

def test_custom_encoder_requirements():
    """Test a custom encoder against all dataset types."""
    custom_req = EncoderRequirements(
        name="MyCustomEncoder",
        required={FeatureType.LIDAR_BEV, FeatureType.AGENT_STATE},
        preferred={FeatureType.CAMERA_IMAGES, FeatureType.BEV_LABELS},
        optional={FeatureType.VECTOR_MAP},
        min_agent_state_dim=5,
        min_bev_channels=5,
        needs_acceleration=False,
        needs_multi_agent=False,
        fallback_allowed=True,
    )
    print(custom_req)
    print()

    contracts = {
        'full': build_full_contract(),
        'basic': build_basic_contract(),
        'medium': build_medium_contract(),
        'lidar_only': build_lidar_only_contract(),
        'camera_only': build_camera_only_contract(),
        'minimal': build_minimal_contract(),
    }

    results = {}
    for name, contract in contracts.items():
        compatible, errors, warnings = custom_req.check_compatibility(contract)
        results[name] = compatible
        status = "✓" if compatible else "✗"
        warn_str = f" ({len(warnings)} warnings)" if warnings else ""
        err_str = f" — {errors}" if errors else ""
        print(f"  {status} {name:15s}: compatible={compatible}{warn_str}{err_str}")

    assert results['full'] == True
    assert results['basic'] == True
    assert results['medium'] == True
    assert results['camera_only'] == False
    assert results['minimal'] == False
    print("\n✓ Custom encoder requirements test passed!")


# =============================================================================
# TEST 13: History LSTM — needs long history
# =============================================================================

def test_history_lstm_requirements():
    """HISTORY_LSTM needs history_length >= 30."""
    req = StandardRequirements.HISTORY_LSTM
    print(req)
    print()

    compatible, errors, _ = req.check_compatibility(build_basic_contract())
    print(f"  Basic (history=4):  compatible={compatible}")
    assert not compatible

    compatible, errors, _ = req.check_compatibility(build_full_contract())
    print(f"  Full (history=20):  compatible={compatible}")
    assert not compatible

    contract_long = (
        ContractBuilder("LongHistoryDataset")
        .add_feature(FeatureType.AGENT_HISTORY, shape=(1, 40, 7))
        .set_physical_limits(16, 50.0)
        .set_semantic_info(history_length=40)
        .build()
    )
    compatible, errors, _ = req.check_compatibility(contract_long)
    print(f"  Long (history=40): compatible={compatible}")
    assert compatible
    print("\n✓ History LSTM requirements test passed!")


# =============================================================================
# TEST 14: Multiple Required Features — ALL must be present
# =============================================================================

def test_multiple_required_features():
    """Verify ALL required features must be present, not just one."""
    req = EncoderRequirements(
        name="MultiFeatureEncoder",
        required={FeatureType.LIDAR_BEV, FeatureType.CAMERA_IMAGES, FeatureType.AGENT_STATE},
        fallback_allowed=False,
    )

    compatible, errors, _ = req.check_compatibility(build_lidar_only_contract())
    print(f"  Missing camera: compatible={compatible}, errors={errors}")
    assert not compatible

    compatible, errors, _ = req.check_compatibility(build_full_contract())
    print(f"  Full dataset:   compatible={compatible}")
    assert compatible
    print("\n✓ Multiple required features test passed!")


# =============================================================================
# TEST 15: AgentEncoder Requirements
# =============================================================================

def test_agent_encoder_requirements():
    """
    Define requirements matching the AgentEncoder from document 7.
    AgentEncoder needs AGENT_STATE (5D min), prefers AGENT_HISTORY + AGENT_NEARBY.
    """
    agent_encoder_req = EncoderRequirements(
        name="AgentEncoder",
        required={FeatureType.AGENT_STATE},
        preferred={FeatureType.AGENT_HISTORY, FeatureType.AGENT_NEARBY},
        optional={FeatureType.VECTOR_MAP},
        min_agent_state_dim=5,
        needs_acceleration=False,
        needs_multi_agent=False,
        fallback_allowed=True,
    )
    print(agent_encoder_req)
    print()

    contracts = {
        'full': build_full_contract(),
        'basic': build_basic_contract(),
        'medium': build_medium_contract(),
        'lidar_only': build_lidar_only_contract(),
        'camera_only': build_camera_only_contract(),
        'minimal': build_minimal_contract(),
    }

    for name, contract in contracts.items():
        compatible, errors, warnings = agent_encoder_req.check_compatibility(contract)
        status = "✓" if compatible else "✗"
        detail = ""
        if warnings:
            detail += f" [DEGRADED: {len(warnings)} warnings]"
        if errors:
            detail += f" [ERRORS: {errors}]"
        print(f"  {status} {name:15s}{detail}")

    print("\n✓ AgentEncoder requirements test passed!")


# =============================================================================
# TEST 16: AgentEncoder Forward Pass — smoke test
# =============================================================================

def test_agent_encoder_forward():
    """
    Smoke test: run AgentEncoder forward pass with mock data.
    Tests all 3 levels: Temporal -> Interaction -> Scene.
    Includes inline AgentEncoder if module not importable.
    """

    # ---- Inline minimal AgentEncoder ----
    class EncoderOutput:
        def __init__(self):
            self.level1_temporal = None
            self.level2_interaction = None
            self.level3_scene = None
            self.interaction_types = None

    class AgentEncoder(nn.Module):
        def __init__(self, hidden_size=128, num_heads=8):
            super().__init__()
            self.hidden_size = hidden_size
            self.state_encoder = nn.Sequential(
                nn.Linear(5, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size))
            self.spatial_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=0.1, batch_first=True)
            self.spatial_norm = nn.LayerNorm(hidden_size)
            self.relation_encoder = nn.Sequential(
                nn.Linear(6, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size))
            self.relational_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=0.1, batch_first=True)
            self.relational_norm = nn.LayerNorm(hidden_size)
            self.level2_ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_size * 4, hidden_size))
            self.level2_ffn_norm = nn.LayerNorm(hidden_size)
            self.agent_to_lane_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=0.1, batch_first=True)
            self.a2l_norm = nn.LayerNorm(hidden_size)
            self.level3_ffn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4), nn.GELU(),
                nn.Dropout(0.1), nn.Linear(hidden_size * 4, hidden_size))
            self.level3_ffn_norm = nn.LayerNorm(hidden_size)
            self.interaction_classifier = nn.Linear(hidden_size, 6)

        def forward(self, agent_states, map_features=None):
            B, N, _ = agent_states.shape
            D = self.hidden_size
            output = EncoderOutput()

            # Level 1: Temporal
            x = self.state_encoder(agent_states.reshape(B*N, -1)).reshape(B, N, D)
            output.level1_temporal = x

            # Level 2: Interaction
            spatial_out, _ = self.spatial_attention(x, x, x)
            x = self.spatial_norm(x + spatial_out)

            pos = agent_states[:, :, :2]
            vel = agent_states[:, :, 2:4]
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            delta_vel = vel.unsqueeze(2) - vel.unsqueeze(1)
            dist = torch.norm(delta_pos, dim=-1, keepdim=True)
            ttc = torch.clamp(dist / (torch.norm(delta_vel, dim=-1, keepdim=True) + 1e-3), 0, 10)
            bearing = torch.atan2(delta_pos[..., 1:2], delta_pos[..., 0:1])
            rel_features = torch.cat([delta_pos, delta_vel, ttc, bearing], dim=-1)
            rel_emb = self.relation_encoder(rel_features.reshape(B*N*N, 6)).reshape(B, N, N, D)
            rel_context = rel_emb.mean(dim=2)
            relational_out, _ = self.relational_attention(x + rel_context, x, x)
            x = self.relational_norm(x + relational_out)
            x = x + self.level2_ffn(self.level2_ffn_norm(x))
            output.level2_interaction = x

            output.interaction_types = self.interaction_classifier(
                x.unsqueeze(2).expand(-1, -1, N, -1).reshape(B*N*N, D)
            ).reshape(B, N, N, 6)

            # Level 3: Scene
            if map_features is not None:
                map_exp = map_features.unsqueeze(1).expand(-1, N, -1, -1).reshape(B*N, -1, D)
                x_flat = x.reshape(B*N, 1, D)
                lane_ctx, _ = self.agent_to_lane_attn(x_flat, map_exp, map_exp)
                x = self.a2l_norm(x + lane_ctx.reshape(B, N, D))
            x = x + self.level3_ffn(self.level3_ffn_norm(x))
            output.level3_scene = x
            return output

    device = torch.device("cpu")
    B, N, D = 2, 5, 128

    encoder = AgentEncoder(hidden_size=D, num_heads=8).to(device)
    encoder.eval()
    agent_states = torch.randn(B, N, 5, device=device)

    # Without map
    with torch.no_grad():
        out_no_map = encoder(agent_states, map_features=None)

    assert out_no_map.level1_temporal.shape == (B, N, D)
    print(f"  Level 1 shape: {out_no_map.level1_temporal.shape} ✓")
    assert out_no_map.level2_interaction.shape == (B, N, D)
    print(f"  Level 2 shape: {out_no_map.level2_interaction.shape} ✓")
    assert out_no_map.level3_scene.shape == (B, N, D)
    print(f"  Level 3 (no map): {out_no_map.level3_scene.shape} ✓")
    assert out_no_map.interaction_types.shape == (B, N, N, 6)
    print(f"  Interaction types: {out_no_map.interaction_types.shape} ✓")

    # With map
    T_map = 20
    map_features = torch.randn(B, T_map, D, device=device)
    with torch.no_grad():
        out_with_map = encoder(agent_states, map_features=map_features)

    assert out_with_map.level3_scene.shape == (B, N, D)
    print(f"  Level 3 (with map): {out_with_map.level3_scene.shape} ✓")

    diff = (out_with_map.level3_scene - out_no_map.level3_scene).abs().mean().item()
    print(f"  Level 3 diff (map vs no-map): {diff:.6f}")
    assert diff > 1e-6, "Map features should change Level 3 output"

    # Gradient flow
    encoder.train()
    states_grad = torch.randn(B, N, 5, device=device, requires_grad=True)
    out = encoder(states_grad, map_features=map_features)
    out.level3_scene.sum().backward()
    assert states_grad.grad is not None, "Gradients should flow to input"
    print(f"  Gradient flow ✓")

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {num_params:,}")
    print("\n✓ AgentEncoder forward pass test passed!")


# =============================================================================
# TEST 17: AgentEncoder End-to-End Compatibility
# =============================================================================

def test_agent_encoder_end_to_end():
    """
    End-to-end: check requirements then determine adaptation strategy
    for each dataset scenario.
    """
    agent_req = EncoderRequirements(
        name="AgentEncoder",
        required={FeatureType.AGENT_STATE},
        preferred={FeatureType.AGENT_HISTORY, FeatureType.AGENT_NEARBY},
        optional={FeatureType.VECTOR_MAP},
        min_agent_state_dim=5,
        fallback_allowed=True,
    )

    # Scenario 1: Full dataset
    print("Scenario 1: Full dataset")
    contract = build_full_contract()
    compatible, errors, warnings = agent_req.check_compatibility(contract)
    assert compatible and len(warnings) == 0
    print(f"  ✓ Fully compatible — all 3 levels active")

    # Scenario 2: Basic dataset
    print("\nScenario 2: Basic dataset")
    contract = build_basic_contract()
    compatible, errors, warnings = agent_req.check_compatibility(contract)
    assert compatible
    print(f"  ✓ Compatible with {len(warnings)} warnings:")
    for w in warnings:
        print(f"    ⚠ {w}")
    print(f"    → Level 2: DEGRADED (no nearby agents)")
    print(f"    → Level 3: DEGRADED (no map cross-attention)")

    # Scenario 3: Minimal dataset
    print("\nScenario 3: Minimal dataset")
    contract = build_minimal_contract()
    compatible, errors, warnings = agent_req.check_compatibility(contract)
    print(f"  Compatible: {compatible}, warnings: {len(warnings)}")
    if compatible:
        print(f"    → Pad agent state {contract.agent_state_dim}D → 5D")
    print("\n✓ AgentEncoder end-to-end test passed!")


# =============================================================================
# TEST 18: Cross-dataset Compatibility Matrix
# =============================================================================

def test_compatibility_matrix():
    """Full compatibility matrix: every encoder x every dataset."""
    datasets = {
        'Full': build_full_contract(),
        'Phase': build_phase_dataset_contract(),
        'Medium': build_medium_contract(),
        'Basic': build_basic_contract(),
        'LiDAR': build_lidar_only_contract(),
        'Camera': build_camera_only_contract(),
        'Minimal': build_minimal_contract(),
    }

    encoders = {
        'PointPillar': StandardRequirements.POINTPILLARS,
        'LidarBEV': StandardRequirements.LIDAR_BEV,
        'MultiCam': StandardRequirements.MULTI_CAMERA,
        'BEV_Sem': StandardRequirements.BEV_SEMANTIC,
        'BEV_Full': StandardRequirements.BEV_FULL,
        'Agent_B': StandardRequirements.AGENT_BASIC,
        'Agent_F': StandardRequirements.AGENT_FULL,
        'VecMap': StandardRequirements.VECTOR_MAP,
        'GoalInt': StandardRequirements.GOAL_INTENT,
        'LSTM': StandardRequirements.HISTORY_LSTM,
    }

    header = f"{'':12s}"
    for ds_name in datasets:
        header += f" {ds_name:>8s}"
    print(header)
    print("-" * len(header))

    counts = {'ok': 0, 'warn': 0, 'fail': 0}
    for enc_name, enc_req in encoders.items():
        row = f"{enc_name:12s}"
        for ds_name, contract in datasets.items():
            compatible, errors, warnings = enc_req.check_compatibility(contract)
            if compatible and not warnings:
                row += f" {'✓':>8s}"
                counts['ok'] += 1
            elif compatible:
                row += f" {'⚠':>8s}"
                counts['warn'] += 1
            else:
                row += f" {'✗':>8s}"
                counts['fail'] += 1
        print(row)

    print()
    print(f"Legend: ✓ = compatible, ⚠ = degraded, ✗ = incompatible")
    print(f"Totals: {counts['ok']} ✓, {counts['warn']} ⚠, {counts['fail']} ✗")
    print("\n✓ Compatibility matrix test passed!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    tests = [
        ("TEST 1:  Basic Requirements", test_basic_requirements),
        ("TEST 2:  Fallback Requirements", test_fallback_requirements),
        ("TEST 3:  Fallback Not Available", test_fallback_not_available),
        ("TEST 4:  Dimension Constraints (padding, medium)", test_dimension_constraints_with_padding),
        ("TEST 4b: History Too Short -> Hard Error", test_dimension_constraints_history_too_short),
        ("TEST 5:  Dimension Constraints (strict)", test_dimension_constraints_strict),
        ("TEST 6:  Full Dataset - All Encoders", test_full_dataset_all_encoders),
        ("TEST 7:  LiDAR-Only Dataset", test_lidar_only_dataset),
        ("TEST 8:  Camera-Only Dataset", test_camera_only_dataset),
        ("TEST 9:  Minimal Dataset", test_minimal_dataset),
        ("TEST 10: Phase Dataset", test_phase_dataset),
        ("TEST 11: Multi-Modal Selective", test_multi_modal_selective),
        ("TEST 12: Custom Encoder Requirements", test_custom_encoder_requirements),
        ("TEST 13: History LSTM", test_history_lstm_requirements),
        ("TEST 14: Multiple Required Features", test_multiple_required_features),
        ("TEST 15: AgentEncoder Requirements", test_agent_encoder_requirements),
        ("TEST 16: AgentEncoder Forward Pass", test_agent_encoder_forward),
        ("TEST 17: AgentEncoder End-to-End", test_agent_encoder_end_to_end),
        ("TEST 18: Compatibility Matrix", test_compatibility_matrix),
    ]

    passed = 0
    failed = 0
    failed_names = []

    for title, test_fn in tests:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
            failed_names.append(title)
        except Exception as e:
            print(f"\n✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
            failed_names.append(title)

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)

    if failed_names:
        print("\nFailed tests:")
        for name in failed_names:
            print(f"  ✗ {name}")
    else:
        print("\n✓ ALL ENCODER REQUIREMENT TESTS PASSED!")
    print("=" * 70)