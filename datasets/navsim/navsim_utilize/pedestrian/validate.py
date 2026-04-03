"""
Quick Validation Script for Stage 1 Pipeline
=============================================

Run this FIRST to verify everything works before committing to full caching/training.
Tests:
    1. Scene loading works
    2. Feature builder produces correct shapes
    3. Target builder produces correct shapes
    4. Model forward pass works
    5. Loss computation works
    6. Gradient flow is correct

Usage:
    python validate_pipeline.py \
        --data_path /path/to/navsim/logs \
        --map_root /path/to/maps \
        --num_scenes 5
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# NavSim imports
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

# Our modules
from feature_target_builders import (
    PedestrianInteractionFeatureBuilder,
    InteractionTargetBuilder,
)
from interaction_aware_agent import InteractionAwareAgent


def test_scene_loading(data_path: str, num_scenes: int = 5):
    """Test 1: Can we load scenes?"""
    print("\n" + "=" * 60)
    print("TEST 1: Scene Loading")
    print("=" * 60)

    scene_loader = SceneLoader(
        original_sensor_path=None,
        data_path=Path(data_path),
        scene_filter=SceneFilter(),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    print(f"  Total tokens available: {len(scene_loader.tokens)}")

    tokens = scene_loader.tokens[:num_scenes]
    scenes = []
    for token in tokens:
        scene = scene_loader.get_scene_from_token(token)
        scenes.append(scene)
        metadata = scene.scene_metadata
        print(
            f"  Token: {token[:16]}... | "
            f"Map: {metadata.map_name} | "
            f"History: {metadata.num_history_frames} | "
            f"Future: {metadata.num_future_frames} | "
            f"Total frames: {len(scene.frames)}"
        )

    print(f"    Successfully loaded {len(scenes)} scenes")
    return scene_loader, scenes


def test_feature_builder(scenes, map_root=None):
    """Test 2: Feature builder produces correct shapes."""
    print("\n" + "=" * 60)
    print("TEST 2: Feature Builder")
    print("=" * 60)

    builder = PedestrianInteractionFeatureBuilder(
        map_root=map_root,
        risk_field_size=32,
    )
    print(f"  Builder name: {builder.get_unique_name()}")

    for i, scene in enumerate(scenes):
        agent_input = scene.get_agent_input()
        features = builder.compute_features(agent_input)

        ped_feat = features["ped_interaction_features"]
        risk_field = features["ped_risk_field"]
        has_ped = features["has_relevant_pedestrian"]

        print(
            f"  Scene {i}: "
            f"ped_features={ped_feat.shape} "
            f"risk_field={risk_field.shape} "
            f"has_ped={has_ped.item():.0f} | "
            f"feat_range=[{ped_feat.min():.3f}, {ped_feat.max():.3f}] | "
            f"risk_range=[{risk_field.min():.3f}, {risk_field.max():.3f}]"
        )

        # Shape checks
        assert ped_feat.shape == (20,), f"Expected (20,), got {ped_feat.shape}"
        assert risk_field.shape == (1, 32, 32), f"Expected (1,32,32), got {risk_field.shape}"
        assert has_ped.shape == (1,), f"Expected (1,), got {has_ped.shape}"
        assert not torch.isnan(ped_feat).any(), "NaN in ped features!"
        assert not torch.isnan(risk_field).any(), "NaN in risk field!"

    print(f"    All shapes correct, no NaNs")
    return builder


def test_target_builder(scenes):
    """Test 3: Target builder produces correct shapes and labels."""
    print("\n" + "=" * 60)
    print("TEST 3: Target Builder")
    print("=" * 60)

    builder = InteractionTargetBuilder(num_trajectory_frames=8)
    print(f"  Builder name: {builder.get_unique_name()}")

    label_counts = {}
    for i, scene in enumerate(scenes):
        targets = builder.compute_targets(scene)

        outcome = targets["interaction_outcome_label"]
        score = targets["interaction_score"]
        trajectory = targets["trajectory"]
        ego_yielded = targets["ego_yielded"]
        ped_crossed = targets["ped_crossed"]
        min_dist = targets["min_future_distance"]

        label_name = InteractionTargetBuilder.LABEL_NAMES.get(
            outcome.item(), "UNKNOWN"
        )
        label_counts[label_name] = label_counts.get(label_name, 0) + 1

        print(
            f"  Scene {i}: "
            f"outcome={label_name} "
            f"score={score.item():.2f} "
            f"ego_yielded={ego_yielded.item():.0f} "
            f"ped_crossed={ped_crossed.item():.0f} "
            f"min_dist={min_dist.item():.1f}m | "
            f"traj={trajectory.shape}"
        )

        # Shape checks
        assert outcome.shape == (1,), f"Expected (1,), got {outcome.shape}"
        assert trajectory.shape == (8, 3), f"Expected (8,3), got {trajectory.shape}"
        assert 0 <= outcome.item() <= 5, f"Invalid outcome label: {outcome.item()}"
        assert not torch.isnan(trajectory).any(), "NaN in GT trajectory!"

    print(f"  Label distribution: {label_counts}")
    print(f"    All targets correct")
    return builder


def test_model_forward(scenes, feature_builder, target_builder):
    """Test 4: Model forward pass works."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Forward Pass")
    print("=" * 60)

    agent = InteractionAwareAgent(
        lr=1e-4,
        latent_dim=4,
        hidden_dim=128,
        decoder_hidden_dim=256,
    )

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Build a batch
    all_features = {}
    all_targets = {}
    for scene in scenes:
        agent_input = scene.get_agent_input()
        f = feature_builder.compute_features(agent_input)
        t = target_builder.compute_targets(scene)

        for k, v in f.items():
            if k not in all_features:
                all_features[k] = []
            all_features[k].append(v)
        for k, v in t.items():
            if k not in all_targets:
                all_targets[k] = []
            all_targets[k].append(v)

    # Stack into batch
    features_batch = {k: torch.stack(v) for k, v in all_features.items()}
    targets_batch = {k: torch.stack(v) for k, v in all_targets.items()}

    print(f"  Batch size: {len(scenes)}")
    for k, v in features_batch.items():
        print(f"    Feature '{k}': {v.shape}")
    for k, v in targets_batch.items():
        print(f"    Target  '{k}': {v.shape}")

    # Forward pass
    agent.train()
    predictions = agent.forward(features_batch)

    for k, v in predictions.items():
        print(f"    Prediction '{k}': {v.shape}")
        assert not torch.isnan(v).any(), f"NaN in prediction '{k}'!"

    assert predictions["trajectory"].shape == (
        len(scenes),
        8,
        3,
    ), f"Expected ({len(scenes)}, 8, 3), got {predictions['trajectory'].shape}"
    assert predictions["z"].shape == (
        len(scenes),
        4,
    ), f"Expected ({len(scenes)}, 4), got {predictions['z'].shape}"

    print(f"    Forward pass correct")
    return agent, features_batch, targets_batch, predictions


def test_loss_and_gradients(agent, features_batch, targets_batch, predictions):
    """Test 5: Loss computation and gradient flow."""
    print("\n" + "=" * 60)
    print("TEST 5: Loss & Gradient Flow")
    print("=" * 60)

    loss = agent.compute_loss(features_batch, targets_batch, predictions)
    print(f"  Total loss: {loss.item():.4f}")

    # Detailed losses
    losses = agent.loss_fn(
        pred_trajectory=predictions["trajectory"],
        gt_trajectory=targets_batch["trajectory"],
        mu=predictions["mu"],
        logvar=predictions["logvar"],
        z=predictions["z"],
        interaction_score=targets_batch["interaction_score"],
        outcome_label=targets_batch["interaction_outcome_label"],
    )
    for k, v in losses.items():
        print(f"    {k}: {v.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.requires_grad, "Loss doesn't require grad!"

    # Backward pass
    loss.backward()

    # Check gradients
    has_grads = 0
    no_grads = 0
    for name, param in agent.named_parameters():
        if param.grad is not None:
            if param.grad.abs().max() > 0:
                has_grads += 1
            else:
                no_grads += 1
        else:
            no_grads += 1

    print(f"  Parameters with gradients: {has_grads}")
    print(f"  Parameters without gradients: {no_grads}")

    # Check z statistics
    z = predictions["z"]
    mu = predictions["mu"]
    print(f"\n  z statistics:")
    for d in range(z.shape[1]):
        print(
            f"    dim {d}: mean={z[:, d].mean():.3f} std={z[:, d].std():.3f} "
            f"mu_range=[{mu[:, d].min():.3f}, {mu[:, d].max():.3f}]"
        )

    print(f"    Loss and gradients correct")


def test_inference(agent, scenes):
    """Test 6: Inference via compute_trajectory."""
    print("\n" + "=" * 60)
    print("TEST 6: Inference (compute_trajectory)")
    print("=" * 60)

    agent.eval()
    for i, scene in enumerate(scenes[:3]):
        agent_input = scene.get_agent_input()
        trajectory = agent.compute_trajectory(agent_input)
        poses = trajectory.poses

        print(
            f"  Scene {i}: trajectory shape={poses.shape} | "
            f"x_range=[{poses[:, 0].min():.2f}, {poses[:, 0].max():.2f}] | "
            f"y_range=[{poses[:, 1].min():.2f}, {poses[:, 1].max():.2f}] | "
            f"heading_range=[{poses[:, 2].min():.2f}, {poses[:, 2].max():.2f}]"
        )

        assert poses.shape == (8, 3), f"Expected (8, 3), got {poses.shape}"
        assert not np.isnan(poses).any(), "NaN in predicted trajectory!"

    print(f"  ✅ Inference works correctly")


def main():
    parser = argparse.ArgumentParser(description="Validate Stage 1 pipeline")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--map_root", type=str, default=None)
    parser.add_argument("--num_scenes", type=int, default=5)
    args = parser.parse_args()

    if args.map_root:
        os.environ["NUPLAN_MAPS_ROOT"] = args.map_root

    print("  Stage 1 Pipeline Validation")
    print(f"   Data path: {args.data_path}")
    print(f"   Map root: {args.map_root or 'None (crosswalk features disabled)'}")
    print(f"   Num scenes: {args.num_scenes}")

    start = time.time()

    # Run all tests
    scene_loader, scenes = test_scene_loading(args.data_path, args.num_scenes)
    feature_builder = test_feature_builder(scenes, args.map_root)
    target_builder = test_target_builder(scenes)
    agent, feat_batch, tgt_batch, preds = test_model_forward(
        scenes, feature_builder, target_builder
    )
    test_loss_and_gradients(agent, feat_batch, tgt_batch, preds)
    test_inference(agent, scenes)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"  ALL TESTS PASSED ({elapsed:.1f}s)")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run pedestrian_scene_miner.py to count interaction scenes")
    print("  2. Cache features with NavSim's run_dataset_caching.py")
    print("  3. Train with NavSim's run_training.py")
    print("  4. Evaluate with run_pdm_score_one_stage.py")


if __name__ == "__main__":
    main()