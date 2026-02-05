"""
Test Script for Multi-Agent TransDiffuser Training Pipeline
Verifies all components work together with multi-agent support
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("Multi-Agent TransDiffuser Training Pipeline Test")
print("=" * 80)

# Test 1: Import modules
print("\n[1/7] Testing imports...")
try:
    from diffusion import create_diffusion
    from diffusion.gaussian_diffusion import GaussianDiffusion
    from datasets.testingdata import TrajectoryDataset, generate_dummy_data
    from transdiffuser.DDPM.model.dydittraj import create_transdiffuser_dit_small
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Generate dummy data
print("\n[2/7] Generating dummy multi-agent data...")
try:
    dummy_data_path = './test_data'
    os.makedirs(dummy_data_path, exist_ok=True)
    generate_dummy_data(dummy_data_path, num_scenes=20, num_agents_range=(4, 8))
    print(f"✓ Generated dummy data at {dummy_data_path}")
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load dataset
print("\n[3/7] Testing dataset loading...")
try:
    dataset = TrajectoryDataset(
        data_path=dummy_data_path,
        future_horizon=20,
        history_length=30,
        traj_dim=5,
        max_agents=32,
        modality_names=['lidar', 'img', 'BEV']
    )
    print(f"✓ Dataset loaded: {len(dataset)} scenes")
    
    # Test getitem
    batch = dataset[0]
    print(f"  - Batch keys: {list(batch.keys())}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
        elif key == 'num_agents':
            print(f"    {key}: {value}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test diffusion process
print("\n[4/7] Testing diffusion process...")
try:
    diffusion = create_diffusion(timestep_respacing="")
    
    # Create dummy multi-agent trajectories
    batch_size = 4
    num_agents = 8
    future_horizon = 20
    traj_dim = 5
    
    clean_trajectory = torch.randn(batch_size, num_agents, future_horizon, traj_dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward diffusion
    noise = torch.randn_like(clean_trajectory)
    noisy_trajectory = diffusion.q_sample(clean_trajectory, t, noise)
    print(f"✓ Forward diffusion works")
    print(f"  - Clean: {clean_trajectory.shape}")
    print(f"  - Noisy: {noisy_trajectory.shape}")
    print(f"  - Noise level (t): {t}")
    
except Exception as e:
    print(f"✗ Diffusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test model creation and forward pass
print("\n[5/7] Testing model creation and forward pass...")
try:
    # Create the actual TransDiffuser model
    model = create_transdiffuser_dit_small(
        traj_channels=5,
        learn_sigma=False,
        use_modality_specific=True,
        parallel=True,
        max_agents=32,
        future_horizon=20,
        history_length=30,
        modality_config={
            'lidar': 2,
            'img': 3,
            'BEV': 7,
        }
    )
    
    print(f"✓ Model created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy inputs matching dataset
    batch_size = 2
    num_agents = 8
    H, W = 64, 64
    
    context = {
        'lidar': torch.randn(batch_size, 2, H, W),
        'img': torch.randn(batch_size, 3, H, W),
        'BEV': torch.randn(batch_size, 7, H, W),
    }
    
    agent_states = torch.randn(batch_size, num_agents, 5)
    agent_history = torch.randn(batch_size, num_agents, 30, 5)
    noisy_trajectory = torch.randn(batch_size, num_agents, 20, 5)
    t = torch.tensor([100, 200])
    
    # Forward pass
    print(f"  - Running forward pass...")
    out, attn_masks, mlp_masks, token_masks = model(
        context, agent_states, noisy_trajectory, agent_history, t,
        complete_model=False
    )
    
    print(f"✓ Model forward pass successful")
    print(f"  - Output shape: {out.shape}")
    print(f"  - Expected: ({batch_size}, {num_agents}, 20, 5)")
    print(f"  - Attention masks: {attn_masks.shape if attn_masks is not None else None}")
    print(f"  - MLP masks: {mlp_masks.shape if mlp_masks is not None else None}")
    print(f"  - Token masks: {token_masks.shape if token_masks is not None else None}")
    
    assert out.shape == (batch_size, num_agents, 20, 5), f"Output shape mismatch!"
    print(f"  ✓ Output shape is correct!")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test DataLoader integration
print("\n[6/7] Testing DataLoader integration...")
try:
    from torch.utils.data import DataLoader
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    
    # Test one batch
    batch = next(iter(dataloader))
    print(f"✓ DataLoader works")
    print(f"  - Batch size: {batch['agent_states'].shape[0]}")
    print(f"  - Max agents per scene: {batch['agent_states'].shape[1]}")
    print(f"  - Actual agents in batch: {batch['num_agents']}")
    
    # Verify shapes
    print(f"\n  Batch shapes:")
    print(f"    - lidar: {batch['lidar'].shape}")
    print(f"    - BEV: {batch['BEV'].shape}")
    print(f"    - agent_states: {batch['agent_states'].shape}")
    print(f"    - agent_history: {batch['agent_history'].shape}")
    print(f"    - agent_future: {batch['agent_future'].shape}")
    
except Exception as e:
    print(f"✗ DataLoader test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test complete training step
print("\n[7/7] Testing complete training step...")
try:
    import torch.nn.functional as F
    
    # Get a batch from dataloader
    batch = next(iter(dataloader))
    batch_size = batch['agent_states'].shape[0]
    num_agents = batch['agent_states'].shape[1]
    
    # Prepare context
    context = {
        'lidar': batch['lidar'],
        'img': batch['img'],
        'BEV': batch['BEV'],
    }
    
    # Get agent data
    agent_states = batch['agent_states']
    agent_history = batch['agent_history']
    clean_trajectory = batch['agent_future']  # (B, N, 20, 5)
    
    # Sample random timesteps
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward diffusion (add noise)
    noise = torch.randn_like(clean_trajectory)
    noisy_trajectory = diffusion.q_sample(clean_trajectory, t, noise)
    
    print(f"  - Prepared inputs:")
    print(f"    Context modalities: {list(context.keys())}")
    print(f"    Agent states: {agent_states.shape}")
    print(f"    Agent history: {agent_history.shape}")
    print(f"    Clean trajectory: {clean_trajectory.shape}")
    print(f"    Noisy trajectory: {noisy_trajectory.shape}")
    print(f"    Timesteps: {t.shape}")
    
    # Model forward pass
    pred_noise, attn_masks, mlp_masks, token_masks = model(
        context, agent_states, noisy_trajectory, agent_history, t,
        complete_model=False
    )
    
    print(f"  - Model prediction: {pred_noise.shape}")
    
    # Compute diffusion loss (predict noise)
    diffusion_loss = F.mse_loss(pred_noise, noise)
    
    # Compute mask regularization losses
    mask_loss = 0.0
    if attn_masks is not None:
        mask_loss += attn_masks.mean() * 0.01
    if mlp_masks is not None:
        mask_loss += mlp_masks.mean() * 0.01
    if token_masks is not None:
        mask_loss += token_masks.mean() * 0.01
    
    total_loss = diffusion_loss + mask_loss
    
    # Backward pass (just to check gradients)
    total_loss.backward()
    
    print(f"✓ Training step simulation successful")
    print(f"  - Diffusion loss: {diffusion_loss.item():.6f}")
    print(f"  - Mask regularization: {mask_loss:.6f}")
    print(f"  - Total loss: {total_loss.item():.6f}")
    print(f"  - Gradients computed successfully")
    
    # Check gradient flow
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  - Parameters with gradients: {has_grad}/{total_params}")
    
except Exception as e:
    print(f"✗ Training step test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nSummary:")
print("  - Multi-agent data generation: ✓")
print("  - Multi-agent dataset loading: ✓")
print("  - Diffusion process: ✓")
print("  - Model creation and forward pass: ✓")
print("  - DataLoader integration: ✓")
print("  - Complete training step: ✓")
print("\nThe pipeline is ready for training!")
# ```

# ## **Key Changes:**

# ### **Dataset Updates:**
# 1. **Multi-agent support**: All trajectories now have agent dimension `(N, ...)`
# 2. **Separate components**: `agent_states`, `agent_history`, `agent_future`
# 3. **Padding/truncation**: Handles variable number of agents per scene
# 4. **Better normalization**: Separate stats for states and futures
# 5. **Metadata tracking**: Records actual number of agents

# ### **Test Script Updates:**
# 1. **Multi-agent data generation**: Creates scenes with 4-8 agents
# 2. **Shape verification**: Tests all `(B, N, T, D)` dimensions
# 3. **Complete training loop**: End-to-end test with backpropagation
# 4. **Gradient checking**: Verifies gradient flow through model

# ### **Generated Data Structure:**
# ```
# test_data/
#   scene_0000/
#     lidar.npy          # (2, 64, 64)
#     img.npy            # (3, 64, 64)
#     BEV.npy            # (7, 64, 64)
#     agent_states.npy   # (N, 5)
#     agent_history.npy  # (N, 30, 5)
#     agent_future.npy   # (N, 20, 5)
#     metadata.json      # {num_agents: N, ...}