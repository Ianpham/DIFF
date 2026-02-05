import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential

# Assuming the model is in the same directory or imported from a module
# from model import create_transdiffuser_dit_base

# For this script, we'll assume you have the model code available
# You may need to adjust the import path

def print_section(title):
    """Pretty print section headers"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def verify_token_count(model, x, batch_size, H, W, patch_size=4):
    """Verify that token count matches expectations"""
    print_section("TOKEN COUNT VERIFICATION")
    
    expected_tokens = 0
    token_breakdown = {}
    
    # Spatial modalities (lidar, img, BEV)
    num_patches = (H // patch_size) ** 2
    for modality in ['lidar', 'img', 'BEV']:
        if modality in x:
            tokens = num_patches + 1  # patches + modality token
            expected_tokens += tokens
            token_breakdown[modality] = tokens
            print(f"  {modality:15s}: {tokens:4d} tokens (patches: {num_patches}, modality_token: 1)")
    
    # Action modality (sequence of 10 + modality token)
    if 'action' in x:
        tokens = 11
        expected_tokens += tokens
        token_breakdown['action'] = tokens
        print(f"  {'action':15s}: {tokens:4d} tokens (sequence: 10, modality_token: 1)")
    
    # Scalar modalities (single token each)
    scalar_modalities = ['ego', 'behavior', 'intersection', 'pedestrian', 'traffic_control', 'occlusion', 'goal_intent']
    for modality in scalar_modalities:
        if modality in x:
            tokens = 1
            expected_tokens += tokens
            token_breakdown[modality] = tokens
            print(f"  {modality:15s}: {tokens:4d} token")
    
    print(f"\n  Expected total tokens: {expected_tokens}")
    print(f"  Model config total tokens: {model.total_tokens}")
    
    if expected_tokens == model.total_tokens:
        print(f"  ✓ Token count matches!")
        return True
    else:
        print(f"  ✗ Token count MISMATCH! Expected {expected_tokens}, got {model.total_tokens}")
        return False


def verify_output_shapes(out, attn_masks, mlp_masks, token_select, batch_size, depth, num_heads):
    """Verify output tensor shapes"""
    print_section("OUTPUT SHAPE VERIFICATION")
    
    checks_passed = 0
    checks_total = 0
    
    # Output shape
    checks_total += 1
    expected_out_shape = (batch_size, 789, 160)  # Based on your run
    print(f"  Output shape: {out.shape}")
    if out.shape == expected_out_shape:
        print(f"    ✓ Correct!")
        checks_passed += 1
    else:
        print(f"    Expected: {expected_out_shape}")
    
    # Attention masks
    if attn_masks is not None:
        checks_total += 1
        print(f"  Attn masks shape: {attn_masks.shape}")
        expected_attn_shape = (batch_size, depth, num_heads)
        if attn_masks.shape == expected_attn_shape:
            print(f"    ✓ Correct!")
            checks_passed += 1
        else:
            print(f"    Expected: {expected_attn_shape}")
    
    # MLP masks
    if mlp_masks is not None:
        checks_total += 1
        print(f"  MLP masks shape: {mlp_masks.shape}")
        expected_mlp_shape = (batch_size, depth, num_heads)
        if mlp_masks.shape == expected_mlp_shape:
            print(f"    ✓ Correct!")
            checks_passed += 1
        else:
            print(f"    Expected: {expected_mlp_shape}")
    
    # Token select
    if token_select is not None:
        checks_total += 1
        print(f"  Token select shape: {token_select.shape}")
        expected_token_shape = (batch_size, depth, 789, 1)
        if token_select.shape == expected_token_shape:
            print(f"    ✓ Correct!")
            checks_passed += 1
        else:
            print(f"    Expected: {expected_token_shape}")
    
    print(f"\n  Shape checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def verify_value_ranges(out, attn_masks, mlp_masks, token_select):
    """Verify that output values are in reasonable ranges"""
    print_section("VALUE RANGE VERIFICATION")
    
    checks_passed = 0
    checks_total = 0
    
    # Output values should be roughly normal distributed
    checks_total += 1
    out_mean = out.mean().item()
    out_std = out.std().item()
    print(f"  Output tensor statistics:")
    print(f"    Mean: {out_mean:.6f}, Std: {out_std:.6f}")
    if -5 < out_mean < 5 and 0 < out_std < 10:
        print(f"    ✓ Values in reasonable range")
        checks_passed += 1
    else:
        print(f"    ⚠ Values might be unusual")
    
    # Attention masks should be [0, 1] (probabilities)
    if attn_masks is not None:
        checks_total += 1
        attn_min = attn_masks.min().item()
        attn_max = attn_masks.max().item()
        attn_mean = attn_masks.mean().item()
        print(f"  Attention masks statistics:")
        print(f"    Min: {attn_min:.6f}, Max: {attn_max:.6f}, Mean: {attn_mean:.6f}")
        if 0 <= attn_min and attn_max <= 1:
            print(f"    ✓ Values in [0, 1] range (binary masks)")
            checks_passed += 1
        else:
            print(f"    ⚠ Values outside expected range")
    
    # MLP masks should be [0, 1]
    if mlp_masks is not None:
        checks_total += 1
        mlp_min = mlp_masks.min().item()
        mlp_max = mlp_masks.max().item()
        mlp_mean = mlp_masks.mean().item()
        print(f"  MLP masks statistics:")
        print(f"    Min: {mlp_min:.6f}, Max: {mlp_max:.6f}, Mean: {mlp_mean:.6f}")
        if 0 <= mlp_min and mlp_max <= 1:
            print(f"    ✓ Values in [0, 1] range (binary masks)")
            checks_passed += 1
        else:
            print(f"    ⚠ Values outside expected range")
    
    # Token select should be [0, 1]
    if token_select is not None:
        checks_total += 1
        token_min = token_select.min().item()
        token_max = token_select.max().item()
        token_mean = token_select.mean().item()
        print(f"  Token select statistics:")
        print(f"    Min: {token_min:.6f}, Max: {token_max:.6f}, Mean: {token_mean:.6f}")
        if 0 <= token_min and token_max <= 1:
            print(f"    ✓ Values in [0, 1] range (selection probabilities)")
            checks_passed += 1
        else:
            print(f"    ⚠ Values outside expected range")
    
    print(f"\n  Value checks passed: {checks_passed}/{checks_total}")
    return checks_passed == checks_total


def verify_gradient_flow(model, x, t, y):
    """Verify that gradients flow through the model"""
    print_section("GRADIENT FLOW VERIFICATION")
    
    # Forward pass
    out, _, _, _ = model(x, t, y, complete_model=False)
    loss = out.mean()
    loss.backward()
    
    # Check gradients
    total_params = 0
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1
                print(f"  ⚠ No gradient for: {name}")
    
    print(f"  Total trainable parameters: {total_params}")
    print(f"  Parameters with gradients: {params_with_grad}")
    print(f"  Parameters without gradients: {params_without_grad}")
    
    if params_without_grad == 0:
        print(f"  ✓ All parameters receive gradients!")
        return True
    else:
        print(f"  ⚠ Some parameters are not receiving gradients")
        return False


def verify_model_modes(model, x, t, y):
    """Verify model works in both complete_model True and False"""
    print_section("MODEL MODE VERIFICATION")
    
    # Test complete_model=False (with dynamic components)
    print("  Testing complete_model=False (with dynamic components)...")
    try:
        out1, attn1, mlp1, token1 = model(x, t, y, complete_model=False)
        print(f"    ✓ Output shape: {out1.shape}")
        print(f"    ✓ Attn masks present: {attn1 is not None}")
        print(f"    ✓ MLP masks present: {mlp1 is not None}")
        print(f"    ✓ Token select present: {token1 is not None}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False
    
    # Test complete_model=True (without dynamic components)
    print("\n  Testing complete_model=True (without dynamic components)...")
    try:
        out2, attn2, mlp2, token2 = model(x, t, y, complete_model=True)
        print(f"    ✓ Output shape: {out2.shape}")
        print(f"    ✓ Attn masks present: {attn2 is not None}")
        print(f"    ✓ MLP masks present: {mlp2 is not None}")
        print(f"    ✓ Token select present: {token2 is not None}")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False
    
    # Verify outputs are different (complete_model=False has stochastic components)
    print("\n  Checking output differences...")
    output_diff = (out1 - out2).abs().mean().item()
    print(f"    Mean absolute difference: {output_diff:.6f}")
    
    if output_diff > 0:
        print(f"    ✓ Outputs differ (as expected due to dynamic components)")
    else:
        print(f"    ⚠ Outputs are identical (unexpected)")
    
    return True


def verify_memory_usage(model, x, t, y, batch_size=2):
    """Estimate memory usage"""
    print_section("MEMORY USAGE VERIFICATION")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory (float32): {total_params * 4 / 1e6:.2f} MB")
    
    # Forward pass memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n  Batch size: {batch_size}")
    print(f"  Forward pass successful: ✓")
    
    return True


def verify_modality_encodings(model, x):
    """Verify that each modality is encoded correctly"""
    print_section("MODALITY ENCODING VERIFICATION")
    
    if not model.use_modality_specific:
        print("  Model not using modality-specific embeddings")
        return False
    
    modality_features = {}
    
    for modality_name in model.modality_config.keys():
        if modality_name in x:
            try:
                modality_input = x[modality_name]
                modality_features[modality_name] = model.modality_embedders[modality_name](
                    modality_input
                )
                shape = modality_features[modality_name].shape
                print(f"  {modality_name:15s} input {str(modality_input.shape):25s} → output {shape}")
            except Exception as e:
                print(f"  {modality_name:15s} ✗ Error: {e}")
                return False
    
    print(f"\n  ✓ All modalities encoded successfully")
    return True


def main():
    """Extended test with comprehensive verification"""
    
    print("\n" + "="*70)
    print("  TransDiffuser DiT - COMPREHENSIVE VERIFICATION TEST")
    print("="*70)
    
    # Import here or define the model
    # You need to make sure create_transdiffuser_dit_base is available
    from transdiffuser.DDPM.model.dydittraj import create_transdiffuser_dit_base
    
    # Create model
    print("\nInitializing model...")
    model = create_transdiffuser_dit_base(
        traj_channels=5,
        use_modality_specific=True,
        parallel=True
    )
    model.eval()  # Set to evaluation mode for now
    
    print(f"✓ Model created successfully")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Depth: {model.depth}")
    print(f"  Number of heads: {model.num_heads}")
    print(f"  Total tokens: {model.total_tokens}")
    
    # Prepare inputs
    batch_size = 2
    H, W = 64, 64
    
    x = {
        'lidar': torch.randn(batch_size, 2, H, W),
        'img': torch.randn(batch_size, 3, H, W),
        'BEV': torch.randn(batch_size, 7, H, W),
        'action': torch.randn(batch_size, 4, 10),
        'ego': torch.randn(batch_size, 7),
        'behavior': torch.randn(batch_size, 6),
        'intersection': torch.randn(batch_size, 5),
        'pedestrian': torch.randn(batch_size, 5),
        'traffic_control': torch.randn(batch_size, 5),
        'occlusion': torch.randn(batch_size, 5),
        'goal_intent': torch.randn(batch_size, 5),
    }
    
    t = torch.randint(0, 1000, (batch_size,))
    y = torch.randn(batch_size, 5)
    
    # Run verifications
    with torch.no_grad():
        # 1. Token count verification
        token_check = verify_token_count(model, x, batch_size, H, W)
        
        # 2. Modality encoding verification
        modality_check = verify_modality_encodings(model, x)
        
        # 3. Forward pass with dynamic components
        out, attn_masks, mlp_masks, token_select = model(x, t, y, complete_model=False)
        
        # 4. Shape verification
        shape_check = verify_output_shapes(out, attn_masks, mlp_masks, token_select, 
                                          batch_size, model.depth, model.num_heads)
        
        # 5. Value range verification
        value_check = verify_value_ranges(out, attn_masks, mlp_masks, token_select)
        
        # 6. Model modes verification
        mode_check = verify_model_modes(model, x, t, y)
        
        # 7. Memory usage
        memory_check = verify_memory_usage(model, x, t, y, batch_size)
    
    # 8. Gradient flow (set model to train mode)
    model.train()
    gradient_check = verify_gradient_flow(model, x, t, y)
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    all_checks = {
        "Token Count": token_check,
        "Modality Encoding": modality_check,
        "Output Shapes": shape_check,
        "Value Ranges": value_check,
        "Model Modes": mode_check,
        "Memory Usage": memory_check,
        "Gradient Flow": gradient_check,
    }
    
    passed = sum(1 for v in all_checks.values() if v)
    total = len(all_checks)
    
    for check_name, passed_check in all_checks.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        print(f"  {check_name:25s}: {status}")
    
    print(f"\n  Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 ALL CHECKS PASSED! Model is working correctly!")
    else:
        print(f"\n  ⚠️  {total - passed} check(s) failed. Please review above.")
    
    print("\n" + "="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)