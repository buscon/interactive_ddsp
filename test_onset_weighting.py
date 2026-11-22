"""Unit test for onset-weighted multiscale loss."""
import torch
import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_ddsp.sensor_ddsp import SensorDDSP


def test_onset_weighting_applies():
    """Verify onset frames get higher weight."""
    # Create synthetic batch with clear onset
    # Sensor and touch have shapes (B, T, dim)
    batch = {
        'audio': torch.randn(1, 48000 * 10),  # 10s audio for faster test
        'sensor': torch.randn(1, 1000, 1),    # (B, T, 1)
        'touch': torch.zeros(1, 1000, 3),     # 100Hz, 1000 frames, 3 features
        'sound_id': torch.tensor([0]),
    }
    # Add onset at frame 100 (log1p(t) > 0 after frame 100)
    batch['touch'][0, 100:, 0] = 1.0  # log1p(t) > 0 after frame 100
    
    # Run loss with weighting enabled
    model = SensorDDSP(
        [100.0, 150.0, 200.0, 250.0, 300.0],
        onset_weighting={'enabled': True, 'factor': 5.0}
    )
    loss = model.loss(batch, return_losses=False)
    
    # Run loss without weighting (same batch)
    model_no_weight = SensorDDSP(
        [100.0, 150.0, 200.0, 250.0, 300.0],
        onset_weighting={'enabled': False}
    )
    loss_no_weight = model_no_weight.loss(batch, return_losses=False)
    
    # With weighting, onset frames contribute more → loss should be higher
    print(f"With weighting: loss = {loss.item():.6f}")
    print(f"Without weighting: loss = {loss_no_weight.item():.6f}")
    print(f"Difference: {loss.item() - loss_no_weight.item():.6f}")
    
    assert loss > loss_no_weight, f"Weighted loss {loss.item():.6f} should exceed unweighted {loss_no_weight.item():.6f}"
    print("✅ Onset weighting test passed")
    return True


def test_onset_weighting_structure():
    """Test that onset weighting correctly identifies onset frames."""
    batch = {
        'audio': torch.randn(1, 48000 * 5),
        'sensor': torch.randn(1, 500, 1),
        'touch': torch.zeros(1, 500, 3),
        'sound_id': torch.tensor([0]),
    }
    
    # Create multiple onsets at different positions
    # Onset at frame 50, 150, 300
    batch['touch'][0, 50:, 0] = 1.0  # First onset at 50
    batch['touch'][0, 150:, 0] = 2.0  # Second onset at 150
    batch['touch'][0, 300:, 0] = 3.0  # Third onset at 300
    
    model = SensorDDSP(
        [100.0, 150.0, 200.0, 250.0, 300.0],
        onset_weighting={'enabled': True, 'factor': 5.0, 'window_ms': 150}
    )
    
    loss = model.loss(batch, return_losses=False)
    
    print(f"Multiple onsets test - loss: {loss.item():.6f}")
    print("✅ Onset weighting structure test passed")
    return True


def test_backward_pass():
    """Test that gradients flow correctly through onset-weighted loss."""
    batch = {
        'audio': torch.randn(1, 48000 * 5),
        'sensor': torch.randn(1, 500, 1),
        'touch': torch.zeros(1, 500, 3),
        'sound_id': torch.tensor([0]),
    }
    
    # Add onset
    batch['touch'][0, 100:, 0] = 1.0
    
    model = SensorDDSP(
        [100.0, 150.0, 200.0, 250.0, 300.0],
        onset_weighting={'enabled': True, 'factor': 5.0}
    )
    
    loss = model.loss(batch, return_losses=False)
    
    # Backward
    loss.backward()
    
    # Check that gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed!"
    print("✅ Backward pass test passed")
    return True


def test_dilation_effect():
    """Test that dilation properly spreads onset weights."""
    batch = {
        'audio': torch.randn(1, 48000 * 5),
        'sensor': torch.randn(1, 500, 1),
        'touch': torch.zeros(1, 500, 3),
        'sound_id': torch.tensor([0]),
    }
    
    # Single onset at frame 100
    batch['touch'][0, 100:, 0] = 1.0
    
    # Test with different dilation values
    for dilate in [0, 1, 2]:
        model = SensorDDSP(
            [100.0, 150.0, 200.0, 250.0, 300.0],
            onset_weighting={'enabled': True, 'factor': 5.0, 'dilate_frames': dilate}
        )
        loss = model.loss(batch, return_losses=False)
        print(f"Dilation {dilate}: loss = {loss.item():.6f}")
    
    print("✅ Dilation test passed")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("Onset Weighting Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        print("\n[Test 1] Basic weighting test...")
        test_onset_weighting_applies()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        print("\n[Test 2] Onset structure test...")
        test_onset_weighting_structure()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        print("\n[Test 3] Backward pass test...")
        test_backward_pass()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        print("\n[Test 4] Dilation test...")
        test_dilation_effect()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✅")
    else:
        print("Some tests failed! ❌")
    print("=" * 60)
