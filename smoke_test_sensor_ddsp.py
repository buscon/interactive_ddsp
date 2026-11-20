#!/usr/bin/env python
"""
Smoke test for SensorDDSP.
Tests dataset alignment, model forward pass, gradients, and provenance guard.
"""
import os
import sys
import json
import tempfile
import shutil
import numpy as np
import torch

# Use the project's Python
sys.path.insert(0, "/home/marcello/Documents/interactive_ddsp")

from engine_ddsp.sensor_data import AlignedChunkDataset, estimate_f0_base
from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import multiscale_fft, safe_log


def test_dataset_alignment():
    """Test that chunks are correctly aligned with the original npz data."""
    print("=" * 60)
    print("TEST 1: Dataset alignment")
    print("=" * 60)
    
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=1.0, sounds=["01_canopy"], seed=42)
    
    # Use full() method to get full file
    item_full = dataset.full("01_canopy", "001")
    npz = np.load("/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned/01_canopy_proximity_curve_001.npz")
    
    # Compare
    sensor_ok = np.allclose(item_full["sensor"], npz["sensor"], atol=1e-6)
    touch_ok = np.allclose(item_full["touch_raw"], npz["touch_time"], atol=1e-6)
    audio_ok = np.allclose(item_full["audio"], npz["audio"].astype(np.float32)/32768, atol=1e-6)
    
    print(f"Sensor alignment: {sensor_ok}")
    print(f"Touch alignment: {touch_ok}")
    print(f"Audio alignment: {audio_ok}")
    
    if not (sensor_ok and touch_ok and audio_ok):
        print("FAILED: Alignment mismatch")
        return False
    
    print("Dataset alignment: PASSED")
    return True


def test_provenance_guard():
    """Test that tampering with index.json raises an error."""
    print("\n" + "=" * 60)
    print("TEST 2: Provenance guard")
    print("=" * 60)
    
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    
    # Create temp copy with tampered source_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy index
        with open(os.path.join(root, "index.json")) as f:
            index = json.load(f)
        index["source_dir"] = "/tampered/path"
        
        tampered_index = os.path.join(tmpdir, "index.json")
        with open(tampered_index, "w") as f:
            json.dump(index, f)
        
        # Copy one npz file
        shutil.copy(
            "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned/01_canopy_proximity_curve_001.npz",
            tmpdir
        )
        
        # Try to create dataset
        try:
            dataset = AlignedChunkDataset(tmpdir, "train", chunk_s=1.0)
            print("FAILED: Provenance guard did not raise")
            return False
        except ValueError as e:
            if "Provenance mismatch" in str(e):
                print(f"Provenance guard raised: {e}")
                print("Provenance guard: PASSED")
                return True
            else:
                print(f"FAILED: Wrong error: {e}")
                return False


def test_forward_pass():
    """Test forward pass shape and no NaN/Inf."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward pass")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    f0_base = [100.0, 150.0, 200.0, 250.0, 300.0]
    model = SensorDDSP(f0_base).to(device)
    model.eval()
    
    # Create batch of 2 real chunks
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=10.0, sounds=["01_canopy"], seed=42)
    
    item1 = dataset[0]
    item2 = dataset[1]
    
    # Prepare tensors
    def to_tensor(x):
        return torch.from_numpy(x).unsqueeze(0).to(device)
    
    sensor = torch.cat([to_tensor(item1["sensor"]), to_tensor(item2["sensor"])], dim=0).unsqueeze(-1)
    touch = torch.cat([to_tensor(item1["touch"]), to_tensor(item2["touch"])], dim=0).unsqueeze(-1)
    sound_id = torch.tensor([item1["sound_id"], item2["sound_id"]], dtype=torch.long, device=device)
    
    print(f"Input shapes: sensor={sensor.shape}, touch={touch.shape}, sound_id={sound_id.shape}")
    
    with torch.no_grad():
        output = model(sensor, touch, sound_id)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (2, {int(10.0 * 100 * 480)}, 1)")
    
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    # 10 seconds * 100 Hz frame rate * 480 samples/frame = 480000 samples
    expected_len = int(10.0 * 100 * 480)
    shape_ok = output.shape == (2, expected_len, 1)
    
    if not shape_ok:
        print(f"FAILED: Wrong shape")
        return False
    if has_nan or has_inf:
        print("FAILED: NaN or Inf in output")
        return False
    
    print("Forward pass: PASSED")
    return True


def test_gradients():
    """Test that loss.backward() gives nonzero gradients."""
    print("\n" + "=" * 60)
    print("TEST 4: Gradients")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    f0_base = [100.0, 150.0, 200.0, 250.0, 300.0]
    model = SensorDDSP(f0_base).to(device)
    model.train()
    
    # Create batch
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=5.0, sounds=["01_canopy"], seed=42)
    
    item1 = dataset[0]
    item2 = dataset[1]
    
    def to_tensor(x):
        return torch.from_numpy(x).unsqueeze(0).to(device)
    
    sensor = torch.cat([to_tensor(item1["sensor"]), to_tensor(item2["sensor"])], dim=0).unsqueeze(-1)
    touch = torch.cat([to_tensor(item1["touch"]), to_tensor(item2["touch"])], dim=0).unsqueeze(-1)
    sound_id = torch.tensor([item1["sound_id"], item2["sound_id"]], dtype=torch.long, device=device)
    
    # Target: audio for 5 seconds = 5 * 100 * 480 = 240000 samples
    audio_target = torch.cat([
        to_tensor(item1["audio"][:int(5.0 * 100 * 480)]),
        to_tensor(item2["audio"][:int(5.0 * 100 * 480)])
    ], dim=0).unsqueeze(-1).squeeze(-1)  # (B, L)
    
    # Forward
    output = model(sensor, touch, sound_id).squeeze(-1)  # (B, L)
    
    # Compute loss
    scales = [4096, 2048, 1024, 512, 256, 128]
    overlap = 0.75
    loss = model.loss_fn(output, audio_target, scales, overlap)
    
    print(f"Loss: {loss.item():.6f}")
    
    # Backward
    model.zero_grad()
    loss.backward()
    
    # Check gradients on sensor MLP first layer
    sensor_mlp_first_weight = model.mlp_sensor[0].weight.grad
    touch_mlp_first_weight = model.mlp_touch[0].weight.grad
    emb_weight = model.sound_emb.weight.grad
    
    sensor_grad_norm = sensor_mlp_first_weight.norm().item()
    touch_grad_norm = touch_mlp_first_weight.norm().item()
    emb_grad_norm = emb_weight.norm().item()
    
    print(f"Sensor MLP grad norm: {sensor_grad_norm:.6f}")
    print(f"Touch MLP grad norm: {touch_grad_norm:.6f}")
    print(f"Embedding grad norm: {emb_grad_norm:.6f}")
    
    sensor_grad_ok = sensor_grad_norm > 0
    touch_grad_ok = touch_grad_norm > 0
    emb_grad_ok = emb_grad_norm > 0
    
    print(f"Sensor grad nonzero: {sensor_grad_ok}")
    print(f"Touch grad nonzero: {touch_grad_ok}")
    print(f"Embedding grad nonzero: {emb_grad_ok}")
    
    if not (sensor_grad_ok and touch_grad_ok and emb_grad_ok):
        print("FAILED: Some gradients are zero")
        return False
    
    print("Gradients: PASSED")
    return True


def test_sound_id_effect():
    """Test that different sound_ids produce different outputs."""
    print("\n" + "=" * 60)
    print("TEST 6: Sound ID effect")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    f0_base = [100.0, 150.0, 200.0, 250.0, 300.0]
    model = SensorDDSP(f0_base).to(device)
    model.eval()
    
    # Same sensor/touch with different sound_ids - use smaller chunk for speed
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=1.0, sounds=["01_canopy"], seed=42)
    item = dataset[0]
    
    def to_tensor(x):
        return torch.from_numpy(x).unsqueeze(0).to(device)
    
    sensor = to_tensor(item["sensor"]).unsqueeze(-1)  # (1, 100, 1)
    touch = to_tensor(item["touch"]).unsqueeze(-1)
    
    with torch.no_grad():
        out1 = model(sensor, touch, torch.tensor([0], dtype=torch.long, device=device))
        out2 = model(sensor, touch, torch.tensor([1], dtype=torch.long, device=device))
    
    diff = (out1 - out2).abs().mean().item()
    print(f"Mean abs difference: {diff:.6e}")
    
    # At init, outputs should differ
    if diff <= 1e-4:
        print("FAILED: Same output for different sound_ids")
        return False
    
    print("Sound ID effect: PASSED")
    return True


def test_optimiser_step():
    """Test one optimiser step."""
    print("\n" + "=" * 60)
    print("TEST 7: Optimiser step")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    f0_base = [100.0, 150.0, 200.0, 250.0, 300.0]
    model = SensorDDSP(f0_base).to(device)
    model.train()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")
    
    # Create batch (batch size 8)
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=10.0, sounds=["01_canopy"], seed=42)
    
    items = [dataset[i] for i in range(8)]
    
    def to_tensor(x):
        return torch.from_numpy(x).unsqueeze(0).to(device)
    
    sensor = torch.cat([to_tensor(it["sensor"]) for it in items], dim=0).unsqueeze(-1)
    touch = torch.cat([to_tensor(it["touch"]) for it in items], dim=0).unsqueeze(-1)
    sound_id = torch.tensor([it["sound_id"] for it in items], dtype=torch.long, device=device)
    
    # Audio for 10 seconds = 10 * 100 * 480 = 480000 samples
    audio_target = torch.cat([
        to_tensor(it["audio"][:int(10.0 * 100 * 480)]).squeeze(-1)
        for it in items
    ], dim=0)
    
    # Optimiser
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward
    output = model(sensor, touch, sound_id).squeeze(-1)
    
    # Compute loss
    scales = [4096, 2048, 1024, 512, 256, 128]
    overlap = 0.75
    loss = model.loss_fn(output, audio_target, scales, overlap)
    
    print(f"Initial loss: {loss.item():.6f}")
    
    # Backward and step
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    # Check loss after step
    output_after = model(sensor, touch, sound_id).squeeze(-1)
    loss_after = model.loss_fn(output_after, audio_target, scales, overlap)
    print(f"Loss after step: {loss_after.item():.6f}")
    
    # Peak GPU memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory: {peak_mem:.1f} MB")
    
    print("Optimiser step: PASSED")
    return True


def test_git_untouched():
    """Test that original files are untouched."""
    print("\n" + "=" * 60)
    print("TEST 8: Git unchanged check")
    print("=" * 60)
    
    import subprocess
    
    files_to_check = [
        "config.yaml",
        "engine_ddsp/model.py",
        "engine_ddsp/synth.py",
        "engine_ddsp/data.py",
        "train.py",
    ]
    
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"] + files_to_check,
        cwd="/home/marcello/Documents/interactive_ddsp",
        capture_output=True
    )
    
    if result.returncode == 0:
        print("All checked files are unchanged: PASSED")
        return True
    else:
        print("FAILED: Some files were modified")
        subprocess.run(["git", "diff", "HEAD", "--"] + files_to_check, cwd="/home/marcello/Documents/interactive_ddsp")
        return False


def main():
    print("SensorDDSP Smoke Test")
    print("=" * 60)
    
    # Run tests
    results = []
    
    # Test 8 first (git check)
    results.append(("Git unchanged", test_git_untouched()))
    
    # Test dataset alignment
    results.append(("Dataset alignment", test_dataset_alignment()))
    
    # Test provenance guard
    results.append(("Provenance guard", test_provenance_guard()))
    
    # Test forward pass
    results.append(("Forward pass", test_forward_pass()))
    
    # Test gradients
    results.append(("Gradients", test_gradients()))
    
    # Test sound ID effect
    results.append(("Sound ID effect", test_sound_id_effect()))
    
    # Test optimiser step
    results.append(("Optimiser step", test_optimiser_step()))
    
    # Test f0 estimation
    print("\n" + "=" * 60)
    print("f0_base estimation")
    print("=" * 60)
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    f0_base = estimate_f0_base(root, n_files=2, excerpt_s=5.0)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests PASSED")
        return 0
    else:
        print("\nSome tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
