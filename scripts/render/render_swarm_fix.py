#!/usr/bin/env python3
"""
Quick render of 03_harmonicSwarm with fixed LFO modulation.
Sets lfo_depth_mul_base to 0.0 (was 0.3) so modulation doesn't start at frame 1.
"""
import os
import json
import numpy as np
import torch
import soundfile as sf

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP

# Paths
CKPT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/v9_stageB_A4_s0_corrected/state_final.pth'
DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
OUTPUT = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/03_harmonicSwarm_fixed'

os.makedirs(OUTPUT, exist_ok=True)

# Load dataset
dataset = AlignedChunkDataset(
    root=DATA_ROOT,
    split='val',
    chunk_s=10.0,
    notes_json='/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json',
    align_audio=False
)

# Load model
print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location='cpu')
model = SensorDDSP(
    f0_base_hz=[110.0, 210.0, 100.0, 110.0],
    n_sounds=4,
    hidden_size=512,
    n_harmonic=100,
    n_bands=65,
    sampling_rate=48000,
    block_size=480,
    sound_emb_dim=16,
    max_f0_octaves=2.0,
    onset_weighting={'enabled': True, 'factor': 5.0, 'pre_ms': 20, 'window_ms': 400, 'peak_ms': 40, 'touch_rate_hz': 100},
    attack_lag_s=[0.38, 0.38, 0.42, 0.39],
    w_lin=100.0,
    w_rms=10.0,
)

# FIX: Set swarm (sound_id=2) LFO depth_mul to 0
print("FIX: Setting lfo_depth_mul_base[2] (swarm) to 0.0 (was 0.3)")
model.lfo_depth_mul_base[2] = 0.0

model.load_state_dict(ckpt, strict=False)
model = model.cuda()
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Build synthetic reverb IR
def build_ir(duration_s=1.8, pre_delay_ms=20, sr=48000):
    pre_delay_samples = int(pre_delay_ms * sr / 1000)
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples) / sr
    noise = np.random.randn(n_samples)
    decay = np.exp(-t / (duration_s / 5))
    ir = noise * decay
    ir_full = np.zeros(pre_delay_samples + n_samples)
    ir_full[pre_delay_samples:] = ir
    from scipy import signal
    lp_order = 101
    lp_filter = signal.windows.hann(lp_order)
    lp_filter = lp_filter / lp_filter.sum()
    ir_full = signal.convolve(ir_full, lp_filter, mode='same')
    ir_full /= np.abs(ir_full).max()
    return ir_full

ir = build_ir()
print(f"Reverb IR: {len(ir)} samples ({len(ir)/48000:.2f}s)")

# Render swarm curves
swarm_curves = ['001', '008', '016']  # short, medium, long touch

for curve in swarm_curves:
    print(f"\nRendering 03_harmonicSwarm_{curve}...")
    
    # Get full file
    item = dataset.full('03_harmonicSwarm', curve)
    audio_target = item['audio']
    sensor = item['sensor']
    touch_time = item['touch_raw']
    sound_id = item['sound_id']
    
    # Convert to tensors
    sensor_t = torch.FloatTensor(sensor).unsqueeze(0).unsqueeze(-1).cuda()  # (1, T, 1)
    touch_t = torch.FloatTensor(touch_time).unsqueeze(0).unsqueeze(-1).cuda()  # (1, T, 1)
    sound_id_t = torch.LongTensor([sound_id]).cuda()
    
    # Forward pass
    with torch.no_grad():
        audio_pred = model(sensor_t, touch_t, sound_id_t)
        audio_pred = audio_pred.squeeze().cpu().numpy()
    
    # Normalize to target peak
    pred_peak = np.abs(audio_pred).max()
    target_peak = np.abs(audio_target).max()
    gain = target_peak / pred_peak if pred_peak > 0 else 1.0
    audio_pred = audio_pred * gain
    
    # Apply reverb
    from scipy import signal
    audio_pred_rev = signal.convolve(audio_pred, ir, mode='full')[:len(audio_pred)]
    audio_target_rev = signal.convolve(audio_target, ir, mode='full')[:len(audio_target)]
    
    # Save
    sf.write(f'{OUTPUT}/03_harmonicSwarm_{curve}_pred.wav', audio_pred, 48000)
    sf.write(f'{OUTPUT}/03_harmonicSwarm_{curve}_target.wav', audio_target, 48000)
    sf.write(f'{OUTPUT}/03_harmonicSwarm_{curve}_pred_reverb.wav', audio_pred_rev, 48000)
    sf.write(f'{OUTPUT}/03_harmonicSwarm_{curve}_target_reverb.wav', audio_target_rev, 48000)
    
    print(f"  Saved: {OUTPUT}/03_harmonicSwarm_{curve}_*.wav")
    print(f"  Peak: pred={pred_peak:.4f}, target={target_peak:.4f}, gain={gain:.4f}")

print("\n✓ Done! Files in:", OUTPUT)
