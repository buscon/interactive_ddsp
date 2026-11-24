#!/usr/bin/env python3
"""
Quick render with touch dynamics enhancement.
Applies touch_time-dependent gain to simulate touch_amp_head behavior.
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
OUTPUT = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/02_forestPad_dynamics'

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

# FIX: Zero swarm LFO
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

# DYNAMICS FIX: touch_time → gain mapping
# Simulates what touch_amp_head would do: more touch = louder
def apply_touch_dynamics(audio, touch_time, gain_min=1.0, gain_max=2.0, touch_sat=10.0):
    """
    Apply time-varying gain based on touch_time.
    - touch_time=0 → gain_min
    - touch_time≥touch_sat → gain_max
    """
    # Normalize touch to 0-1 range
    touch_norm = np.clip(touch_time / touch_sat, 0, 1)
    # Map to gain
    gain = gain_min + (gain_max - gain_min) * touch_norm
    # Apply frame-by-frame (need to resample gain to audio length)
    audio_len = len(audio)
    touch_len = len(touch_time)
    if touch_len != audio_len:
        # Resample gain to audio length
        from scipy.interpolate import interp1d
        f = interp1d(np.arange(touch_len), gain, kind='linear', fill_value='extrapolate')
        gain = f(np.linspace(0, touch_len-1, audio_len))
    return audio * gain

# Render forestPad curves (best performer)
forestpad_curves = ['003', '007', '014']  # short, medium, long touch

for curve in forestpad_curves:
    print(f"\nRendering 02_forestPad_{curve} with enhanced dynamics...")
    
    # Get full file
    item = dataset.full('02_forestPad', curve)
    audio_target = item['audio']
    sensor = item['sensor']
    touch_time = item['touch_raw']
    sound_id = item['sound_id']
    
    # Convert to tensors
    sensor_t = torch.FloatTensor(sensor).unsqueeze(0).unsqueeze(-1).cuda()
    touch_t = torch.FloatTensor(touch_time).unsqueeze(0).unsqueeze(-1).cuda()
    sound_id_t = torch.LongTensor([sound_id]).cuda()
    
    # Forward pass
    with torch.no_grad():
        audio_pred = model(sensor_t, touch_t, sound_id_t)
        audio_pred = audio_pred.squeeze().cpu().numpy()
    
    # Apply touch dynamics enhancement
    # More touch = more gain (simulates touch_amp_head)
    audio_pred_enhanced = apply_touch_dynamics(audio_pred, touch_time, gain_min=1.0, gain_max=1.8, touch_sat=8.0)
    
    # Normalize both to target peak
    pred_peak = np.abs(audio_pred_enhanced).max()
    target_peak = np.abs(audio_target).max()
    gain = target_peak / pred_peak if pred_peak > 0 else 1.0
    audio_pred_enhanced = audio_pred_enhanced * gain
    
    # Also save original for comparison
    pred_peak_orig = np.abs(audio_pred).max()
    gain_orig = target_peak / pred_peak_orig if pred_peak_orig > 0 else 1.0
    audio_pred_orig = audio_pred * gain_orig
    
    # Apply reverb
    from scipy import signal
    audio_pred_rev = signal.convolve(audio_pred_enhanced, ir, mode='full')[:len(audio_pred_enhanced)]
    audio_pred_orig_rev = signal.convolve(audio_pred_orig, ir, mode='full')[:len(audio_pred_orig)]
    audio_target_rev = signal.convolve(audio_target, ir, mode='full')[:len(audio_target)]
    
    # Save
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_pred.wav', audio_pred_enhanced, 48000)
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_pred_orig.wav', audio_pred_orig, 48000)
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_target.wav', audio_target, 48000)
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_pred_reverb.wav', audio_pred_rev, 48000)
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_pred_orig_reverb.wav', audio_pred_orig_rev, 48000)
    sf.write(f'{OUTPUT}/02_forestPad_{curve}_target_reverb.wav', audio_target_rev, 48000)
    
    print(f"  Saved: {OUTPUT}/02_forestPad_{curve}_*.wav")
    print(f"  Peak: enhanced={pred_peak:.4f}, orig={pred_peak_orig:.4f}, target={target_peak:.4f}")

print("\n✓ Done! Files in:", OUTPUT)
print("\nCompare: *_pred.wav (enhanced dynamics) vs *_pred_orig.wav (original)")
