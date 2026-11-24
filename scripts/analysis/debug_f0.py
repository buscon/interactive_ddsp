#!/usr/bin/env python3
"""Debug f0 prediction to see what's happening."""
import numpy as np
import torch
import sys
sys.path.insert(0, '/home/marcello/Documents/interactive_ddsp')

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
CKPT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/v9_stageB_fixed_s0/state_final.pth'

# Load dataset
dataset = AlignedChunkDataset(
    root=DATA_ROOT,
    split='val',
    chunk_s=10.0,
    notes_json='/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json',
    align_audio=False
)

# Get swarm item
for f in dataset.files:
    if f['sound'] == '03_harmonicSwarm' and f['curve'] == '001':
        item = dataset.full(f['sound'], f['curve'])
        break

print("=== Input data ===")
print(f"note shape: {item['note'].shape}")
print(f"note range: [{item['note'].min():.1f}, {item['note'].max():.1f}] Hz")
print(f"note non-zero: {(item['note'] > 0).sum()} / {len(item['note'])} frames")
print(f"note median (non-zero): {np.median(item['note'][item['note'] > 0]):.1f} Hz")

# Load model
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
    touch_amp_head=True,
)
model.load_state_dict(ckpt, strict=False)
model = model.cuda()
model.eval()

# Forward pass with debugging
sensor = torch.FloatTensor(item['sensor']).unsqueeze(0).unsqueeze(-1).cuda()  # (B, T, 1)
touch = torch.FloatTensor(item['touch']).unsqueeze(0).cuda()  # (B, T, 3)
note = torch.FloatTensor(item['note']).unsqueeze(0).unsqueeze(-1).cuda()  # (B, T, 1)
sound_id = torch.LongTensor([item['sound_id']]).cuda()  # (B,)

print(f"\n=== Forward pass ===")
print(f"sound_id: {sound_id.item()}")
print(f"note input to model: min={note.min():.1f}, max={note.max():.1f}, mean={note.mean():.1f}")

with torch.no_grad():
    # Manually trace f0 computation
    T = sensor.shape[1]
    
    # Get f0_base for swarm (sound_id=2)
    f0_base_sound = model.f0_base[sound_id].item()
    print(f"f0_base for swarm: {f0_base_sound} Hz")
    
    # Run model and capture intermediate values
    audio = model(sensor, touch, sound_id, note=note)
    
    # Check internal f0
    if hasattr(model, '_last_f0') and model._last_f0 is not None:
        f0_output = model._last_f0
        print(f"\nf0 output from model:")
        print(f"  shape: {f0_output.shape}")
        print(f"  min: {f0_output.min():.1f} Hz")
        print(f"  max: {f0_output.max():.1f} Hz")
        print(f"  median: {f0_output.median():.1f} Hz")
        
        # Check f0 deviation
        # f0 = note_f0 * 2^(max_f0_octaves * tanh(f0_dev))
        # If note=0, note_f0 = f0_base
        note_f0 = torch.where(note > 0, note, torch.full_like(note, f0_base_sound))
        f0_ratio = f0_output / (note_f0 + 1e-8)
        print(f"\nf0 / note_f0 ratio:")
        print(f"  median: {f0_ratio.median():.4f}")
        print(f"  This should be 2^(2*tanh(f0_dev))")
        
        # Invert to get f0_dev
        # ratio = 2^(2*tanh(dev)) => log2(ratio) = 2*tanh(dev) => dev = atanh(log2(ratio)/2)
        log2_ratio = torch.log2(f0_ratio + 1e-8)
        f0_dev_implied = torch.atanh(torch.clamp(log2_ratio / 2, -0.99, 0.99))
        print(f"\nImplied f0_deviation:")
        print(f"  median: {f0_dev_implied.median():.4f}")
        print(f"  abs mean: {f0_dev_implied.abs().mean():.4f}")

print(f"\n=== Audio output ===")
print(f"audio shape: {audio.shape}")
print(f"audio range: [{audio.min():.4f}, {audio.max():.4f}]")
