#!/usr/bin/env python3
"""Test if f0 freeze is actually working."""
import torch
import sys
sys.path.insert(0, '/home/marcello/Documents/interactive_ddsp')

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
CKPT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/v9_stageB_f0frozen_s0/state_final.pth'

dataset = AlignedChunkDataset(
    root=DATA_ROOT, split='val', chunk_s=10.0,
    notes_json='/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json',
    align_audio=False
)

for f in dataset.files:
    if f['sound'] == '03_harmonicSwarm' and f['curve'] == '001':
        item = dataset.full(f['sound'], f['curve'])
        break

print(f"Loaded: {f['sound']}_{f['curve']}, note median={item['note'][item['note']>0].mean():.1f} Hz")

# Load model WITH THE CURRENT CODE
ckpt = torch.load(CKPT, map_location='cpu')
model = SensorDDSP(
    f0_base_hz=[110.0, 210.0, 100.0, 110.0], n_sounds=4, hidden_size=512,
    n_harmonic=100, n_bands=65, sampling_rate=48000, block_size=480,
    sound_emb_dim=16, max_f0_octaves=2.0,
    onset_weighting={'enabled': True, 'factor': 5.0, 'pre_ms': 20, 'window_ms': 400, 'peak_ms': 40, 'touch_rate_hz': 100},
    attack_lag_s=[0.38, 0.38, 0.42, 0.39], w_lin=100.0, w_rms=10.0, touch_amp_head=True,
)
model.load_state_dict(ckpt, strict=False)
model = model.cuda()
model.eval()

sensor = torch.FloatTensor(item['sensor']).unsqueeze(0).unsqueeze(-1).cuda()
touch = torch.FloatTensor(item['touch']).unsqueeze(0).cuda()
note = torch.FloatTensor(item['note']).unsqueeze(0).unsqueeze(-1).cuda()
sound_id = torch.LongTensor([item['sound_id']]).cuda()

print(f"Note input: min={note.min():.1f}, max={note.max():.1f}, median={note.median():.1f}")

with torch.no_grad():
    audio = model(sensor, touch, sound_id, note=note)
    f0 = model._last_f0

print(f"\nf0 output: min={f0.min():.1f}, max={f0.max():.1f}, median={f0.median():.1f} Hz")
print(f"Expected: median ~178 Hz (following note)")

if f0.median() < 110:
    print("⚠️  f0 is STILL stuck at base frequency!")
    print("   The freeze code may not be executing.")
else:
    print("✓ f0 is following note input!")
