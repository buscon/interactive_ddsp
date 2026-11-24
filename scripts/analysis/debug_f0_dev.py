#!/usr/bin/env python3
"""Debug f0_deviation output."""
import numpy as np
import torch
import sys
sys.path.insert(0, '/home/marcello/Documents/interactive_ddsp')

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
CKPT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/v9_stageB_fixed_s0/state_final.pth'

dataset = AlignedChunkDataset(
    root=DATA_ROOT, split='val', chunk_s=10.0,
    notes_json='/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json',
    align_audio=False
)

for f in dataset.files:
    if f['sound'] == '03_harmonicSwarm' and f['curve'] == '001':
        item = dataset.full(f['sound'], f['curve'])
        break

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

# Manually compute f0_head output
T = sensor.shape[1]

# Reproduce forward pass up to f0_head
touch_onset = (touch[..., 0] > 0.98).float().unsqueeze(-1)  # Approximate
sensor_mlp = model.mlp_sensor(sensor)
touch_mlp = model.mlp_touch(touch)
emb = model.sound_emb(sound_id).unsqueeze(1).expand(-1, T, -1)
gru_out, _ = model.gru(torch.cat([sensor_mlp, touch_mlp, emb], dim=-1))
mlp_out = model.out_mlp(torch.cat([gru_out, sensor, touch, emb, touch_onset], dim=-1))

# f0_head input
f0_head_in = torch.cat([mlp_out, touch_onset, touch, note], dim=-1)
f0_deviation = model.f0_head(f0_head_in)

print("=== f0_head output ===")
print(f"f0_deviation: min={f0_deviation.min():.3f}, max={f0_deviation.max():.3f}, mean={f0_deviation.mean():.3f}")
print(f"f0_deviation abs mean: {f0_deviation.abs().mean():.3f}")
print(f"tanh(f0_deviation): min={torch.tanh(f0_deviation).min():.3f}, max={torch.tanh(f0_deviation).max():.3f}")

# Compute expected f0
f0_base_sound = model.f0_base[sound_id].item()
note_f0 = torch.where(note > 0, note, torch.full_like(note, f0_base_sound))
f0 = note_f0 * (2 ** (model.max_f0_octaves * torch.tanh(f0_deviation)))

print(f"\n=== Computed f0 ===")
print(f"note_f0: min={note_f0.min():.1f}, max={note_f0.max():.1f}, median={note_f0.median():.1f}")
print(f"f0: min={f0.min():.1f}, max={f0.max():.1f}, median={f0.median():.1f}")

# What SHOULD f0_deviation be to match note?
# f0 = note_f0 * 2^(2*tanh(dev)) => for f0=note_f0, we need tanh(dev)=0, so dev=0
print(f"\n=== What we need ===")
print(f"For f0 = note_f0, f0_deviation should be ~0 (tanh(0)=0, 2^0=1)")
print(f"Current f0_deviation mean: {f0_deviation.mean():.3f}")
print(f"This gives 2^(2*tanh({f0_deviation.mean():.3f})) = {2**(2*torch.tanh(torch.tensor(f0_deviation.mean())).item()):.3f}")
print(f"So f0 = note_f0 * {2**(2*torch.tanh(torch.tensor(f0_deviation.mean())).item()):.3f}")
