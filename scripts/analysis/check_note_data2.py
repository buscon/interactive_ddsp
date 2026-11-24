#!/usr/bin/env python3
"""
Check what note values are in the dataset for swarm.
"""
import numpy as np
import json

NOTES_JSON = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json'

# Load notes
with open(NOTES_JSON) as f:
    notes_data = json.load(f)

print("=== Check swarm curve 001 ===")
swarm_001_key = '03_harmonicSwarm_proximity_curve_001'
if swarm_001_key in notes_data:
    curve_001 = notes_data[swarm_001_key]
    print(f"Type: {type(curve_001)}")
    if isinstance(curve_001, list):
        print(f"  It's a list with {len(curve_001)} items")
        # Check first item
        if len(curve_001) > 0 and isinstance(curve_001[0], dict):
            print(f"  First item keys: {list(curve_001[0].keys())}")
            if 'note_hz' in curve_001[0]:
                note_hz = curve_001[0]['note_hz']
                print(f"  note_hz length: {len(note_hz)}")
                nonzero = [x for x in note_hz if x > 0]
                if nonzero:
                    print(f"  note_hz range: [{min(nonzero):.1f}, {max(nonzero):.1f}] Hz")
                    print(f"  Non-zero: {len(nonzero)} frames ({100*len(nonzero)/len(note_hz):.1f}%)")
                else:
                    print(f"  ⚠️  ALL note_hz values are ZERO!")
    elif isinstance(curve_001, dict):
        if 'note_hz' in curve_001:
            note_hz = curve_001['note_hz']
            print(f"note_hz length: {len(note_hz)}")
            nonzero = [x for x in note_hz if x > 0]
            if nonzero:
                print(f"Non-zero note_hz: [{min(nonzero):.1f}, {max(nonzero):.1f}] Hz")
            else:
                print(f"⚠️  ALL note_hz values are ZERO!")
else:
    print(f"⚠️  {swarm_001_key} not found!")

# Check all swarm curves
print("\n=== All swarm curves ===")
for i in range(1, 21):
    key = f'03_harmonicSwarm_proximity_curve_{i:03d}'
    if key in notes_data:
        curve = notes_data[key]
        if 'note_hz' in curve:
            note_hz = curve['note_hz']
            nonzero = [x for x in note_hz if x > 0]
            pct = 100 * len(nonzero) / len(note_hz) if len(note_hz) > 0 else 0
            median = np.median(nonzero) if nonzero else 0
            print(f"  {key}: {pct:.1f}% non-zero, median={median:.1f} Hz")
        else:
            print(f"  {key}: no note_hz")
    else:
        print(f"  {key}: not found")

# Check what the loader does
print("\n=== What the loader returns ===")
from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'

dataset = AlignedChunkDataset(
    root=DATA_ROOT,
    split='val',
    chunk_s=10.0,
    notes_json='/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json',
    align_audio=False,
    use_npz_note_hz=False  # Default: use notes_json
)

# Get a swarm item
for f in dataset.files:
    if f['sound'] == '03_harmonicSwarm':
        item = dataset.full(f['sound'], f['curve'])
        print(f"\n{f['sound']}_{f['curve']}:")
        print(f"  item['note'] shape: {item['note'].shape}")
        print(f"  item['note'] range: [{item['note'].min():.1f}, {item['note'].max():.1f}] Hz")
        print(f"  item['note'] median: {np.median(item['note'][item['note'] > 0]):.1f} Hz")
        nonzero = item['note'][item['note'] > 0]
        if len(nonzero) > 0:
            print(f"  Non-zero: {len(nonzero)} frames ({100*len(nonzero)/len(item['note']):.1f}%)")
        else:
            print(f"  ⚠️  ALL note values are ZERO!")
        break
