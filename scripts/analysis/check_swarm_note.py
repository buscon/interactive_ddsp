#!/usr/bin/env python3
"""Check what note values are for swarm_001."""
import numpy as np
import json
from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
NOTES_JSON = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json'

dataset = AlignedChunkDataset(
    root=DATA_ROOT, split='val', chunk_s=10.0,
    notes_json=NOTES_JSON,
    align_audio=False
)

for f in dataset.files:
    if f['sound'] == '03_harmonicSwarm' and f['curve'] == '001':
        item = dataset.full(f['sound'], f['curve'])
        break

note = item['note']
note_voiced = note[note > 0]

print(f"03_harmonicSwarm_001:")
print(f"  note stored: median={np.median(note_voiced):.1f} Hz, mean={np.mean(note_voiced):.1f} Hz")
print(f"  note range: [{np.min(note_voiced):.1f}, {np.max(note_voiced):.1f}] Hz")
print(f"  f0_base (config): 100.0 Hz")
print()
print(f"Target audio f0 (from pYIN): median=247 Hz")
print()
print(f"⚠️  NOTE DATA DOESn'T MATCH TARGET AUDIO!")
print(f"   Note says: ~178 Hz")
print(f"   Audio says: ~247 Hz")
print(f"   Difference: ~5 semitones")
