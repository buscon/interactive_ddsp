#!/usr/bin/env python3
"""Simple note/f0 verification."""
import numpy as np
import json
from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
NOTES_JSON = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json'

# Config f0_base
f0_base_config = {
    '01_canopy': 110.0,
    '02_forestPad': 210.0,
    '03_harmonicSwarm': 100.0,
    '05_moss_and_mist': 110.0
}

print("=" * 80)
print("STORED NOTE VALUES")
print("=" * 80)

dataset = AlignedChunkDataset(
    root=DATA_ROOT,
    split='val',
    chunk_s=10.0,
    notes_json=NOTES_JSON,
    align_audio=False
)

for sound, f0_base in f0_base_config.items():
    # Find one curve for this sound
    for f in dataset.files:
        if f['sound'] == sound:
            item = dataset.full(f['sound'], f['curve'])
            note = item['note']
            note_voiced = note[note > 0]
            
            print(f"\n{sound}_{f['curve']}:")
            print(f"  Stored note: median={np.median(note_voiced):.1f} Hz, range=[{np.min(note_voiced):.1f}, {np.max(note_voiced):.1f}]")
            print(f"  f0_base config: {f0_base} Hz")
            print(f"  Ratio note/f0_base: {np.median(note_voiced)/f0_base:.2f}x")
            
            if abs(np.median(note_voiced) - f0_base) > 20:
                print(f"  ⚠️  f0_base differs from note median")
            break

print("\n" + "=" * 80)
