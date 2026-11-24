#!/usr/bin/env python3
"""
Check what note values are in the dataset for swarm.
"""
import numpy as np
import json

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
NOTES_JSON = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json'

# Load notes
with open(NOTES_JSON) as f:
    notes_data = json.load(f)

print("=== Notes JSON structure ===")
print(f"Keys: {list(notes_data.keys())}")

# Check swarm notes
swarm_key = '03_harmonicSwarm'
if swarm_key in notes_data:
    swarm_notes = notes_data[swarm_key]
    print(f"\n=== {swarm_key} ===")
    print(f"Number of curves: {len(swarm_notes)}")
    
    # Check curve 001
    if '001' in swarm_notes:
        curve_001 = swarm_notes['001']
        print(f"\nCurve 001:")
        print(f"  Keys: {list(curve_001.keys())}")
        if 'note_hz' in curve_001:
            note_hz = curve_001['note_hz']
            print(f"  note_hz length: {len(note_hz)}")
            print(f"  note_hz range: [{min(note_hz):.1f}, {max(note_hz):.1f}] Hz")
            print(f"  note_hz mean: {np.mean(note_hz):.1f} Hz")
            print(f"  note_hz median: {np.median(note_hz):.1f} Hz")
            # Non-zero values
            nonzero = [x for x in note_hz if x > 0]
            if nonzero:
                print(f"  Non-zero note_hz: [{min(nonzero):.1f}, {max(nonzero):.1f}] Hz")
            else:
                print(f"  ⚠️  ALL note_hz values are ZERO!")
        else:
            print(f"  ⚠️  No 'note_hz' key in curve 001")
    else:
        print(f"  ⚠️  Curve 001 not found in notes")
else:
    print(f"⚠️  {swarm_key} not found in notes JSON!")

# Check all sounds
print("\n=== All sounds ===")
for sound in ['01_canopy', '02_forestPad', '03_harmonicSwarm', '05_moss_and_mist']:
    if sound in notes_data:
        sound_notes = notes_data[sound]
        total_nonzero = 0
        total_frames = 0
        for curve_id, curve_data in sound_notes.items():
            if 'note_hz' in curve_data:
                note_hz = curve_data['note_hz']
                total_nonzero += sum(1 for x in note_hz if x > 0)
                total_frames += len(note_hz)
        
        pct = 100 * total_nonzero / total_frames if total_frames > 0 else 0
        print(f"{sound}: {pct:.1f}% non-zero note_hz frames")
