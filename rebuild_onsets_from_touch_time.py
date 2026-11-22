#!/usr/bin/env python3
"""
Rebuild onset labels from sensor array (OSC touch events).

OSC touch events are triggered when the proximity signal crosses threshold 0.98 from below.
This is NOT separately logged — it's derived from the sensor/proximity curve itself.
"""

import numpy as np
import json
import glob
import os
from pathlib import Path

DATA_DIR = Path("bakeoff_data_aligned_v5")
OUTPUT_JSON = Path("bakeoff_notes_v7.json")
CONTROL_RATE = 100  # Hz
THRESHOLD_TOUCH = 0.98  # OSC touch event threshold (from record_dataset_v5.scd)
DEFAULT_F0 = 220.0  # Placeholder frequency


def extract_touch_onsets(sensor):
    """Extract touch onset times from sensor array.
    
    Touch onset = sensor crossing THRESHOLD_TOUCH (0.98) from below
    """
    onsets = []
    for i in range(1, len(sensor)):
        # Touch starts when sensor crosses 0.98 from below
        if sensor[i-1] < THRESHOLD_TOUCH and sensor[i] >= THRESHOLD_TOUCH:
            onset_time_s = i / CONTROL_RATE
            onsets.append(onset_time_s)
    return onsets


def main():
    print(f"Rebuilding onset labels from sensor array (OSC touch events @ 0.98)...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output: {OUTPUT_JSON}")
    print()
    
    # Find all NPZ files
    npz_files = sorted(glob.glob(str(DATA_DIR / "*.npz")))
    print(f"Found {len(npz_files)} NPZ files")
    
    # Build notes dictionary
    notes = {}
    total_onsets = 0
    
    for npz_path in npz_files:
        # Extract sound_id from filename
        filename = Path(npz_path).stem  # e.g., "01_canopy_proximity_curve_001"
        
        # Load sensor array
        data = np.load(npz_path)
        sensor = data['sensor']
        
        # Extract onsets (sensor crossing 0.98 from below)
        onsets = extract_touch_onsets(sensor)
        
        # Convert to note format (6000 frames at 100Hz = 60s)
        n_frames = 6000
        note_array = [0.0] * n_frames
        
        for onset_time in onsets:
            frame = int(onset_time * CONTROL_RATE)
            if 0 <= frame < n_frames:
                note_array[frame] = DEFAULT_F0
        
        notes[filename] = note_array
        total_onsets += len(onsets)
        
        if len(onsets) > 0:
            print(f"  {filename}: {len(onsets)} onsets at {[f'{t:.2f}s' for t in onsets[:5]]}{'...' if len(onsets) > 5 else ''}")
    
    print()
    print(f"Total onsets: {total_onsets}")
    print(f"Total files: {len(notes)}")
    
    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(notes, f, indent=2)
    
    print(f"\nSaved to: {OUTPUT_JSON}")
    print(f"File size: {os.path.getsize(OUTPUT_JSON) / 1024:.1f} KB")
    
    # Verify by loading back
    with open(OUTPUT_JSON) as f:
        loaded = json.load(f)
    print(f"Verified: {len(loaded)} entries loaded back")


if __name__ == "__main__":
    main()
