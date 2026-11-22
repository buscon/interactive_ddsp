#!/usr/bin/env python3
"""
Generate bakeoff_notes_v5.json from notes.csv onset times.

The notes.csv contains exact onset times from sensor threshold crossings.
This script converts them to the format expected by sensor_data.py:
{
  "sound_curve_id": [f0_value_at_each_frame, ...],
  ...
}

Since freq_hz is 0.0 (placeholder), we use a default f0=220 Hz at onset frames.
The onset is marked at the exact frame corresponding to onset_time_s.
"""

import csv
import json
from pathlib import Path

NOTES_CSV_PATH = Path("/home/marcello/sentire/Recordings/Dataset_20261121_134808/notes.csv")
OUTPUT_PATH = Path("/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5.json")

CONTROL_RATE = 100  # Hz
DURATION_S = 60.0
DEFAULT_F0 = 220.0  # Default f0 when no onset


def main():
    print(f"Reading notes.csv: {NOTES_CSV_PATH}")
    
    # Read notes.csv
    with open(NOTES_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        notes_rows = list(reader)
    
    print(f"Notes CSV rows: {len(notes_rows)}")
    
    # Group onsets by sound+curve
    onsets_by_key = {}
    for row in notes_rows:
        sound = row["sound"]
        curve = row["curve"]
        onset_time_s = float(row["onset_time_s"])
        freq_hz = float(row["freq_hz"])
        
        key = f"{sound}_proximity_curve_{curve[-3:]}"  # e.g., "01_canopy_proximity_curve_001"
        
        if key not in onsets_by_key:
            onsets_by_key[key] = []
        
        onsets_by_key[key].append({
            "onset_time_s": onset_time_s,
            "freq_hz": freq_hz,
        })
    
    print(f"Unique sound+curve combinations: {len(onsets_by_key)}")
    
    # Build note labels: f0 value at each frame (100 Hz, 6000 frames)
    n_frames = int(DURATION_S * CONTROL_RATE)  # 6000 frames
    
    note_labels = {}
    
    for key, onsets in onsets_by_key.items():
        # Initialize with zeros (no f0)
        f0_sequence = [0.0] * n_frames
        
        # Mark onsets with default f0
        for onset in onsets:
            onset_frame = int(round(onset["onset_time_s"] * CONTROL_RATE))
            onset_frame = max(0, min(onset_frame, n_frames - 1))
            
            # Use default f0 at onset frame
            # (In v5, freq_hz is 0.0 placeholder, so we use a constant)
            f0_sequence[onset_frame] = DEFAULT_F0
        
        note_labels[key] = f0_sequence
    
    # Also add entries for curves without onsets (all zeros)
    all_sounds = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
    for sound in all_sounds:
        for curve_num in range(1, 21):
            curve_id = f"{curve_num:03d}"
            key = f"{sound}_proximity_curve_{curve_id}"
            if key not in note_labels:
                note_labels[key] = [0.0] * n_frames
    
    print(f"Total note label entries: {len(note_labels)}")
    
    # Count onsets per sound
    onset_counts = {}
    for key, onsets in onsets_by_key.items():
        sound = key.split("_proximity_curve_")[0]
        onset_counts[sound] = onset_counts.get(sound, 0) + len(onsets)
    
    print("\nOnset counts per sound:")
    for sound, count in sorted(onset_counts.items()):
        print(f"  {sound}: {count} onsets")
    
    # Verify onset distribution
    print("\nOnset time distribution (per sound):")
    for sound in sorted(onsets_by_key.keys()):
        if sound.startswith("01_canopy") or sound.startswith("03_harmonicSwarm"):
            # Timer-driven LFO sounds
            key = sound
            onsets = onsets_by_key.get(key, [])
            if onsets:
                times = [o["onset_time_s"] for o in onsets]
                print(f"  {key}: {len(onsets)} onsets at {times}")
    
    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(note_labels, f)
    
    print(f"\nSaved to: {OUTPUT_PATH}")
    
    # Verify file size
    import os
    size = os.path.getsize(OUTPUT_PATH)
    print(f"File size: {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
