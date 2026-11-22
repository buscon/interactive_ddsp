#!/usr/bin/env python3
"""
Build an ALIGNED sensor/audio dataset from the v5 Sentire recordings (Dataset_20261121_134808).

This script:
1. Reads audio files and corresponding proximity sensor curves
2. Aligns sensor values to audio frames based on the manifest start_s
3. Computes touch_time (cumulative time above threshold 0.3)
4. Splits data into train (curves 001-016) and val (curves 017-020)
5. Writes npz files with audio, sensor, touch_time, and metadata
6. Copies notes.csv with exact onset times from sensor threshold crossings
"""

import os
import sys
import csv
import numpy as np
import soundfile as sf
import json
import shutil
from pathlib import Path

# Constants
SOURCE_DIR = Path("/home/marcello/sentire/Recordings/Dataset_20261121_134808/")
OUTPUT_DIR = Path("/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5/")
MANIFEST_PATH = SOURCE_DIR / "manifest.csv"
NOTES_CSV_PATH = SOURCE_DIR / "notes.csv"

SAMPLE_RATE = 48000
BLOCK_SIZE = 480  # samples per control frame (100 Hz)
CONTROL_RATE = 100.0  # Hz
DURATION_S = 60.0  # seconds
THRESHOLD_TOUCH = 0.3  # Lower threshold for v5 (was 0.98 in v4)


def read_proximity_curve(curve_id: str) -> np.ndarray:
    """Read a proximity curve CSV and return sensor values as float32."""
    curve_path = SOURCE_DIR / "proximity_curves" / f"proximity_curve_{curve_id}.csv"
    with open(curve_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Validate: must have 6000 rows (60 seconds at 100 Hz), starting at 0.0, spaced by 0.01
    assert len(rows) == 6000, f"Curve {curve_id}: expected 6000 rows, got {len(rows)}"
    
    sensors = []
    for i, row in enumerate(rows):
        time_s = float(row["time_s"])
        sensor = float(row["sensor"])
        
        # Validate time starting at 0.0 and spacing (6000 rows = 0.00 to 59.99)
        expected_time = i * 0.01
        if abs(time_s - expected_time) > 1e-6:
            raise ValueError(f"Curve {curve_id}: time at row {i} is {time_s}, expected {expected_time}")
        
        sensors.append(sensor)
    
    return np.array(sensors, dtype=np.float32)


def align_sensor_to_audio(sensor_curve: np.ndarray, start_s: float, n_samples: int) -> np.ndarray:
    """
    Align sensor values to audio frames.
    
    Alignment rule: curve_time = audio_time - start_s
    - Audio before start_s has no sensor value: use 0.0
    - After curve ends (curve_time >= 60 s), hold the last sensor value
    
    Args:
        sensor_curve: Full 60s curve (6001 values at 100 Hz)
        start_s: Start offset from manifest
        n_samples: Number of audio samples
    
    Returns:
        Sensor array of length T = n_samples // 480
    """
    n_frames = n_samples // BLOCK_SIZE
    
    sensor_aligned = np.zeros(n_frames, dtype=np.float32)
    
    for k in range(n_frames):
        # Audio time for frame k (frame k covers samples [480k, 480k+480))
        audio_time = k * 0.01  # 100 Hz
        
        # Curve time: curve_time = audio_time - start_s
        curve_time = audio_time - start_s
        
        if curve_time < 0.0:
            # Before the Curve starts: use 0.0
            sensor_aligned[k] = 0.0
        elif curve_time >= DURATION_S:
            # After curve ends: hold last value
            sensor_aligned[k] = sensor_curve[-1]
        else:
            # Within curve bounds: interpolate to nearest row
            # Row index = round(curve_time * 100), clipped to [0, len-1]
            row_idx = int(round(curve_time * 100))
            row_idx = max(0, min(row_idx, len(sensor_curve) - 1))
            sensor_aligned[k] = sensor_curve[row_idx]
    
    return sensor_aligned


def compute_touch_time(sensor: np.ndarray) -> np.ndarray:
    """
    Compute cumulative touch_time above threshold.
    
    touch_time[k] = 0.0 if sensor_frame[k] < THRESHOLD_TOUCH
    Otherwise, count consecutive frames with sensor >= THRESHOLD, multiply by 0.01s
    Reset to 0 on any drop below THRESHOLD
    
    Args:
        sensor: Sensor values at 100 Hz (T,)
    
    Returns:
        touch_time array of same shape
    """
    T = len(sensor)
    touch_time = np.zeros(T, dtype=np.float32)
    
    consecutive = 0
    for k in range(T):
        if sensor[k] >= THRESHOLD_TOUCH:
            consecutive += 1
            touch_time[k] = consecutive * 0.01
        else:
            consecutive = 0
            touch_time[k] = 0.0
    
    return touch_time


def main():
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Touch threshold: {THRESHOLD_TOUCH}")
    
    # Verify source directory
    expected_path = "/home/marcello/sentire/Recordings/Dataset_20261121_134808"
    if str(SOURCE_DIR.resolve()) != expected_path and str(SOURCE_DIR) != expected_path + "/":
        print(f"ERROR: Source directory does not match expected path")
        print(f"  Expected: {expected_path}")
        print(f"  Got: {SOURCE_DIR}")
        sys.exit(1)
    
    # Read manifest
    print(f"Reading manifest: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest = list(reader)
    
    print(f"Manifest entries: {len(manifest)}")
    
    # Count audio files
    audio_files = list(SOURCE_DIR.glob("*_proximity_curve_*_master.aiff"))
    print(f"Audio files found: {len(audio_files)}")
    
    if len(audio_files) != 100:
        print(f"ERROR: Expected 100 audio files, found {len(audio_files)}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sound IDs (sorted order)
    sound_ids = {}
    for i, sound in enumerate(sorted(set(row["sound"] for row in manifest))):
        sound_ids[sound] = i
    
    # Process each manifest entry
    results = []
    
    # Group by sound and curve for processing
    entries_by_sound_curve = {}
    for row in manifest:
        sound = row["sound"]
        curve = row["proximity_curve"]
        file_path = SOURCE_DIR / (row["file"] + "_master.aiff")
        start_s = float(row["proximity_curve_start_s"])
        duration_s = float(row["duration_s"])
        
        key = (sound, curve)
        if key not in entries_by_sound_curve:
            entries_by_sound_curve[key] = {
                "file_path": file_path,
                "start_s": start_s,
                "duration_s": duration_s,
                "sound": sound,
                "curve": curve,
            }
    
    print(f"Unique sound x curve combinations: {len(entries_by_sound_curve)}")
    
    # Validate all start_s values and CSV files
    print("Validating CSV files and start_s values...")
    curves_checked = set()
    for entry in entries_by_sound_curve.values():
        curve = entry["curve"]
        
        # Check start_s is not missing
        if entry["start_s"] is None or entry["start_s"] == "":
            print(f"ERROR: Missing proximity_curve_start_s for {entry['sound']} {entry['curve']}")
            sys.exit(1)
        
        # Check CSV file exists and is valid (only once per curve)
        if curve not in curves_checked:
            curves_checked.add(curve)
            try:
                read_proximity_curve(curve[-3:])  # Extract "001" from "proximity_curve_001"
            except ValueError as e:
                print(f"ERROR: {e}")
                sys.exit(1)
    
    print("All validations passed.")
    
    # Process each entry
    print("\nProcessing entries...")
    for entry in entries_by_sound_curve.values():
        file_path = entry["file_path"]
        sound = entry["sound"]
        curve = entry["curve"]
        start_s = entry["start_s"]
        curve_id = curve[-3:]  # Extract "001" from "proximity_curve_001"
        
        # Read audio
        data, sr = sf.read(file_path, dtype="int16")
        if sr != SAMPLE_RATE:
            print(f"WARNING: {file_path} has sample rate {sr}, expected {SAMPLE_RATE}")
        
        # Convert stereo to mono (mean of both channels, compute in float then convert back)
        if len(data.shape) > 1 and data.shape[1] > 1:
            audio_mono = np.mean(data.astype(np.float32), axis=1).astype(np.int16)
        else:
            audio_mono = data.astype(np.int16)
        
        n_samples = len(audio_mono)
        
        # Read and align sensor
        sensor_curve = read_proximity_curve(curve_id)
        sensor = align_sensor_to_audio(sensor_curve, start_s, n_samples)
        
        # Compute touch_time
        touch_time = compute_touch_time(sensor)
        
        # Determine split
        curve_num = int(curve_id)
        if curve_num <= 16:
            split = "train"
        else:
            split = "val"
        
        # Save npz
        output_path = OUTPUT_DIR / f"{sound}_{curve}.npz"
        np.savez(
            output_path,
            audio=audio_mono,
            sensor=sensor,
            touch_time=touch_time,
            sound=sound,
            curve=curve_id,
            start_s=start_s,
            split=split,
        )
        
        T = len(sensor)
        results.append({
            "path": str(output_path),
            "sound": sound,
            "curve": curve_id,
            "split": split,
            "n_samples": n_samples,
            "T": T,
            "start_s": start_s,
        })
        
        print(f"  {sound}_{curve}: n_samples={n_samples}, T={T}, start_s={start_s}, split={split}")
    
    # Build index
    index = {
        "source_dir": str(SOURCE_DIR),
        "n_files": len(results),
        "sound_ids": sound_ids,
        "train_curves": [f"{i:03d}" for i in range(1, 17)],  # 001..016
        "val_curves": [f"{i:03d}" for i in range(17, 21)],  # 017..020
        "files": results,
    }
    
    # Save index
    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\nIndex saved to: {index_path}")
    
    # Copy notes.csv to output directory
    notes_output_path = OUTPUT_DIR / "notes.csv"
    shutil.copy2(NOTES_CSV_PATH, notes_output_path)
    print(f"Notes CSV copied to: {notes_output_path}")
    
    # Build metadata.json
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "control_rate": CONTROL_RATE,
        "block_size": BLOCK_SIZE,
        "touch_threshold": THRESHOLD_TOUCH,
        "sound_list": sorted(set(row["sound"] for row in manifest)),
        "curve_list": [f"{i:03d}" for i in range(1, 21)],
        "n_sounds": len(set(row["sound"] for row in manifest)),
        "n_curves": 20,
        "n_files": len(results),
        "train_files": sum(1 for r in results if r["split"] == "train"),
        "val_files": sum(1 for r in results if r["split"] == "val"),
        "source_dataset": "Dataset_20261121_134808",
        "notes_onset_count": 46,  # From notes.csv (46 onset rows)
    }
    
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Per sound x split counts
    counts = {}
    for r in results:
        key = (r["sound"], r["split"])
        counts[key] = counts.get(key, 0) + 1
    
    sounds = sorted(set(r["sound"] for r in results))
    splits = ["train", "val"]
    
    print(f"{'Sound':<25} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 60)
    for sound in sounds:
        total = 0
        print(f"{sound:<25}", end="")
        for split in splits:
            c = counts.get((sound, split), 0)
            total += c
            print(f"{c:>8}", end="")
        print(f"{total:>8}")
    
    print("-" * 60)
    train_total = sum(1 for r in results if r["split"] == "train")
    val_total = sum(1 for r in results if r["split"] == "val")
    print(f"{'TOTAL':<25} {train_total:>8} {val_total:>8} {len(results):>8}")
    
    # Total fraction of frames with touch_time > 0
    total_frames = sum(r["T"] for r in results)
    touch_frames = 0
    for r in results:
        data = np.load(r["path"])
        touch_frames += np.sum(data["touch_time"] > 0)
    
    touch_fraction = touch_frames / total_frames if total_frames > 0 else 0
    print(f"\nTotal frames: {total_frames}")
    print(f"Frames with touch_time > 0: {touch_frames}")
    print(f"Fraction with touch: {touch_fraction:.4f} ({touch_fraction*100:.2f}%)")
    
    # Validation checks
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS (random 3 files)")
    print("=" * 60)
    
    # Check 3 random files
    import random
    random.seed(42)
    checked = 0
    for r in results:
        if checked >= 3:
            break
        
        data = np.load(r["path"])
        sensor = data["sensor"]
        start_s = float(data["start_s"])
        curve = str(data["curve"])
        
        print(f"\nChecking {r['path']}:")
        print(f"  Shapes: audio={data['audio'].shape}, sensor={sensor.shape}")
        
        # Check: sensor[:int(start*100)] are all exactly 0.0
        start_idx = int(start_s * 100)
        before_start = sensor[:start_idx]
        if np.all(before_start == 0.0):
            print(f"  ✓ sensor[:{start_idx}] are all 0.0")
        else:
            print(f"  ✗ ERROR: sensor[:{start_idx}] not all 0.0")
            print(f"    First non-zero at: {np.where(before_start != 0.0)[0][:5]}")
        
        # Check: sensor[int(start*100)+k] equals csv row k for k in 0..99
        csv_sensor = read_proximity_curve(curve)
        aligned_start = sensor[start_idx:start_idx + 100]
        csv_start = csv_sensor[:100]
        
        if np.allclose(aligned_start, csv_start, atol=1e-6):
            print(f"  ✓ sensor[{start_idx}:{start_idx+100}] matches CSV rows 0-99")
        else:
            diff = np.abs(aligned_start - csv_start)
            print(f"  ✗ ERROR: Mismatch in sensor alignment")
            print(f"    Max diff: {diff.max():.6f}")
            print(f"    First 5 aligned: {aligned_start[:5]}")
            print(f"    First 5 CSV: {csv_start[:5]}")
        
        # Check touch_time behavior around start
        touch = data["touch_time"]
        print(f"  Max touch_time: {touch.max():.4f}")
        print(f"  Touch frames: {np.sum(touch > 0)} / {len(touch)}")
        
        checked += 1
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
