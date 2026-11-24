#!/usr/bin/env python3
"""
Audit Audio Onsets - Verify that audio-derived onsets are consistent

This script verifies that the audio-derived onsets in bakeoff_notes_v5_audio.json
are well-distributed and reasonable for use in training.
"""

import json
import os
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from scipy.signal import find_peaks


def compute_spectral_flux(audio: np.ndarray, sample_rate: int = 48000,
                          n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute spectral flux from audio signal."""
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    n_frames = (len(audio) - n_fft) // hop_length + 1
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    
    window = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    
    flux = np.sum(np.maximum(0, np.diff(spec, axis=1)), axis=0)
    return flux


def get_sensor_onsets(touch_time: np.ndarray, frame_rate: float = 100.0,
                      min_gap_s: float = 0.5) -> List[float]:
    """Extract sensor onset times from touch_time array."""
    onsets = []
    last_onset = -min_gap_s
    
    for i in range(1, len(touch_time)):
        if touch_time[i-1] == 0.0 and touch_time[i] > 0.0:
            onset_time = i / frame_rate
            if onset_time - last_onset >= min_gap_s:
                onsets.append(onset_time)
                last_onset = onset_time
    
    return onsets


def load_audio_onsets(notes_json_path: Path) -> Dict[str, List[Dict]]:
    """Load audio-derived onsets from JSON."""
    with open(notes_json_path, 'r') as f:
        notes = json.load(f)
    
    # Convert frame-based format to onset times
    result = {}
    for sound_id, frame_list in notes.items():
        onsets = []
        for i, val in enumerate(frame_list):
            if val > 0:
                onsets.append({'onset_time_s': i / 100.0, 'freq_hz': val})
        result[sound_id] = onsets
    
    return result


@dataclass
class SoundStats:
    sound_id: str
    n_onsets: int
    median_time_s: float
    time_range_s: float
    min_gap_s: float
    max_gap_s: float


def main():
    data_dir = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5')
    audio_notes_path = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5_audio.json')
    sensor_notes_path = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5.json')
    output_dir = Path('/tmp/opencode/wt-audit')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Audio Onset Audit")
    print("=" * 60)
    
    # Load audio onsets
    audio_onsets = load_audio_onsets(audio_notes_path)
    
    # Analyze each file
    all_matches = []
    sound_stats = []
    
    npz_files = sorted(data_dir.glob('*.npz'))
    print(f"Analyzing {len(npz_files)} files...")
    
    for npz_path in npz_files:
        sound_id = npz_path.stem
        data = np.load(npz_path)
        audio = data['audio']
        touch_time = data['touch_time']
        
        # Get sensor onsets
        sensor_onsets = get_sensor_onsets(touch_time)
        
        # Get audio onsets
        audio_onset_list = audio_onsets.get(sound_id, [])
        
        # Compute audio onset statistics
        if len(audio_onset_list) > 0:
            times = [o['onset_time_s'] for o in audio_onset_list]
            times_arr = np.array(times)
            gaps = np.diff(times_arr) if len(times_arr) > 1 else np.array([0])
            
            sound_stats.append(SoundStats(
                sound_id=sound_id,
                n_onsets=len(audio_onset_list),
                median_time_s=float(np.median(times_arr)),
                time_range_s=float(np.max(times_arr) - np.min(times_arr)),
                min_gap_s=float(np.min(gaps)),
                max_gap_s=float(np.max(gaps))
            ))
        
        # Compare by index (sensor vs audio at same position)
        min_len = min(len(sensor_onsets), len(audio_onset_list))
        
        for i in range(min_len):
            sensor_onset = sensor_onsets[i]
            audio_onset_s = audio_onset_list[i]['onset_time_s']
            lag = audio_onset_s - sensor_onset
            
            all_matches.append({
                'file': sound_id,
                'note_index': i,
                'sensor_onset_s': sensor_onset,
                'audio_onset_s': audio_onset_s,
                'lag_ms': lag * 1000
            })
    
    print(f"Total audio onsets: {sum(s.n_onsets for s in sound_stats)}")
    
    # Compute overall statistics
    if len(sound_stats) > 0:
        n_onsets_arr = np.array([s.n_onsets for s in sound_stats])
        median_n = float(np.median(n_onsets_arr))
        range_n = (int(np.min(n_onsets_arr)), int(np.max(n_onsets_arr)))
        
        times_arr = np.array([s.median_time_s for s in sound_stats])
        median_time = float(np.median(times_arr))
        range_time = (float(np.min(times_arr)), float(np.max(times_arr)))
        
        gaps_arr = np.array([s.min_gap_s for s in sound_stats if s.min_gap_s > 0])
        min_gap = float(np.min(gaps_arr)) if len(gaps_arr) > 0 else 0
        max_gap = float(np.max(gaps_arr)) if len(gaps_arr) > 0 else 0
        
        print("\n=== AUDIO ONSET STATISTICS ===")
        print(f"Total files: {len(sound_stats)}")
        print(f"Total onsets: {sum(s.n_onsets for s in sound_stats)}")
        print(f"Onsets per file:")
        print(f"  Median: {median_n:.1f}")
        print(f"  Range: {range_n[0]} - {range_n[1]}")
        print(f"Median onset time: {median_time:.2f}s")
        print(f"Time range: {range_time[0]:.2f}s - {range_time[1]:.2f}s")
        print(f"Min gap between onsets: {min_gap:.2f}s")
        print(f"Max gap between onsets: {max_gap:.2f}s")
        
        # Check distribution
        print(f"\nOnset count distribution:")
        from collections import Counter
        dist = Counter(n_onsets_arr)
        for count in sorted(dist.keys()):
            pct = dist[count] / len(sound_stats) * 100
            print(f"  {count} onsets: {dist[count]} files ({pct:.1f}%)")
        
        # Check gaps
        print(f"\nGap distribution:")
        gap_dist = Counter()
        for s in sound_stats:
            if s.min_gap_s > 0:
                gap_dist[s.min_gap_s] += 1
        for gap in sorted(gap_dist.keys())[:10]:
            print(f"  {gap:.2f}s: {gap_dist[gap]} files")
        
        # Decision
        print(f"\n=== DECISION ===")
        
        # Check if onsets are reasonable
        issues = []
        
        if range_n[0] < 1:
            issues.append("Some files have 0 onsets")
        if range_n[1] > 10:
            issues.append("Some files have >10 onsets (possibly too many)")
        if min_gap < 0.1:
            issues.append("Some files have onsets <0.1s apart")
        if np.std(times_arr) < 5:
            issues.append("Onset times are too clustered")
        
        if len(issues) == 0:
            print("✅ Audio onsets appear reasonable for training")
            print(f"  - Consistent onset count ({median_n:.1f} median)")
            print(f"  - Well-distributed across time")
            print(f"  - Minimum gap {min_gap:.2f}s between onsets")
        else:
            print("⚠️ Audio onsets have potential issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    else:
        print("No files analyzed!")
        return False


if __name__ == '__main__':
    main()
