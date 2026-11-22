#!/usr/bin/env python3
"""
Onset Alignment Audit — Sensor vs Audio Onset Comparison

Diagnoses whether v6 dataset has misaligned onset labels by comparing
touch sensor threshold crossings (from touch_time array) vs actual audio onsets.

Decision Rules:
- Jitter <15ms AND median lag <20ms → ✅ Labels fine
- Median lag ≥20ms with low jitter → ⚠️ Shift touch features by median lag  
- Jitter ≥15ms → ❌ Rebuild onsets from audio (random error)
"""

import json
import os
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from scipy.signal import find_peaks


@dataclass
class OnsetMatch:
    file: str
    note_index: int
    sensor_onset_s: float
    audio_onset_s: float
    lag_s: float
    lag_ms: float


@dataclass
class SoundStats:
    sound_id: str
    median_lag_ms: float
    jitter_iqr_ms: float
    jitter_std_ms: float
    within_47ms_pct: float
    n_onsets: int


def detect_audio_onset(audio: np.ndarray, expected_time_s: float,
                       sample_rate: int = 48000, window_s: float = 0.2) -> float:
    """
    Detect audio onset near expected time using spectral flux.
    
    Args:
        audio: Audio signal (int16 or float)
        expected_time_s: Expected onset time from sensor
        sample_rate: Audio sample rate (default 48kHz)
        window_s: Search window ± around expected time (default 200ms)
    
    Returns:
        Detected onset time in seconds
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    center_sample = int(expected_time_s * sample_rate)
    window_samples = int(window_s * sample_rate)
    start_sample = max(0, center_sample - window_samples)
    end_sample = min(len(audio), center_sample + window_samples)
    
    if end_sample <= start_sample + 1024:
        return expected_time_s
    
    window = audio[start_sample:end_sample]
    
    # Compute STFT for spectral flux
    n_fft = 2048
    hop_length = 512
    window_sig = np.hanning(n_fft)
    
    n_frames = (len(window) - n_fft) // hop_length + 1
    if n_frames < 2:
        return expected_time_s
    
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = window[start:start + n_fft] * window_sig
        spec[:, i] = np.abs(np.fft.rfft(frame))
    
    # Compute spectral flux (sum of positive differences)
    flux = np.sum(np.maximum(0, np.diff(spec, axis=1)), axis=0)
    
    if len(flux) < 2:
        return expected_time_s
    
    # Find peak in flux
    prominence = np.std(flux) * 0.5
    peaks, properties = find_peaks(flux, prominence=max(prominence, 1e-8))
    
    # Convert frame index to time
    frame_times = start_sample / sample_rate + np.arange(len(flux)) * hop_length / sample_rate
    
    if len(peaks) == 0:
        # No clear peak, use maximum flux
        peak_idx = np.argmax(flux)
    else:
        # Use highest peak
        peak_idx = peaks[np.argmax(flux[peaks])]
    
    return frame_times[peak_idx]


def get_touch_onsets(touch_time: np.ndarray, frame_rate: float = 100.0,
                     min_gap_s: float = 0.5) -> List[float]:
    """
    Extract touch onset times from touch_time array.
    Onsets are transitions from 0 to positive values.
    
    Args:
        touch_time: Touch sensor values at each frame
        frame_rate: Frame rate in Hz (default 100)
        min_gap_s: Minimum gap between onsets to avoid duplicates (default 0.5s)
    
    Returns:
        List of onset times in seconds
    """
    onsets = []
    last_onset = -min_gap_s
    
    for i in range(1, len(touch_time)):
        if touch_time[i-1] == 0.0 and touch_time[i] > 0.0:
            onset_time = i / frame_rate
            # Avoid duplicate onsets too close together
            if onset_time - last_onset >= min_gap_s:
                onsets.append(onset_time)
                last_onset = onset_time
    
    return onsets


def analyze_file(npz_path: Path) -> List[OnsetMatch]:
    """
    Analyze one NPZ file, comparing touch sensor onsets to audio onsets.
    """
    sound_id = npz_path.stem
    data = np.load(npz_path)
    audio = data['audio']
    touch_time = data['touch_time']
    
    # Get touch sensor onsets
    touch_onsets = get_touch_onsets(touch_time)
    
    if len(touch_onsets) == 0:
        print(f"  Warning: No touch onsets found for {sound_id}")
        return []
    
    matches = []
    for note_idx, sensor_onset in enumerate(touch_onsets):
        # Detect audio onset
        audio_onset = detect_audio_onset(audio, sensor_onset)
        lag = audio_onset - sensor_onset
        
        matches.append(OnsetMatch(
            file=sound_id,
            note_index=note_idx,
            sensor_onset_s=sensor_onset,
            audio_onset_s=audio_onset,
            lag_s=lag,
            lag_ms=lag * 1000
        ))
    
    return matches


def compute_statistics(matches: List[OnsetMatch]) -> tuple:
    """
    Compute overall and per-sound statistics.
    """
    if len(matches) == 0:
        return {}, []
    
    lags_ms = [m.lag_ms for m in matches]
    lags_ms_arr = np.array(lags_ms)
    
    # Overall statistics
    median_lag = float(np.median(lags_ms_arr))
    q1, q3 = np.percentile(lags_ms_arr, [25, 75])
    jitter_iqr = float(q3 - q1)
    jitter_std = float(np.std(lags_ms_arr))
    within_47ms = float(np.mean(np.abs(lags_ms_arr) <= 47) * 100)
    
    overall = {
        'total_onsets': len(matches),
        'median_lag_ms': round(median_lag, 2),
        'jitter_iqr_ms': round(jitter_iqr, 2),
        'jitter_std_ms': round(jitter_std, 2),
        'within_47ms_pct': round(within_47ms, 1),
    }
    
    # Per-sound statistics
    sound_matches: Dict[str, List[OnsetMatch]] = {}
    for m in matches:
        if m.file not in sound_matches:
            sound_matches[m.file] = []
        sound_matches[m.file].append(m)
    
    per_sound = []
    for sound_id, sound_onsets in sorted(sound_matches.items()):
        lags = np.array([m.lag_ms for m in sound_onsets])
        q1, q3 = np.percentile(lags, [25, 75]) if len(lags) > 1 else (0, 0)
        
        per_sound.append(SoundStats(
            sound_id=sound_id,
            median_lag_ms=round(float(np.median(lags)), 2),
            jitter_iqr_ms=round(float(q3 - q1), 2),
            jitter_std_ms=round(float(np.std(lags)), 2),
            within_47ms_pct=round(float(np.mean(np.abs(lags) <= 47) * 100), 1),
            n_onsets=len(lags)
        ))
    
    return overall, per_sound


def make_decision(stats: Dict[str, Any]) -> tuple:
    """
    Apply decision rules to determine if labels are usable.
    
    Returns:
        (decision, reason) tuple
        decision: 'PASS', 'SHIFT', or 'REBUILD'
    """
    jitter = stats['jitter_iqr_ms']
    median_lag = abs(stats['median_lag_ms'])
    
    if jitter >= 15:
        return ('REBUILD', 
                f'Jitter ({jitter:.1f}ms IQR) exceeds 15ms threshold → random error, rebuild from audio')
    elif median_lag >= 20:
        return ('SHIFT',
                f'Median lag ({median_lag:.1f}ms) ≥20ms with low jitter ({jitter:.1f}ms) → systematic offset, shift touch features')
    else:
        return ('PASS',
                f'Jitter ({jitter:.1f}ms) <15ms AND median lag ({median_lag:.1f}ms) <20ms → labels are fine')


def main():
    # Paths
    data_dir = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5')
    output_dir = Path('/tmp/opencode/wt-audit')
    
    print("=" * 60)
    print("Onset Alignment Audit — Touch Sensor vs Audio Onset")
    print("=" * 60)
    
    # Find all NPZ files
    npz_files = sorted(data_dir.glob('*.npz'))
    print(f"\nFound {len(npz_files)} NPZ files")
    
    # Analyze each file
    print("\nAnalyzing files...")
    all_matches: List[OnsetMatch] = []
    
    for i, npz_path in enumerate(npz_files):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(npz_files)}...")
        
        matches = analyze_file(npz_path)
        all_matches.extend(matches)
    
    print(f"\n  Total onsets analyzed: {len(all_matches)}")
    
    # Compute statistics
    print("\nComputing statistics...")
    overall_stats, per_sound_stats = compute_statistics(all_matches)
    
    # Make decision
    decision, reason = make_decision(overall_stats)
    
    # Write CSV output
    csv_path = output_dir / 'onset_alignment_audit.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'note_index', 'sensor_onset_s', 
                                                'audio_onset_s', 'lag_s', 'lag_ms'])
        writer.writeheader()
        for m in all_matches:
            writer.writerow(asdict(m))
    print(f"\n  CSV written to {csv_path}")
    
    # Write JSON output
    json_output = {
        'summary': overall_stats,
        'decision': decision,
        'reason': reason,
        'per_sound_stats': [asdict(s) for s in per_sound_stats]
    }
    json_path = output_dir / 'onset_alignment_audit.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  JSON written to {json_path}")
    
    # Print report
    print("\n" + "=" * 60)
    print("ALIGNMENT AUDIT RESULTS")
    print("=" * 60)
    
    print("\n### Summary Statistics")
    print(f"- Total onsets analyzed: {overall_stats['total_onsets']}")
    print(f"- Median lag: {overall_stats['median_lag_ms']:.2f} ms (positive = audio after sensor)")
    print(f"- Jitter (IQR): {overall_stats['jitter_iqr_ms']:.2f} ms")
    print(f"- Jitter (std): {overall_stats['jitter_std_ms']:.2f} ms")
    print(f"- Within ±47ms (1 STFT frame): {overall_stats['within_47ms_pct']:.1f}%")
    
    print("\n### Per-Sound Breakdown")
    print(f"{'Sound':<45} {'Median (ms)':<12} {'IQR (ms)':<12} {'Within ±47ms':<15}")
    print("-" * 84)
    for s in per_sound_stats:
        print(f"{s.sound_id:<45} {s.median_lag_ms:<12.2f} {s.jitter_iqr_ms:<12.2f} {s.within_47ms_pct:<15.1f}%")
    
    print("\n### Decision")
    decision_icon = {'PASS': '✅', 'SHIFT': '⚠️', 'REBUILD': '❌'}
    print(f"{decision_icon.get(decision, '?')} {decision}")
    print(f"\n{reason}")
    
    print("\n### Recommendation")
    if decision == 'PASS':
        print("Proceed with onset-weighted loss. Touch sensor labels are sufficiently accurate.")
    elif decision == 'SHIFT':
        print("Apply systematic time shift to touch features before training.")
        print(f"Shift amount: {overall_stats['median_lag_ms']:.1f} ms")
    else:
        print("Rebuild onset labels from audio analysis. Random jitter is too high for simple correction.")
    
    print("\n" + "=" * 60)
    
    return overall_stats, decision


if __name__ == '__main__':
    main()
