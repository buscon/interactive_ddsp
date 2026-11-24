#!/usr/bin/env python3
"""
Rebuild Onset Labels from Audio-Derived Spectral Flux Peaks

This script replaces sensor-derived onset labels with audio-derived onsets
using spectral flux peak detection. This addresses the alignment issues
identified in the audit (median lag: -86ms, jitter: 112ms IQR).

ALGORITHM:
1. Compute spectral flux for each audio file
2. Use scipy.signal.find_peaks to detect onset peaks in flux
3. Filter peaks by prominence and minimum distance
4. Convert peak times to frame indices for output format (list of 6000 values)

OUTPUT FORMAT:
{
  "sound_curve_id": [f0_value_at_each_frame, ...],  # 6000 frames at 100Hz
  ...
}
Where f0_value is 220.0 at onset frames, 0.0 elsewhere.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks


def compute_spectral_flux(audio: np.ndarray, sample_rate: int = 48000,
                          n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Compute spectral flux from audio signal.
    
    Spectral flux = sum of positive differences in magnitude spectrogram.
    High flux indicates sudden changes (onsets).
    
    Args:
        audio: Audio signal (int16 or float32)
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        Spectral flux array (one value per STFT frame)
    """
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    # Compute STFT
    n_frames = (len(audio) - n_fft) // hop_length + 1
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    
    window = np.hanning(n_fft)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    
    # Compute spectral flux (sum of positive differences)
    flux = np.sum(np.maximum(0, np.diff(spec, axis=1)), axis=0)
    
    return flux


def detect_audio_onsets_from_flux(flux: np.ndarray, sample_rate: int = 48000,
                                   hop_length: int = 512, frame_rate: int = 100,
                                   min_distance_s: float = 0.5,
                                   prominence_std_factor: float = 2.0,
                                   max_onsets_per_file: int = 6) -> list:
    """
    Detect onsets from spectral flux using peak picking.
    
    Args:
        flux: Spectral flux array
        sample_rate: Audio sample rate
        hop_length: STFT hop length
        frame_rate: Output frame rate
        min_distance_s: Minimum time between onsets (seconds)
        prominence_std_factor: Prominence = factor * std(flux)
        max_onsets_per_file: Maximum onsets to return
    
    Returns:
        List of dicts with 'onset_time_s' and 'freq_hz'
    """
    if len(flux) < 2:
        return []
    
    # Compute prominence threshold
    flux_std = np.std(flux)
    prominence = max(flux_std * prominence_std_factor, 1e-8)
    
    # Minimum distance in flux frames
    min_distance_frames = int(min_distance_s * sample_rate / hop_length)
    
    # Find peaks with prominence
    peaks, properties = find_peaks(flux, prominence=prominence, distance=min_distance_frames)
    
    if len(peaks) == 0:
        # Fall back to global maximum if no peaks found
        peaks = [np.argmax(flux)]
    
    # Sort by flux value (highest first) and take top N
    peak_fluxes = flux[peaks]
    top_indices = np.argsort(peak_fluxes)[-max_onsets_per_file:][::-1]
    peaks = peaks[top_indices]
    
    # Convert peak frames to times
    hop_time = hop_length / sample_rate
    onset_times = sorted(peaks * hop_time)
    
    # Convert to output format
    audio_onsets = []
    for t in onset_times:
        audio_onsets.append({
            'onset_time_s': float(t),
            'freq_hz': 220.0,  # Keep same placeholder as before
        })
    
    return audio_onsets


def onsets_to_frame_list(audio_onsets: list, n_frames: int = 6000,
                          frame_rate: int = 100, default_value: float = 0.0,
                          onset_value: float = 220.0) -> list:
    """
    Convert onset times to frame-based list (6000 frames at 100Hz).
    
    Args:
        audio_onsets: List of dicts with 'onset_time_s'
        n_frames: Number of frames in output (default 6000 for 60s at 100Hz)
        frame_rate: Frame rate in Hz (default 100)
        default_value: Value for non-onset frames
        onset_value: Value for onset frames
    
    Returns:
        List of n_frames values
    """
    result = [default_value] * n_frames
    
    for onset in audio_onsets:
        # Convert time to frame index
        frame = int(round(onset['onset_time_s'] * frame_rate))
        frame = max(0, min(frame, n_frames - 1))
        
        # Mark onset frame
        result[frame] = onset_value
    
    return result


def main():
    # Paths
    data_dir = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5')
    reference_notes_path = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5.json')
    output_path = Path('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5_audio.json')
    
    print("=" * 60)
    print("Onset Rebuild from Audio - Spectral Flux Peaks")
    print("=" * 60)
    
    # Find all NPZ files
    npz_files = sorted(data_dir.glob('*.npz'))
    print(f"NPZ files found: {len(npz_files)}")
    
    # Build new notes with audio-derived onsets
    new_notes = {}
    total_audio_onsets = 0
    files_processed = 0
    
    # Parameters for peak detection
    sample_rate = 48000
    n_fft = 2048
    hop_length = 512
    frame_rate = 100
    n_frames = 6000  # 60s at 100Hz
    min_distance_s = 0.5  # 500ms min between onsets
    prominence_std_factor = 2.0  # Prominence threshold
    max_onsets_per_file = 6  # Max onsets per file
    
    for i, npz_path in enumerate(npz_files):
        sound_id = npz_path.stem
        
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(npz_files)}...")
        
        # Load NPZ file
        data = np.load(npz_path)
        audio = data['audio']  # 48kHz, 60s
        
        # Compute spectral flux
        flux = compute_spectral_flux(audio, sample_rate, n_fft, hop_length)
        
        # Detect onsets from flux peaks
        audio_onsets = detect_audio_onsets_from_flux(
            flux, sample_rate, hop_length, frame_rate,
            min_distance_s, prominence_std_factor, max_onsets_per_file
        )
        
        # Convert to frame-based format
        note_frame_list = onsets_to_frame_list(
            audio_onsets, n_frames, frame_rate
        )
        
        new_notes[sound_id] = note_frame_list
        total_audio_onsets += len(audio_onsets)
        files_processed += 1
    
    print(f"\nProcessed {files_processed} files")
    print(f"Total audio-derived onsets: {total_audio_onsets}")
    
    # Save new notes
    with open(output_path, 'w') as f:
        json.dump(new_notes, f)
    
    print(f"\nSaved to: {output_path}")
    
    # Verify file size
    import os
    size = os.path.getsize(output_path)
    print(f"File size: {size / 1024:.1f} KB")
    
    # Summary
    print("\n" + "=" * 60)
    print("REBUILD SUMMARY")
    print("=" * 60)
    print(f"Files processed: {files_processed}")
    print(f"Total onsets (audio-derived): {total_audio_onsets}")
    print(f"Output file: {output_path}")
    print("\nParameters used:")
    print(f"  - n_fft: {n_fft}")
    print(f"  - hop_length: {hop_length} ({hop_length/sample_rate*1000:.1f}ms)")
    print(f"  - min_distance: {min_distance_s}s ({min_distance_s*frame_rate} frames)")
    print(f"  - prominence_factor: {prominence_std_factor}*std")
    print(f"  - max_onsets_per_file: {max_onsets_per_file}")
    
    # Sample verification
    print("\nSample verification:")
    sample_key = list(new_notes.keys())[0]
    sample = new_notes[sample_key]
    onset_frames = [i for i, v in enumerate(sample) if v > 0]
    print(f"  {sample_key}: {len(onset_frames)} onsets at frames {onset_frames[:5]}...")
    
    return new_notes


if __name__ == '__main__':
    main()
