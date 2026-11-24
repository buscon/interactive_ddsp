#!/usr/bin/env python3
"""
Analyze f0/pitch difference between pred and target for swarm.
Uses librosa for pitch tracking.
"""
import numpy as np
import soundfile as sf

# Try librosa, fall back to simple autocorrelation if not available
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("librosa not available, using simple autocorrelation")

def estimate_f0_autocorr(audio, sr, frame_len=2048, hop=512, f0_min=50, f0_max=2000):
    """Simple autocorrelation-based f0 estimation."""
    f0s = []
    times = []
    
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i+frame_len]
        
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find peaks in valid lag range
        lag_min = int(sr / f0_max)
        lag_max = int(sr / f0_min)
        
        if lag_max > len(corr):
            lag_max = len(corr) - 1
        
        search_region = corr[lag_min:lag_max]
        if len(search_region) == 0:
            f0s.append(0)
            times.append(i / sr)
            continue
        
        # Find highest peak
        peak_idx = np.argmax(search_region) + lag_min
        
        if peak_idx > 0:
            f0 = sr / peak_idx
            f0s.append(f0)
        else:
            f0s.append(0)
        times.append(i / sr)
    
    return np.array(times), np.array(f0s)

FOLDER = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/03_harmonicSwarm_fix2'
curve = '001'

pred, sr = sf.read(f'{FOLDER}/03_harmonicSwarm_{curve}_pred.wav')
target, _ = sf.read(f'{FOLDER}/03_harmonicSwarm_{curve}_target.wav')

print(f"Analyzing: 03_harmonicSwarm_{curve}")
print(f"Duration: {len(pred)/sr:.1f}s")
print(f"Sample rate: {sr} Hz")
print()

if HAS_LIBROSA:
    print("Using librosa pyin for f0 estimation...")
    # Librosa pyin returns (f0, note, prob)
    result_pred = librosa.pyin(pred, fmin=80, fmax=500, sr=sr, frame_length=2048, hop_length=512)
    result_target = librosa.pyin(target, fmin=80, fmax=500, sr=sr, frame_length=2048, hop_length=512)
    f0_pred = result_pred[0] if isinstance(result_pred, tuple) else result_pred
    f0_target = result_target[0] if isinstance(result_target, tuple) else result_target
else:
    print("Using autocorrelation...")
    _, f0_pred = estimate_f0_autocorr(pred, sr)
    _, f0_target = estimate_f0_autocorr(target, sr)

# Filter out unvoiced (f0=0)
f0_pred_voiced = f0_pred[f0_pred > 0]
f0_target_voiced = f0_target[f0_target > 0]

print(f"\nF0 Statistics:")
print(f"  pred:   mean={np.mean(f0_pred_voiced):.1f} Hz, median={np.median(f0_pred_voiced):.1f} Hz, std={np.std(f0_pred_voiced):.1f} Hz")
print(f"  target: mean={np.mean(f0_target_voiced):.1f} Hz, median={np.median(f0_target_voiced):.1f} Hz, std={np.std(f0_target_voiced):.1f} Hz")
print()

# Ratio
ratio = np.median(f0_pred_voiced) / np.median(f0_target_voiced)
print(f"  pred/target ratio: {ratio:.3f}")

if ratio < 0.9:
    print(f"  ⚠️  PRED IS FLAT: ~{1/ratio:.2f}x lower than target")
    # Estimate semitones
    semitones = 12 * np.log2(1/ratio)
    print(f"     Difference: {semitones:.1f} semitones ({semitones/12:.2f} octaves)")
elif ratio > 1.1:
    print(f"  ⚠️  PRED IS SHARP: ~{ratio:.2f}x higher than target")
    semitones = 12 * np.log2(ratio)
    print(f"     Difference: {semitones:.1f} semitones")
else:
    print(f"  ✓ Pitch is similar")

print()

# Check for octave error
expected_f0 = 100.0  # swarm base f0 from config
print(f"Expected f0 (config): {expected_f0} Hz")
print(f"Target median f0: {np.median(f0_target_voiced):.1f} Hz")
print(f"Pred median f0: {np.median(f0_pred_voiced):.1f} Hz")

# Is pred an octave lower?
if np.median(f0_pred_voiced) < np.median(f0_target_voiced) * 0.6:
    print("\n⚠️  Possible octave error — pred is ~1 octave lower")
elif np.median(f0_pred_voiced) < np.median(f0_target_voiced) * 0.75:
    print("\n⚠️  Pred is significantly lower (~4-7 semitones)")

# F0 distribution
print("\nF0 Distribution (target):")
percentiles_t = np.percentile(f0_target_voiced, [10, 25, 50, 75, 90])
print(f"  10th: {percentiles_t[0]:.1f}, 25th: {percentiles_t[1]:.1f}, 50th: {percentiles_t[2]:.1f}, 75th: {percentiles_t[3]:.1f}, 90th: {percentiles_t[4]:.1f}")

print("\nF0 Distribution (pred):")
percentiles_p = np.percentile(f0_pred_voiced, [10, 25, 50, 75, 90])
print(f"  10th: {percentiles_p[0]:.1f}, 25th: {percentiles_p[1]:.1f}, 50th: {percentiles_p[2]:.1f}, 75th: {percentiles_p[3]:.1f}, 90th: {percentiles_p[4]:.1f}")
