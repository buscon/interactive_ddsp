#!/usr/bin/env python3
"""Analyze f0 for f0frozen eval."""
import numpy as np
import soundfile as sf
import librosa

FOLDER = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_f0frozen'
curve = '001'

pred, sr = sf.read(f'{FOLDER}/03_harmonicSwarm_{curve}_pred.wav')
target, _ = sf.read(f'{FOLDER}/03_harmonicSwarm_{curve}_target.wav')

print(f"Analyzing: 03_harmonicSwarm_{curve} (f0frozen eval)")
print(f"Duration: {len(pred)/sr:.1f}s")

result_pred = librosa.pyin(pred, fmin=80, fmax=500, sr=sr, frame_length=2048, hop_length=512)
result_target = librosa.pyin(target, fmin=80, fmax=500, sr=sr, frame_length=2048, hop_length=512)
f0_pred = result_pred[0] if isinstance(result_pred, tuple) else result_pred
f0_target = result_target[0] if isinstance(result_target, tuple) else result_target

f0_pred_voiced = f0_pred[f0_pred > 0]
f0_target_voiced = f0_target[f0_target > 0]

print(f"\nF0 Statistics:")
print(f"  pred:   mean={np.mean(f0_pred_voiced):.1f} Hz, median={np.median(f0_pred_voiced):.1f} Hz")
print(f"  target: mean={np.mean(f0_target_voiced):.1f} Hz, median={np.median(f0_target_voiced):.1f} Hz")

ratio = np.median(f0_pred_voiced) / np.median(f0_target_voiced)
print(f"\n  pred/target ratio: {ratio:.3f}")

if 0.9 <= ratio <= 1.1:
    print(f"  ✓ Pitch MATCHES target!")
elif ratio < 0.9:
    print(f"  ⚠️  PRED IS FLAT: ~{1/ratio:.2f}x lower")
    semitones = 12 * np.log2(1/ratio)
    print(f"     Difference: {semitones:.1f} semitones")
else:
    print(f"  ⚠️  PRED IS SHARP: ~{ratio:.2f}x higher")
