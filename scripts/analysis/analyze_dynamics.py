#!/usr/bin/env python3
"""
Analyze why pred files have less dynamic range than target.
"""
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert

FOLDER = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/02_forestPad'
OUTPUT = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/dynamics_analysis.png'

curves = ['003', '007', '014']

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for i, curve in enumerate(curves):
    pred, sr = sf.read(f'{FOLDER}/02_forestPad_{curve}_pred.wav')
    target, _ = sf.read(f'{FOLDER}/02_forestPad_{curve}_target.wav')
    
    # Envelope via Hilbert transform
    pred_env = np.abs(hilbert(pred))
    target_env = np.abs(hilbert(target))
    
    # RMS in 100ms windows
    hop = int(48000 * 0.1)
    pred_rms = []
    target_rms = []
    for j in range(0, len(pred)-hop, hop):
        pred_rms.append(np.sqrt(np.mean(pred[j:j+hop]**2)))
        target_rms.append(np.sqrt(np.mean(target[j:j+hop]**2)))
    
    # Plot waveform + envelope
    ax = axes[i, 0]
    t = np.arange(len(pred)) / 48000
    ax.plot(t, pred, alpha=0.5, label='pred', linewidth=0.5)
    ax.plot(t, target, alpha=0.5, label='target', linewidth=0.5)
    ax.plot(t, pred_env, 'r', label='pred envelope', linewidth=1.5)
    ax.plot(t, target_env, 'g', label='target envelope', linewidth=1.5)
    ax.set_xlim(0, min(10, len(pred)/48000))  # First 10 seconds
    ax.set_title(f'02_forestPad_{curve} - Waveform + Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot RMS comparison
    ax = axes[i, 1]
    t_rms = np.arange(len(pred_rms)) * 0.1
    ax.plot(t_rms, pred_rms, 'r.-', label='pred RMS', linewidth=1.5, markersize=8)
    ax.plot(t_rms, target_rms, 'g.-', label='target RMS', linewidth=1.5, markersize=8)
    ax.set_title(f'RMS per 100ms window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats
    pred_dyn = np.max(pred_env) / (np.mean(pred_env) + 1e-8)
    target_dyn = np.max(target_env) / (np.mean(target_env) + 1e-8)
    print(f"Curve {curve}:")
    print(f"  Peak: pred={np.abs(pred).max():.4f}, target={np.abs(target).max():.4f}")
    print(f"  RMS:  pred={np.sqrt(np.mean(pred**2)):.4f}, target={np.sqrt(np.mean(target**2)):.4f}")
    print(f"  Dynamic ratio (peak/mean env): pred={pred_dyn:.2f}, target={target_dyn:.2f}")
    print()

plt.tight_layout()
plt.savefig(OUTPUT, dpi=100)
print(f"Plot saved: {OUTPUT}")
