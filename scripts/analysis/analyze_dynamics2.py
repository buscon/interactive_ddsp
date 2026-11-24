#!/usr/bin/env python3
"""
Analyze why pred files have less dynamic range than target.
"""
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

FOLDER = '/home/marcello/Documents/interactive_ddsp/listening_test_v9_stageB_corrected/02_forestPad'

curves = ['003', '007', '014']

print("=" * 80)
print("DYNAMICS ANALYSIS: pred vs target")
print("=" * 80)

for curve in curves:
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
    
    # Crest factor (peak / RMS) - measure of dynamic range
    pred_crest = np.abs(pred).max() / (np.sqrt(np.mean(pred**2)) + 1e-8)
    target_crest = np.abs(target).max() / (np.sqrt(np.mean(target**2)) + 1e-8)
    
    # Envelope dynamic ratio
    pred_dyn = np.max(pred_env) / (np.mean(pred_env) + 1e-8)
    target_dyn = np.max(target_env) / (np.mean(target_env) + 1e-8)
    
    # RMS range
    pred_rms_range = (np.min(pred_rms), np.max(pred_rms))
    target_rms_range = (np.min(target_rms), np.max(target_rms))
    
    print(f"\n{'='*60}")
    print(f"02_forestPad_{curve}")
    print(f"{'='*60}")
    print(f"Length: {len(pred)/48000:.1f}s")
    print()
    print("Level:")
    print(f"  Peak:  pred={np.abs(pred).max():.4f}, target={np.abs(target).max():.4f}, ratio={np.abs(pred).max()/np.abs(target).max():.2f}")
    print(f"  RMS:   pred={np.sqrt(np.mean(pred**2)):.4f}, target={np.sqrt(np.mean(target**2)):.4f}, ratio={np.sqrt(np.mean(pred**2))/np.sqrt(np.mean(target**2)):.2f}")
    print()
    print("Dynamic Range:")
    print(f"  Crest factor (peak/RMS): pred={pred_crest:.2f}, target={target_crest:.2f}")
    print(f"  Envelope dynamic (max/mean): pred={pred_dyn:.2f}, target={target_dyn:.2f}")
    print()
    print("RMS per 100ms window:")
    print(f"  pred:  min={pred_rms_range[0]:.4f}, max={pred_rms_range[1]:.4f}, range={pred_rms_range[1]/(pred_rms_range[0]+1e-8):.2f}x")
    print(f"  target: min={target_rms_range[0]:.4f}, max={target_rms_range[1]:.4f}, range={target_rms_range[1]/(target_rms_range[0]+1e-8):.2f}x")
    print()
    
    # Check if pred is compressed
    if pred_dyn < target_dyn * 0.8:
        print("  ⚠️  PRED IS COMPRESSED: less dynamic than target")
    elif pred_dyn > target_dyn * 1.2:
        print("  ⚠️  PRED HAS MORE DYNAMICS than target")
    else:
        print("  ✓ Dynamics are similar")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("If pred has lower crest factor and lower envelope dynamic ratio,")
print("the model is producing compressed output - likely because:")
print("  1. touch_amp_head is disabled (no direct touch→amplitude path)")
print("  2. Model learns average level, not touch-responsive dynamics")
print("=" * 80)
