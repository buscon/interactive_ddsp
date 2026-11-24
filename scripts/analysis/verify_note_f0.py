#!/usr/bin/env python3
"""
Verify note values and f0_base match target audio pitches.
"""
import numpy as np
import json
import librosa

DATA_ROOT = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6'
NOTES_JSON = '/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json'

# Load notes
with open(NOTES_JSON) as f:
    notes_data = json.load(f)

# Config f0_base
f0_base_config = {
    '01_canopy': 110.0,
    '02_forestPad': 210.0,
    '03_harmonicSwarm': 100.0,
    '05_moss_and_mist': 110.0
}

print("=" * 80)
print("NOTE / F0_BASE VERIFICATION")
print("=" * 80)

for sound, f0_base in f0_base_config.items():
    print(f"\n{sound} (f0_base config: {f0_base} Hz)")
    print("-" * 60)
    
    # Find curves for this sound
    sound_curves = {k: v for k, v in notes_data.items() if k.startswith(sound)}
    
    if not sound_curves:
        print(f"  ⚠️  No curves found in notes JSON!")
        continue
    
    # Analyze first curve
    curve_key = list(sound_curves.keys())[0]
    curve_data = sound_curves[curve_key]
    
    # curve_data is a list of dicts (one per frame)
    if isinstance(curve_data, list) and len(curve_data) > 0 and 'note_hz' in curve_data[0]:
        note_hz_all = [frame['note_hz'] for frame in curve_data if 'note_hz' in frame]
        
        # note_hz can be scalar or array
        note_hz_flat = []
        for n in note_hz_all:
            if isinstance(n, (list, np.ndarray)):
                note_hz_flat.extend(n)
            else:
                note_hz_flat.append(n)
        
        note_hz = np.array(note_hz_flat)
        nonzero = note_hz[note_hz > 0]
        
        if len(nonzero) > 0:
            print(f"  Curve: {curve_key}")
            print(f"  note_hz non-zero: {len(nonzero)} / {len(note_hz)} frames ({100*len(nonzero)/len(note_hz):.1f}%)")
            print(f"  note_hz median: {np.median(nonzero):.1f} Hz")
            print(f"  note_hz range: [{np.min(nonzero):.1f}, {np.max(nonzero):.1f}] Hz")
            print(f"  f0_base: {f0_base} Hz")
            print(f"  Ratio median/f0_base: {np.median(nonzero)/f0_base:.2f}x")
            
            # Check if f0_base matches
            if abs(np.median(nonzero) - f0_base) > 20:
                print(f"  ⚠️  f0_base ({f0_base}) differs from note median ({np.median(nonzero):.1f})")
        else:
            print(f"  ⚠️  All note_hz values are zero!")
    else:
        print(f"  ⚠️  Unexpected data format")

# Also check actual audio f0 with pYIN
print("\n" + "=" * 80)
print("AUDIO F0 VERIFICATION (pYIN on target audio)")
print("=" * 80)

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as AlignedChunkDataset

dataset = AlignedChunkDataset(
    root=DATA_ROOT,
    split='val',
    chunk_s=10.0,
    notes_json=NOTES_JSON,
    align_audio=False
)

for sound, f0_base in f0_base_config.items():
    # Find one curve for this sound
    for f in dataset.files:
        if f['sound'] == sound:
            item = dataset.full(f['sound'], f['curve'])
            audio_target = item['audio']
            note_stored = item['note']
            
            # Estimate f0 from audio
            f0_audio, _ = librosa.pyin(audio_target, fmin=80, fmax=500, sr=48000, frame_length=2048, hop_length=512)
            f0_audio_voiced = f0_audio[f0_audio > 0]
            
            note_stored_voiced = note_stored[note_stored > 0]
            
            print(f"\n{sound}_{f['curve']}:")
            print(f"  Audio f0 (pYIN): median={np.median(f0_audio_voiced):.1f} Hz ({len(f0_audio_voiced)} frames)")
            print(f"  Stored note:     median={np.median(note_stored_voiced):.1f} Hz ({len(note_stored_voiced)} frames)")
            print(f"  f0_base config:  {f0_base} Hz")
            
            # Check agreement
            if len(f0_audio_voiced) > 0 and len(note_stored_voiced) > 0:
                diff_cents = 1200 * np.log2(np.median(f0_audio_voiced) / np.median(note_stored_voiced))
                print(f"  Audio vs note: {diff_cents:+.1f} cents ({'✓' if abs(diff_cents) < 50 else '⚠️'})")
            
            break

print("\n" + "=" * 80)
