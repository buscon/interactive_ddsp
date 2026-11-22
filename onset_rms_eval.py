"""Onset-window RMS ratio eval: V6 vs V7-control vs V7-main on held-out curves 017-020.

For each touch onset (from bakeoff_notes_v7.json), computes RMS ratio
post/pre (500 ms post window vs 500 ms pre baseline) for BOTH the model
prediction and the target audio. Reports median ratio, %>2x, and
pred/target median ratio.
"""
import numpy as np, json, glob, soundfile as sf, os

SR = 48000
FRAME = 480          # samples per control frame (100 Hz)
WINDOW_S = 0.5       # 500 ms post / 500 ms pre

notes = json.load(open('bakeoff_notes_v7.json'))

MODELS = [
    ('v6',      'runs/bakeoff_ddsp_poc/stageB_v6/eval_v7grid'),
    ('control', 'runs/bakeoff_ddsp_poc/stageB_v7_control/eval_v7grid'),
    ('v7',      'runs/bakeoff_ddsp_poc/stageB_v7/eval_v7grid'),
]

def sid_to_note_key(sid):
    # '01_canopy_017' -> '01_canopy_proximity_curve_017'
    sound, curve = sid.rsplit('_', 1)
    return f'{sound}_proximity_curve_{curve}'

w = int(WINDOW_S * SR)

results = {}
for model_name, eval_dir in MODELS:
    ratios_pred, ratios_tgt = [], []
    n_files, n_onsets = 0, 0
    for wav in sorted(glob.glob(f'{eval_dir}/*_pred.wav')):
        sid = os.path.basename(wav).replace('_pred.wav', '')
        key = sid_to_note_key(sid)
        npz = f'bakeoff_data_aligned_v5/{key}.npz'
        if not os.path.exists(npz):
            print(f'  [{model_name}] skip {sid}: no npz {npz}')
            continue
        if key not in notes:
            print(f'  [{model_name}] skip {sid}: no notes for {key}')
            continue
        pred, _ = sf.read(wav)
        pred = np.asarray(pred, dtype=np.float64)
        if pred.ndim > 1:
            pred = pred.mean(axis=1)
        tgt = np.load(npz)['audio'].astype(np.float64) / 32768.0
        onset_frames = [i for i, v in enumerate(notes[key]) if v > 0]
        n_files += 1
        for o in onset_frames:
            s = o * FRAME
            for audio, ratios in ((pred, ratios_pred), (tgt, ratios_tgt)):
                if s + w > len(audio) or s - w < 0:
                    continue
                pre = np.sqrt(np.mean(audio[s - w:s] ** 2))
                post = np.sqrt(np.mean(audio[s:s + w] ** 2))
                ratios.append(post / (pre + 1e-9))
            n_onsets += 1
    rp, rt = np.array(ratios_pred), np.array(ratios_tgt)
    results[model_name] = dict(rp=rp, rt=rt, n_files=n_files, n_onsets=n_onsets)
    print(f'{model_name}: files={n_files} onsets={n_onsets} '
          f'pred median {np.median(rp):.2f}x ({(rp > 2).mean() * 100:.0f}% >2x), '
          f'target median {np.median(rt):.2f}x, '
          f'pred/tgt {np.median(rp) / np.median(rt):.2f}')

# Cross-model summary + per-file detail
print('\n=== Summary ===')
rt = results['v6']['rt']
print(f'target (shared): median {np.median(rt):.2f}x ({(rt > 2).mean() * 100:.0f}% >2x), n={len(rt)}')
for name, r in results.items():
    rp = r['rp']
    print(f'{name:8s} pred median {np.median(rp):.3f}x  mean {rp.mean():.3f}x  '
          f'%>2x {(rp > 2).mean() * 100:.1f}%  %>.5x {(rp > 0.5).mean() * 100:.1f}%  '
          f'pred/tgt {np.median(rp) / np.median(rt):.3f}')

# Save raw ratios for the report
out = {name: {'pred_ratios': r['rp'].tolist(), 'target_ratios': r['rt'].tolist()}
       for name, r in results.items()}
json.dump(out, open('onset_rms_results.json', 'w'), indent=1)
print('\nSaved onset_rms_results.json')
