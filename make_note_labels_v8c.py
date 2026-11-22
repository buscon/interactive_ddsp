#!/usr/bin/env python3
"""
make_note_labels_v8c.py — held per-touch pitch labels snapped to each sound's SC scale.

For each file in bakeoff_data_aligned_v8 (touch threshold 0.98):
  1. onsets = frames where touch_time starts a run (must match bakeoff_notes_v7.json)
  2. attack lag D[sound] = median over onsets of (largest 20ms LP<1.5kHz rise within 0-1s) - crossing
  3. per-onset f0 estimated over attack+50..250ms (autocorr), snapped to minor pentatonic
     rooted at the sound's baseFreq (canopy 110, forestPad 210, swarm 100 (+1.5x set), perc 200, moss 110)
  4. note array: 0 before first touch; held f0 from (crossing + D) until next (crossing + D)

Output: bakeoff_notes_v8c.json {file_key: [T floats]}, attack_lag_report.json
"""
import json, glob, os
import numpy as np

SR = 48000
ROOT = '/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v8'
NOTES_V7 = '/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v7.json'
OUT_JSON = '/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v8c.json'
LAG_JSON = '/tmp/opencode/wt-v8c-labels/attack_lag_report.json'

SOUNDS = {'01_canopy': 110.0, '02_forestPad': 210.0, '03_harmonicSwarm': 100.0,
          '04_modularPercussions': 200.0, '05_moss_and_mist': 110.0}
PENTA = [0, 3, 5, 7, 10]  # minor pentatonic degrees


def scale_freqs(root):
    freqs = []
    for oct_ in range(-1, 4):
        for d in PENTA:
            freqs.append(root * 2 ** ((d + 12 * oct_) / 12))
    return sorted(freqs)


def lp_logenergy(audio, sr):
    """log energy below 1.5kHz, 20ms hop"""
    hop, win = 960, 1920
    n = (len(audio) - win) // hop
    out = np.empty(n)
    freqs = np.fft.rfftfreq(win, 1 / sr)
    lp = freqs < 1500
    for i in range(n):
        fr = np.abs(np.fft.rfft(audio[i * hop:i * hop + win] * np.hanning(win)))
        out[i] = 10 * np.log10((fr[lp] ** 2).sum() + 1e-12)
    return out, hop


def find_attack(logE, hop, sr, crossing_sample):
    """largest 20ms rise within 0-1s after crossing; return attack sample"""
    c = crossing_sample
    w0 = max(0, c)
    w1 = min(len(audio_g) if False else 10**9, c + sr)
    i0 = max(0, (w0) // hop)
    i1 = min(len(logE) - 2, (c + sr) // hop)
    if i1 - i0 < 3:
        return c + int(0.4 * sr)
    seg = logE[i0:i1]
    d = np.diff(seg)
    k = int(np.argmax(d))
    return c + (k + 1) * hop - c + c  # attack at i0+k+1 in samples = (i0+k+1)*hop


def estimate_f0(seg, sr):
    """autocorrelation f0 estimate on a segment"""
    seg = seg - seg.mean()
    if np.sqrt((seg ** 2).mean()) < 1e-4:
        return None
    ac = np.correlate(seg, seg, mode='full')[len(seg) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 600), int(sr / 60)  # 60-600 Hz
    if hi >= len(ac):
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.3:
        return None
    return sr / lag


def main():
    notes_v7 = json.load(open(NOTES_V7))
    files = sorted(glob.glob(f'{ROOT}/*.npz'))
    assert len(files) == 100
    out = {}
    lag_per_sound = {}
    pitch_stats = {}
    for f in files:
        key = os.path.basename(f).replace('.npz', '')
        d = np.load(f)
        audio = d['audio'].astype(np.float64) / 32768.0
        tt = d['touch_time']
        sound = key.split('_proximity')[0]
        root = SOUNDS[sound]
        sfs = scale_freqs(root)
        # onsets from touch_time runs
        onsets = [i for i in range(len(tt)) if tt[i] > 0 and (i == 0 or tt[i - 1] == 0)]
        note_v7 = [i for i, v in enumerate(notes_v7.get(key, [])) if v > 0]
        assert len(onsets) == len(note_v7) and all(abs(a - b) <= 1 for a, b in zip(onsets, note_v7)), key
        logE, hop = lp_logenergy(audio, SR)
        T = len(tt)
        arr = np.zeros(T, dtype=np.float32)
        lags, ests = [], []
        for o in onsets:
            c = o * 480
            # attack: largest 20ms rise in LP within 0-1s after crossing
            i0 = c // hop
            i1 = min(len(logE) - 2, (c + SR) // hop)
            seg = logE[i0:i1]
            k = int(np.argmax(np.diff(seg))) if len(seg) > 2 else int(0.4 * SR / hop)
            attack_sample = (i0 + k + 1) * hop
            lags.append((attack_sample - c) / SR)
            # f0 over attack+50..250ms
            a0 = attack_sample + int(0.05 * SR)
            a1 = attack_sample + int(0.25 * SR)
            f0est = estimate_f0(audio[a0:a1], SR) if a1 < len(audio) else None
            ests.append(f0est)
            # snap
            if f0est is None:
                f0 = sfs[len(sfs) // 2]
            else:
                f0 = min(sfs, key=lambda x: abs(12 * np.log2(x / f0est)))
            pitch_stats.setdefault(sound, []).append((f0est, f0))
            # hold from crossing + D until next crossing + D — D per sound computed after; use 0.4 here, refine after
            arr[o:] = f0  # provisional; refined in second pass
        lag_per_sound.setdefault(sound, []).extend(lags)
        out[key] = arr
    # per-sound lag
    D = {}
    for sound, lags in lag_per_sound.items():
        D[sound] = float(np.median(lags))
    print('attack lag D per sound (s):', {k: round(v, 3) for k, v in D.items()})
    json.dump(D, open(LAG_JSON, 'w'), indent=2)
    # second pass: hold from crossing + D[sound]
    for f in files:
        key = os.path.basename(f).replace('.npz', '')
        sound = key.split('_proximity')[0]
        d = np.load(f)
        tt = d['touch_time']
        T = len(tt)
        arr = np.zeros(T, dtype=np.float32)
        onsets = [i for i in range(len(tt)) if tt[i] > 0 and (i == 0 or tt[i - 1] == 0)]
        prev = out[key]
        for j, o in enumerate(onsets):
            f0 = prev[o]  # snapped pitch from first pass
            start = o + int(D[sound] * 100)
            end = (onsets[j + 1] + int(D[sound] * 100)) if j + 1 < len(onsets) else T
            arr[max(0, start):end] = f0
        out[key] = arr
    json.dump({k: v.tolist() for k, v in out.items()}, open(OUT_JSON, 'w'))
    print(f'wrote {OUT_JSON}: {len(out)} files')
    for sound, ps in pitch_stats.items():
        ok = sum(1 for e, s in ps if e is not None and abs(1200 * np.log2(s / e)) < 50)
        tot = sum(1 for e, _ in ps if e is not None)
        print(f'  {sound}: {ok}/{tot} estimated f0 within 50 cents of snapped note')


if __name__ == '__main__':
    main()
