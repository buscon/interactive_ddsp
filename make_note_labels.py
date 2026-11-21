"""One-time: derive per-frame note_hz labels for the aligned dataset.

For each npz: segment touch runs (touch_time > 0), estimate median f0 of the audio
in each segment (autocorr, voiced frames only), snap to the sound's pentatonic scale
(baseFreq * 2^(degree/12)), hold until next touch. Output: bakeoff_notes.json (new file).
"""
import json, os, glob
import numpy as np

ROOT = '/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned'
SCALE = [0, 2, 4, 7, 9]  # pentatonic degrees (major pentatonic; matches Sentire scale usage)
BASE = {'01_canopy': 110.0, '02_forestPad': 210.0, '03_harmonicSwarm': 100.0,
        '04_modularPercussions': 200.0, '05_moss_and_mist': 110.0}
SR = 48000

def f0_median(x):
    """Median autocorr f0 over voiced 2048-windows; None if unvoiced."""
    W = 2048
    n = len(x) // W
    if n == 0:
        return None
    ests = []
    for i in range(n):
        w = x[i*W:(i+1)*W].astype(np.float64)
        w -= w.mean()
        if np.sqrt((w**2).mean()) < 1e-3:
            continue
        ac = np.correlate(w, w, 'full')[W-1:]
        ac /= (ac[0] + 1e-12)
        lo, hi = int(SR/500), int(SR/50)
        if hi >= len(ac):
            hi = len(ac) - 1
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            ests.append(SR / lag)
    if not ests:
        return None
    return float(np.median(ests))

def snap(f0, base):
    notes = [base * 2 ** (d / 12) * 2 ** o for d in SCALE for o in (-1, 0, 1)]
    return min(notes, key=lambda n: abs(1200 * np.log2(n / f0)))

def main():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, '*.npz'))):
        d = np.load(p)
        sound, curve = str(d['sound']), str(d['curve'])
        key = f'{sound}_proximity_curve_{curve}'
        audio = d['audio'].astype(np.float64) / 32768.0
        tt = d['touch_time'].astype(np.float64)
        base = BASE[sound]
        note = np.zeros(len(tt), dtype=np.float32)  # 0 = no touch
        i = 0
        T = len(tt)
        while i < T:
            if tt[i] > 0:
                j = i
                while j < T and tt[j] > 0:
                    j += 1
                seg = audio[i*480:j*480]
                f = f0_median(seg)
                note[i:j] = snap(f, base) if f else base
                i = j
            else:
                i += 1
        out[key] = note.tolist()
        print(key, 'segments:', int((np.diff(note) != 0).sum() + 1), 'note range:',
              float(note[note > 0].min()) if (note > 0).any() else 0,
              float(note[note > 0].max()) if (note > 0).any() else 0)
    with open('/home/marcello/Documents/interactive_ddsp/bakeoff_notes.json', 'w') as f:
        json.dump(out, f)
    print('wrote bakeoff_notes.json with', len(out), 'files')

if __name__ == '__main__':
    main()
