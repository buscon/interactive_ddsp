"""Event-locked HP(>300Hz) log-RMS profile aligned at each 0.98 crossing (notes_v7)."""
import numpy as np, json, glob, os
notes = json.load(open('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v7.json'))
sr = 48000
profiles = []
for f in sorted(glob.glob('/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v8/*.npz')):
    key = os.path.basename(f).replace('.npz','')
    audio = np.load(f)['audio'].astype(np.float64)/32768.0
    for o, v in enumerate(notes.get(key, [])):
        if v <= 0: continue
        s = o*480
        a, b = s-int(0.5*sr), s+int(1.5*sr)
        if a < 0 or b > len(audio): continue
        seg = audio[a:b]
        hop, win = 480, 1024
        n = (len(seg)-win)//hop
        fr = np.stack([np.abs(np.fft.rfft(seg[i*hop:i*hop+win]*np.hanning(win))) for i in range(n)])
        freqs = np.fft.rfftfreq(win, 1/sr)
        profiles.append(10*np.log10((fr[:, freqs>300]**2).sum(axis=1) + 1e-12))
P = np.stack([p - p.mean() for p in profiles])
mp = P.mean(axis=0)
t = (np.arange(P.shape[1])*hop + win/2 - 0.5*sr)/sr*1000
base = mp[(t>=-300)&(t<=-50)].mean()
d = np.diff(mp); pk = int(np.argmax(d))
print(f'n={len(profiles)}  max rise at +{t[pk+1]:.0f} ms  baseline={base:.1f} dB')
for thr in [3, 10]:
    idx = np.where((t>0)&(mp-base>thr))[0]
    print(f'+{thr}dB at {t[idx[0]]:.0f} ms' if len(idx) else f'+{thr}dB never')
