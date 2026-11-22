"""Assert touch_time onsets (v8 data, 0.98) match bakeoff_notes_v7.json within ±1 frame."""
import numpy as np, json, glob, os
notes = json.load(open('/home/marcello/Documents/interactive_ddsp/bakeoff_notes_v7.json'))
root = '/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v8'
total, matched = 0, 0
for f in sorted(glob.glob(f'{root}/*.npz')):
    key = os.path.basename(f).replace('.npz','')
    tt = np.load(f)['touch_time']
    tt_on = [i for i in range(len(tt)) if tt[i] > 0 and (i==0 or tt[i-1]==0)]
    for o, v in enumerate(notes.get(key, [])):
        if v <= 0: continue
        total += 1
        if any(abs(o-t) <= 1 for t in tt_on): matched += 1
print(f'onset match: {matched}/{total} within ±1 frame')
assert matched == total, f'MISMATCH {matched}/{total}'
print('PASS')
