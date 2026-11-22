# V8c Brief for Opus — "no touch event in preds; staying-in-touch plays at touch time"

**Date:** 2026-11-22 · Branch `orch/v8b-eval` · V8b training done (val_loss 6.55, best so far)

---

## User's domain description (ground truth)

- **Touch event** = a clear, short, percussive attack sound at the moment of touch.
- **Staying in touch** = a modulation that begins only ~**3 s after** the touch event
  (cf. `delayTouched = 3` in `SentireOSCInterface.sc`; in `01_canopy.scd` LFO depth
  `= timer*0.5` so it only becomes audible around timer≈3 s).
- Symptom in our preds: at touch time the model plays the *staying-in-touch* character
  immediately (timing mismatch), and the short percussive attack is missing/weak.

## Measured target behavior (event-locked, 230 onsets, 0.98 crossings, v8 data)

- Big energy jump at **+321..+471 ms** after the crossing: HP(>1.5kHz) +14 dB,
  LP(<1.5kHz) +20..23 dB (peak +471 ms), decaying by ~2.2 s.
- Envelope fluctuation (modulation depth) rises from ~+1.5 s onward (0.31 → 0.5+).
- So the audible "attack" swell is at +0.3..0.5 s, and audible modulation grows after ~1.5–3 s.

## Demo clips (`listening_demo/01_canopy_*.wav`, 3 events, 3 s gaps)

Event-window metric (RMS +250..550 ms / pre, peak in +0..600 ms, spectral flux at onset):

| model | ev1 ratio/pk/flux | ev3 ratio/pk/flux | val_loss |
|---|---|---|---|
| target | 1.74 / **0.554** / **734** | 2.66 / **0.710** / **595** | — |
| v6 | 1.34 / 0.163 / 243 | 1.54 / 0.156 / 234 | 10.36 |
| v7 (buggy window) | 1.64 / 0.057 / 42 | 1.50 / 0.054 / 41 | 7.28 |
| v7 control | 1.71 / 0.176 / 108 | 1.44 / 0.172 / 93 | 7.53 |
| **v8b (fixed)** | 1.27 / 0.170 / **561** | 1.36 / 0.146 / **401** | **6.55** |

V8b onset flux is 2.3x v6 / 13x v7 (weighting now works), but peak is still ~3x below target.

## Model architecture facts (`engine_ddsp/sensor_ddsp.py`, branch orch/v8b-eval)

- Control rate **100 Hz** (block 480): all heads (harmonic/noise/f0) output per-10 ms-frame params.
- `touch_onset` (1-frame blip at 0.98 crossing) → out_mlp + f0_head. `note` blip → f0 only.
- V8b `fast_head`: gated fast decays `exp(-t/0.03|0.1|0.3)` added to harmonic+noise params (zero-init, grows from V6 ckpt).
- LFO: `phase = cumsum(rate/100)` from frame 0, depth gated by `in_touch` → **modulation starts immediately at touch** (target: audible only after ~1.5–3 s).
- Onset-weighted loss: gamma profile peak +350 ms, k=5, shared spectral+RMS; data rebuilt at 0.98.

## Questions for Opus

1. **Why is the percussive attack still weak (peak 0.17 vs 0.55) even with fast_head + onset-weighted loss?** Candidates: (a) 100 Hz control rate too coarse for a sharp attack — the synth is harmonic osc + filtered noise with per-frame params; (b) fast_head starts at zero and 2000 steps @ 3e-4 may be too few to grow it; (c) the SC touch layer fires at a *random* pitch (`Dxrand`) the model can't predict, so under L1 spectral loss a wrong-pitch transient costs ~2x a missing one → model prefers no transient; (d) loss normalization dilutes the onset boost. Which dominates? How to verify cheaply?
2. **Timing mismatch:** should the model's LFO/modulation timer be `max(0, t-3s)` (or depth ramp `clip(t-3,0,..)/..`) to match "modulation starts ~3 s after touch"? What does the target actually do — is there a hard 3 s gate or a gradual ramp? (Measured: fluctuation rises after +1.5 s, no hard gate visible in the average.)
3. **Concrete next experiment:** exact changes (code/config/data), training budget, and the success metric (e.g., demo-clip peak ≥ 0.4, flux ≥ 600, plus val_loss guard ≤ 7.0). Prefer the smallest change that could produce a real percussive attack.
4. If the random touch-pitch is the blocker (1c), options: estimate per-onset target f0 50–300 ms after crossing and feed it as the note input at eval/train (sample-and-hold per touch), or use a noise-only transient pathway whose spectrum doesn't depend on pitch.

## Output format

```markdown
## 1. Why the attack is weak — ranked causes with evidence
## 2. Timing/mismatch verdict (3 s gate or ramp?) + fix if any
## 3. Recommended next experiment (concrete: files, changes, config, steps, metric)
## 4. If pitch is the blocker: preferred option
```

Read-only review of /home/marcello/Documents/interactive_ddsp. Do not modify files.
