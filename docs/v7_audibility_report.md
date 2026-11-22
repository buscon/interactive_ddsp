# V7 Target Audibility Report — CORRECTED (Full Dataset)

**Date:** 2026-11-22
**Status:** Corrected after full-dataset re-analysis (230 onsets, 100 files)

---

## Summary

**Touch events DO produce a clear, strong sound event.** The earlier "marginally audible"
finding was an artifact of a too-short measurement window (300ms) and a small sample
(23 onsets). Full-dataset analysis shows a strong attack developing over 300–500ms
after the 0.98 crossing.

---

## Quantitative Results (230 onsets, 100 files)

RMS ratio (post-onset window vs 500ms pre-onset baseline):

| Window | Median ratio | >1.5x | >2x | >3x |
|--------|-------------|-------|-----|-----|
| 50ms   | 0.90x       | 6%    | 2%  | 0%  |
| 150ms  | 0.93x       | 5%    | 2%  | 0%  |
| 300ms  | 0.95x       | 6%    | 3%  | 2%  |
| 400ms  | 1.14x       | 33%   | 28% | 25% |
| **500ms** | **2.58x** | **75%** | **63%** | **45%** |
| 700ms  | 3.46x       | 78%   | 71% | 56% |
| 1000ms | 3.54x       | 82%   | 70% | 57% |
| 2000ms | 3.33x       | 80%   | 68% | 54% |

### Interpretation

- **50–300ms after the 0.98 crossing: slight energy DIP (~0.9x)** — the curve is still
  ramping (touchRise=0.1s), and the previous sound state is fading.
- **400–500ms: attack energy arrives** — median 2.58x at 500ms, 63% of onsets >2x.
- **700ms+: sustained elevated energy (~3.4x)** — the touch layer (Env.perc attack 0.01s,
  decay 0.3–0.5s in SC touchSource) plus ongoing modulation.

The SC `touchSource` (e.g., 01_canopy.scd) fires a percussive layer on touch trigger:
`EnvGen.kr(Env.perc(0.01, 0.3), trig)` — fast 10ms attack but 300ms decay, so peak
energy lands ~300–500ms after trigger.

---

## Decision

**Onset weighting window: 500ms** (updated in `config_sensor_v6_onsetw.yaml`).

This captures the full attack energy. The 150ms window was too short and would have
weighted frames in the energy dip, not the attack.

---

## Onset alignment check

Sensor 0.98 crossings (bakeoff_notes_v7.json) are audio-aligned (sensor array in NPZ
is aligned to audio via `start_s` in prepare_aligned_data_v5.py). Spectral flux peaks
near onsets confirm alignment within ~1 STFT frame for clear attacks.

---

## Audio clips

`audibility_clips/` — 23 WAV files (1s before → 1s after onset) for qualitative check.
