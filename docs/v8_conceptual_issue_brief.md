# V8 Conceptual Issue Brief — for Claude Opus 5.5 review

**Date:** 2026-11-22
**Question:** Why do V6/control/V7 predictions contain no audible touch events (only sustained "staying-in-touch" sound), even though touch onsets are precisely labeled (sensor ≥ 0.98 crossings, 230 onsets) and target audio clearly has events (median 2.15x RMS jump at onsets)?

---

## Leading hypothesis: touch_time threshold mismatch (0.3 vs 0.98)

**Training data** (`prepare_aligned_data_v5.py:33`): `THRESHOLD_TOUCH = 0.3`
→ `touch_time` in every NPZ starts counting as soon as `sensor > 0.3`.
Comment says "was 0.98 in v4" — the threshold was lowered in v5.

**Audio-generating process** (Sentire SC): the touch event (`touchSource` perc layer, `\touch.ar` trigger) fires when the touch detection signal crosses `thresholdTouch = 0.98` (dataset_info.txt). So the *audible event* is at the 0.98 crossing.

**Consequence:** in the example file `01_canopy_proximity_curve_001`, sensor holds at 0.435 for many frames before ramping to 1.0 (touchRise=0.1s). Since 0.435 > 0.3, `touch_time` was **already > 0 during the hold** — the model's `touch_onset` (derived from `touch[...,0] > 0` transition, sensor_ddsp.py:192-194) fires when sensor first exceeds 0.3, which can be **seconds before** the 0.98 touch event. The model therefore sees "in touch" during the whole approach ramp → produces sustained modulation, no discrete event at the audible onset.

**Model input path:** `touch_onset` (single-frame blip) feeds `out_mlp` (→ harmonic/noise amplitude heads) and `f0_head` (sensor_ddsp.py:221,227). So there IS an onset pathway into amplitude — but it fires at the wrong time (0.3 crossing, not 0.98).

**Eval inconsistency:** main eval path uses `item["touch"]` from NPZ (0.3-based, consistent with training), but the rolled-ablation path recomputes touch at threshold **0.98** (`eval_stageB_v5.py:509`, `render_sensor_ddsp.py:395`) — train/eval mismatch in the ablation.

**Onset labels:** `bakeoff_notes_v7.json` = 0.98 crossings (230 onsets) — these ARE aligned with the audio events (flux peak lag median −161ms... actually near ramp). But the *loss onset mask* and *model touch_onset* derive from touch_time (0.3) — misaligned with both the notes and the audio.

---

## Measured facts

| Fact | Value |
|------|-------|
| Target onset RMS ratio (500ms post/pre, 30 held-out onsets) | median **2.15x**, 53% >2x |
| V6 pred | 1.42x (27% >2x), overall rms_ratio 0.69 |
| control pred (weighting OFF) | 1.60x, rms_ratio 0.57 |
| V7 pred (spectral onset weighting k=5, 500ms) | 1.18x, rms_ratio **0.31** (quieter!) |
| V7 metrics | multiscale 9.49 (best), rms_corr 0.817 (best), onset_f0_err 790c (best) |
| Flux peak vs 0.98 crossing | median lag −161ms (audio event during ramp, before 0.98) |
| Touch-time onset (0.3) vs 0.98 crossing | can be seconds apart (hold at 0.435 > 0.3) |

---

## Proposed fix

Rebuild aligned dataset with `THRESHOLD_TOUCH = 0.98` (as in v4), so that:
- `touch_time` starts at the 0.98 crossing = SC touch trigger = audio event = `bakeoff_notes_v7.json` onset.
- Model `touch_onset` (touch_time 0→>0) aligns with audio event.
- Loss onset mask aligns with audio event.
- Fix eval rolled path to use the same threshold as the NPZ (or just use NPZ touch).

Then retrain V8 (spectral + RMS onset weighting, k=5, 500ms) on the 0.98-aligned data.

---

## Questions for Opus

1. Does the 0.3-vs-0.98 threshold mismatch fully explain "no audible events, only sustained sound"? Any other conceptual error we're missing (e.g., note blip only affects f0, not amplitude; GRU smoothing of the 1-frame onset blip)?
2. Is rebuilding the dataset at 0.98 the right fix, or should we instead pass an explicit `touch_onset` input derived from the 0.98 crossings while keeping touch_time at 0.3?
3. Should the onset weighting window be measured from the 0.98 crossing (yes, per notes_v7) — and does the 500ms window still make sense given the energy peaks at 400-700ms after the crossing?
4. Any architecture tweak to make the transient sharper (e.g., feed touch_onset into the noise head only, or add a fast-decay onset feature exp(-t/0.05) alongside the slow exp(-t/0.5))?
5. Recommended next experiment, concretely (config + code changes + success metric)?

## Constraints
- Focus ONLY on touch events (onset perceptibility). Sustained-modulation ramp is a separate known issue, parked.
- Low-effort first.
