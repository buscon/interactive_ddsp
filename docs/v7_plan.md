# V7 Plan — Onset-Weighted Multiscale Loss (Updated)

**Date:** 2026-11-22
**Branch:** `orch/v7-onsetw`
**Base:** `orch/v6-ablation` @ `69f68a4`

---

## 1. Critical Correction

The previous alignment audit compared sensor onsets to audio spectral flux — **wrong approach**.

This is a **continuous sound dataset with touch modulation**, not percussive sounds. The "attack" to improve is **touch event perceptibility**, not audio onset alignment.

**Correct onset source:** proximity curve crossing threshold 0.98 from below = **touch events** (this IS the OSC touch trigger).

Note: OSC touch events are NOT separately logged to disk — they are defined by the proximity signal itself crossing 0.98.

---

## 2. What We Know

### Touch Onsets (OSC Events @ 0.98 Threshold)

| Metric | Value |
|--------|-------|
| Files | 100 |
| Total onsets | 230 |
| Onsets per file | 1–6 (varies by curve) |
| Source | sensor array crossing 0.98 from below |
| Reliability | High — this IS the OSC touch trigger (not separately logged) |

### Root Cause of V6 Attack Problem

Touch events are "almost not perceptible" despite correct timing:

1. Onset frames ≈ 1% of 60s clips → uniform loss gives them almost no gradient
2. Separate `w_attack` loss creates gradient conflict with multiscale loss
3. Model optimizes sustained spectral accuracy, not transient sharpness

**Not a data alignment problem** — `touch_time` onsets are reliable.

---

## 3. V7 Solution

**Onset-weighted multiscale loss:** weight onset frames higher *within* the existing multiscale loss (no separate attack loss → no gradient conflict).

### Files (already on branch)

| File | Status |
|------|--------|
| `engine_ddsp/sensor_ddsp.py` | Done — added `_compute_onset_weights()` |
| `config_sensor_v6_onsetw.yaml` | Done — k=5.0, window=150ms, dilate=1 |
| `test_onset_weighting.py` | Done — 4 unit tests passing |
| `bakeoff_notes_v7.json` | Done — 230 onsets @ 0.98 threshold |

### Config

```yaml
notes_json: bakeoff_notes_v7.json   # sensor onsets @ 0.98 (OSC touch events)

loss:
  w_attack: 0.0
  w_mod_spec: 0.0
  multiscale:
    onset_weighting:
      enabled: true
      factor: 5.0
      window_ms: 150
      dilate_frames: 1
      touch_rate_hz: 100
```

---

## 4. Execution Steps

| # | Step | Command / Detail | ETA |
|---|------|------------------|-----|
| 1 | Update config to v7 notes | point `notes_json` to `bakeoff_notes_v7.json` | done |
| 2 | Smoke test | `train_sensor_ddsp_v5.py config_sensor_v6_onsetw.yaml --steps 100` | 10 min |
| 3 | Full training | fine-tune from V6 baseline, 1000 steps, lr 0.3×, batch 8, val every 250 | 2–3 h |
| 4 | Evaluation | `eval_stageB_v5.py` on curves 017–020 (20 files) | 30 min |
| 5 | Listening test | V6 baseline vs V7, focus on touch event perceptibility | 15 min |

---

## 5. Success Criteria

| Metric | V6 baseline | V7 target |
|--------|-------------|-----------|
| multiscale_loss | 10.36 | ≤ 10.5 |
| rms_envelope_corr | 0.74 | ≥ 0.72 |
| silence_loss | 19.50 | ≤ 19.5 |
| **Touch events (qualitative)** | almost inaudible | perceptible |

---

## 6. Open Questions for Review

1. Is onset-weighted multiscale (k=5.0, 150 ms) the right fix given reliable touch-time onsets?
2. Fine-tune from V6 baseline (1000 steps, 0.3× lr) — safe, or full retrain better?
3. Any extra metric to add beyond `onset_peak_ratio` (0.8–1.2 target)?
