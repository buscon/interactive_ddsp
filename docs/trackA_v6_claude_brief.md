# V6 Ablation Brief for Claude Code Review

**Date:** 2026-11-22  
**Task:** Review attack transient degradation in v6 baseline and recommend fix

---

## Executive Summary

**Problem:** V6 baseline improved continuous touch modulation and noise floor, but **made attack transients LESS perceptible** (user reports: "touch events got worst, they are almost not perceptible").

**Failed fix:** Adding `w_attack=1.0` loss degraded performance further (RMS correlation 0.74→0.63, no multiscale improvement).

**Question:** What is the root cause, and what is the best fix that preserves V6's gains (touch modulation, noise floor)?

---

## Context

### Model Architecture
- **DDSP with sensor conditioning** (proximity sensor + touch features)
- **Touch features:** 3-dim (log1p(t), exp(-t/0.5), clipped t)
- **Timer input:** 3-dim (for timer-driven LFO sounds)
- **Hybrid LFO:** rate + depth modulation, touch-gated per sound

### Loss Functions
1. **multiscale_loss** — STFT comparison, always ON (w=1.0)
2. **w_attack** — Spectrogram onset energy matching in 200ms window
3. **w_mod_spec** — Modulation spectral matching

### Dataset Change (V4c → V6)
| V4c | V6 |
|-----|-----|
| Scale-snapped onset times (approximate) | Exact sensor threshold crossings (threshold=0.3) |
| 10s chunks | Whole 30s files |

---

## Results

### Quantitative (20 held-out samples, curves 017-020)

| Model | w_attack | w_mod_spec | multiscale_loss ↓ | rms_envelope_corr ↑ | silence_loss |
|-------|----------|------------|-------------------|---------------------|--------------|
| **v4c** (shipped) | — | — | 9.40 | 0.77 | 23.22 |
| **v6 baseline** | 0 | 0 | 10.36 | 0.74 | 19.50 |
| v6 + attack | 1 | 0 | 10.27 | 0.63 | 19.50 |
| v6 + modspec | 0 | 1 | 11.53 | 0.63 | 19.50 |
| v6 + both | 1 | 1 | 12.00 | 0.65 | 19.50 |

### Qualitative (User Listening Test)

| Aspect | V4c | V6 Baseline | Verdict |
|--------|-----|-------------|---------|
| **Attack sharpness** | Soft but perceptible | **Almost inaudible** | ❌ Worse |
| **Continuous touch modulation** | None | **Active modulation** | ✅ Much better |
| **Noise floor** | Acceptable | **Cleaner** | ✅ Better |

---

## The Paradox

**Exact sensor onsets (V6) made touch events LESS perceptible, not more.**

Intuitively, precise timing should improve attacks. Instead, attacks became nearly inaudible.

---

## What We Tried

| Experiment | Result |
|------------|--------|
| `w_attack=1.0` | No multiscale improvement, RMS correlation degraded (0.74→0.63) |
| `w_mod_spec=1.0` | Worse on all metrics |
| `w_attack=1.0, w_mod_spec=1.0` | Training instability (collapse at step 1750), worst metrics |

---

## Qwen3.5-397B Analysis (For Reference)

### Root Cause Hypothesis

**The attack loss optimizes the wrong signal at the wrong time:**

1. **Temporal mismatch** — `touch_onset` derived from touch features at 100 Hz control rate, but STFT operates at ~21 Hz (9600 sample window / 2400 hop). Loss weights wrong audio frames.

2. **Energy ≠ Perceptual Sharpness** — Attack loss matches total spectrogram energy in 200ms window. But perceptual sharpness comes from:
   - Fast amplitude rise time (20-50ms, not 200ms)
   - High-frequency content at onset (brightness)
   - Transient-to-sustain ratio
   
   Matching energy over 200ms can *smear* transients.

3. **Gradient conflict** — Multiscale loss (spectral accuracy with long windows) and attack loss (temporal precision with short windows) compete. Model compromises, degrading both.

### Why w_attack=1.0 Failed

| Symptom | Explanation |
|---------|-------------|
| RMS correlation dropped | Attack loss forces energy into onset windows, disrupting envelope shaping |
| multiscale_loss unchanged | Model can't satisfy both losses; multiscale dominates |
| Touch events inaudible | Derived `touch_onset` doesn't match perceptual onsets; loss reinforces wrong frames |
| Combined losses collapsed | Gradient conflict became severe; no stable optimum |

### Recommended Fix (Priority #1)

**Onset-Weighted Multiscale Loss** — Instead of separate attack loss, weight multiscale loss higher at onset frames:

```python
# Derive touch_onset from touch features
touch_onset = ((touch[..., :1] > 0).float().diff().clamp(0))

# Downsample to STFT frame rate (~21 Hz)
onset_weight = 1.0 + 5.0 * downsample(touch_onset)

# Apply to multiscale loss
weighted_loss = multiscale_loss * onset_weight
```

**Rationale:**
- Single loss function (no gradient conflict)
- Onset frames get more attention during spectral matching
- Preserves V6 gains (touch modulation, noise floor)

**Config:**
```yaml
model:
  w_attack: 0.0
  w_mod_spec: 0.0

loss:
  onset_weight_multiscale: true
  onset_weight_factor: 5.0
  onset_window_frames: 5  # 50ms after onset
```

---

## Your Task

### Please Analyze

1. **Do you agree with Qwen's root cause hypothesis?** (temporal mismatch + energy≠sharpness + gradient conflict)

2. **Is onset-weighted multiscale the right fix?** Or is there a better approach?

3. **Alternative suggestions:**
   - Perceptual attack loss (brightness + rise time)?
   - Reduced attack weight (w=0.25) + curriculum learning?
   - Data augmentation (boost transients in training targets)?
   - Architecture change (separate transient pathway)?

### Please Recommend

**Concrete experiment to run next**, with:
- Config changes (yaml)
- Code modifications (if any)
- Success criteria (quantitative + qualitative)
- Expected timeline (training steps, eval protocol)

---

## Constraints

- **Preserve V6 gains:** Continuous touch modulation, improved noise floor
- **Low effort first:** Prefer minimal code changes before architecture rewrites
- **Testable:** Must have clear success criteria

---

## Files Available (If You Have Access)

- Full ablation report: `docs/trackA_v6_ablation_final.md`
- Metrics: `runs/bakeoff_ddsp_poc/stageB_v6*/eval_output/metrics.json`
- Audio comparison: `runs/bakeoff_ddsp_poc/stageB_v4c_eval/` vs `runs/bakeoff_ddsp_poc/stageB_v6/eval_output/`
- Model code: `engine_ddsp/sensor_ddsp.py`, `engine_ddsp/sensor_data.py`

---

**Thank you for your analysis!** Please provide your independent assessment.
