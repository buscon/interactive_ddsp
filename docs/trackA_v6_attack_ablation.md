# V6 Attack Ablation Report

**Date:** 2026-11-22  
**Branch:** `orch/v6-ablation`  
**Status:** ✅ Complete

---

## Executive Summary

The v6 attack ablation (`w_attack=1.0, w_mod_spec=0.0`) trained successfully and was evaluated on 20 held-out samples. The attack loss was introduced to address the "soft attack" issue identified in v4c user feedback.

**Key Finding:** The attack loss did **not improve** attack transients compared to v6 baseline. Multiscale loss remained similar (10.27 vs 10.36), but RMS envelope correlation decreased (0.63 vs 0.74).

**Recommendation:** Do not ship attack loss alone. Wait for modspec and combined ablation results before deciding on loss configuration.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Config | `config_sensor_v6.yaml` |
| Loss weights | `w_multiscale=1.0, w_attack=1.0, w_mod_spec=0.0` |
| Steps | 2000 |
| Batch size | 8 |
| Dataset | `bakeoff_data_aligned_v5/` (100 files, exact sensor onsets) |
| Note labels | `bakeoff_notes_v5.json` (onset frames from sensor threshold) |

---

## Training Results

| Metric | Value |
|--------|-------|
| Initial val_loss | 24.21 |
| Best val_loss | 11.25 (step 250) |
| Final val_loss | 12.63 |
| Silence loss | 19.50 |
| Stability | ✅ No NaN, no collapse |

**Interpretation:**
- Training converged normally (val_loss decreased from 24.21 → 12.63)
- Silence loss (19.50) > val_loss (12.63): model learned signal vs noise
- Best checkpoint at step 250, slight overfitting afterward

---

## Evaluation Results

### Comparison Table (20 held-out samples, curves 017-020)

| Model | multiscale_loss ↓ | rms_envelope_corr ↑ | Notes |
|-------|-------------------|---------------------|-------|
| **v4c** (shipped) | 9.40 | 0.77 | Baseline |
| **v6 baseline** (losses OFF) | 10.36 | 0.74 | New dataset |
| **v6 + attack** (w_attack=1.0) | 10.27 | 0.63 | This ablation |

### Per-Sound Breakdown (v6 + attack)

| Sound | multiscale_loss | rms_envelope_corr |
|-------|-----------------|-------------------|
| 01_canopy | 9.69 | 0.80 |
| 02_forestPad | 9.62 | 0.78 |
| 03_harmonicSwarm | 9.74 | 0.73 |
| 04_modularPercussions | 11.38 | 0.58 |
| 05_moss_and_mist | 11.38 | 0.79 |

**Weakest:** `04_modularPercussions` (highest loss, lowest correlation)

### Ablation Analysis

| Ablation | multiscale_loss | rms_corr vs normal |
|----------|-----------------|-------------------|
| Zero (sensor=0, touch=0) | 16.91 | -0.02 |
| Rolled (F0 rolled) | 13.80 | -0.20 |

**Interpretation:**
- Zero ablation: Model produces near-silence without sensor input (expected)
- Rolled ablation: Higher loss confirms model depends on correct F0 input

---

## Audio Validity

| Metric | Min | Max | Expected |
|--------|-----|-----|----------|
| Peak level | 0.09 | 0.29 | > 0.001, < 0.99 ✅ |
| RMS | 0.06 | 0.12 | Reasonable range ✅ |
| Silent files | 0 | — | 0 (except abl_zero) ✅ |

All 20 samples produced valid audio.

---

## Comparison: V4c → V6 Baseline → V6 + Attack

### Multiscale Loss Trend

```
v4c:        9.40  ████████
v6 base:   10.36  █████████
v6+attack: 10.27  █████████
```

**Δ vs v6 baseline:** -0.09 (negligible improvement)

### RMS Envelope Correlation Trend

```
v4c:        0.77  ███████████████
v6 base:    0.74  ██████████████
v6+attack:  0.63  ███████████
```

**Δ vs v6 baseline:** -0.11 (degradation)

---

## Interpretation

### What Worked
- ✅ Training stability (no NaN, no collapse)
- ✅ Sensor conditioning verified (ablation shows dependency)
- ✅ Audio validity (all samples audible, no clipping)

### What Didn't Work
- ❌ Attack loss did **not** improve multiscale loss
- ❌ RMS envelope correlation **decreased** (0.74 → 0.63)
- ❌ No evidence of sharper attack transients

### Why?
Possible explanations:
1. **Exact onsets already help:** V6 baseline has sensor-based onset times; attack loss may be redundant
2. **Loss weight too high:** `w_attack=1.0` may overpower multiscale loss
3. **Wrong target:** Attack loss targets spectrogram onset, but user issue may be perceptual (e.g., harmonic content at onset)

---

## Recommendations

### Immediate
1. **Do not ship attack loss alone** — no improvement, some degradation
2. **Wait for modspec + both ablations** — combined losses may behave differently
3. **Listen qualitatively** — metrics may not capture attack sharpness perception

### Future Experiments
If modspec/both ablations also fail:
1. **Reduce attack weight:** Try `w_attack=0.5`
2. **Modify attack loss:** Target harmonic content at onset, not just energy
3. **Architectural change:** Separate transient pathway (e.g., noise burst at onset)

---

## Files Produced

| File | Location |
|------|----------|
| Training config | `config_sensor_v6.yaml` (modified: `w_attack=1.0`) |
| Checkpoint | `/tmp/opencode/wt-attack/runs/bakeoff_ddsp_poc/stageB_v6_attack/state_final.pth` |
| Metrics | `/tmp/opencode/wt-attack/runs/bakeoff_ddsp_poc/stageB_v6_attack/eval_output/metrics.json` |
| Audio renders | `/tmp/opencode/wt-attack/runs/bakeoff_ddsp_poc/stageB_v6_attack/eval_output/*.wav` |
| Training log | `/tmp/opencode/wt-attack/runs/bakeoff_ddsp_poc/stageB_v6_attack/train_log.csv` |

---

## Git Status

| Branch | Commit | Status |
|--------|--------|--------|
| `orch/v6-attack` | `e53c54b` | ✅ Complete |
| `orch/v6-modspec` | — | ⏳ Training |
| `orch/v6-both` | — | ⏳ Training |
| `orch/v6-ablation` (integration) | — | Pending |

---

## Next Steps

1. ⏳ Await modspec ablation completion
2. ⏳ Await both ablation completion
3. 📊 Synthesize full comparison table (v4c, v6, v6+attack, v6+modspec, v6+both)
4. 👂 Qualitative listening test
5. 🔀 Merge to integration branch with full report

---

**Model:** Qwen3.5-397B (orchestration)  
**Verification:** Pending (two-step: Qwen3.5 + Mistral-Large agreement)  
**Author:** AI Agent (orchestration)
