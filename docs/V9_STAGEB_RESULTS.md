# V9 Stage B Results — NO-GO Decision

**Date:** 2026-11-23  
**Branch:** `orch/v9attack` @ `0c93df0`  
**Config:** A4 (R=10, peak_ms=40, w_rms=10)

---

## Executive Summary

**Verdict: NO-GO for production**

Both seeds **failed** all success criteria on held-out validation curves. The excellent overfit results from A4 confirmation (dtpk ≈ 0 ms) **did not generalize** to unseen curves.

| Criterion | Target | Seed 0 | Seed 1 | Verdict |
|-----------|--------|--------|--------|---------|
| `med|dpk|` | ≤4 dB | **-6.16** ❌ | **-5.62** ❌ | FAIL |
| `med|dtpk|` | ≤20 ms | **53** ❌ | **134** ❌ | FAIL |
| `p_bump` | ≥5 dB | 4.33 ⚠️ | 3.69 ❌ | UNDERSHOOT |
| `strict_pass` | ≥3/5 | **0/5** ❌ | **0/5** ❌ | FAIL |

---

## Full Results Comparison

### Overfit Confirmation (A4) vs Stage B Generalization

| Metric | A4 s0 (overfit) | A4 s1 (overfit) | Stage B s0 (val) | Stage B s1 (val) |
|--------|-----------------|-----------------|------------------|------------------|
| dpk (dB) | -3.54 ✅ | -3.33 ✅ | -6.16 ❌ | -5.62 ❌ |
| dtpk (ms) | **0** ✅ | **-1** ✅ | 53 ❌ | 134 ❌ |
| p_bump (dB) | 3.1 | 4.3 | 4.33 | 3.69 |
| strict_pass | 1/5 | 1/5 | 0/5 | 0/5 |
| f0ratio | 0.25 | 0.25 | — | — |
| MS loss | — | — | 14.86 | **10.95** |

**Key finding:** Overfit timing (dtpk ≈ 0 ms) collapsed to 53-134 ms late on held-out curves.

---

## Per-Sound Breakdown

### Seed 0 (val)

| Sound | dpk (dB) | dtpk (ms) | p_bump (dB) | Pass |
|-------|----------|-----------|-------------|------|
| 01_canopy | -8.17 | 67 | 3.85 | ❌ |
| 02_forestPad | -5.90 | 20 | 4.00 | ❌ (timing edge) |
| 03_harmonicSwarm | -6.67 | 120 | 4.79 | ❌ |
| 04_modularPercussions | -1.79 | -92 | 4.42 | ❌ (early) |
| 05_moss_and_mist | -9.57 | 48 | 4.26 | ❌ |

### Seed 1 (val)

| Sound | dpk (dB) | dtpk (ms) | p_bump (dB) | Pass |
|-------|----------|-----------|-------------|------|
| 01_canopy | -6.43 | 139 | 2.60 | ❌ |
| 02_forestPad | -4.74 | 194 | 0.72 | ❌ |
| 03_harmonicSwarm | -8.42 | 153 | 3.69 | ❌ |
| 04_modularPercussions | -1.02 | 26 | **7.34** ✅ | ❌ (timing) |
| 05_moss_and_mist | -6.18 | 90 | 4.09 | ❌ |

**Pattern:** 04_modularPercussions is the only sound with acceptable dpk (-1.79/-1.02 dB) but timing is inconsistent (-92/+26 ms).

---

## Seed Divergence Analysis

Seeds 0 and 1 diverged significantly:

| Metric | Seed 0 | Seed 1 | Delta |
|--------|--------|--------|-------|
| Final val_loss (train) | 5.09 | **2.66** | -48% |
| MS loss (eval) | 14.86 | **10.95** | -26% |
| dtpk (median) | 53 ms | **134 ms** | +153% |
| RMS ratio | 0.73 | **0.86** | +18% |

**Interpretation:** Seed 1 achieved lower loss but **worse timing** (134 ms late). Lower loss ≠ better attack.

---

## What Went Wrong

### 1. Overfit ≠ Generalization
- A4 overfit: 8 chunks × 5 sounds = 40 files, same curves
- Stage B val: 3 held-out curves per sound (004, 011, 017)
- Model learned **curve-specific** timing, not **sound-specific**

### 2. Peak_ms=40 Insufficient
- Overfit confirmation showed dtpk ≈ 0 ms
- But val curves have different attack characteristics
- Single peak_ms cannot adapt to per-curve variation

### 3. w_rms=10 May Be Too High
- A4 used w_rms=10 (vs v9's 4.0)
- Higher RMS weight may prioritize envelope match over attack timing
- Seed 1's better RMS ratio (0.86) correlates with worse timing

### 4. No Per-Sound Adaptation
- attack_lag_s is fixed per sound
- But curves within a sound vary (e.g., 01_canopy_004 vs 01_canopy_017)
- Model cannot adapt to per-curve attack characteristics

---

## Recommended Next Steps

### Option A: Architectural Changes (HIGH PRIORITY)

1. **Per-curve attack lag learning**
   - Add curve embedding to predict attack_lag dynamically
   - Train curve→lag mapping on train curves, test on held-out

2. **Multi-scale onset weighting**
   - Current: single peak_ms=40
   - Proposed: [20, 40, 80] ms branches, attention-weighted

3. **Additive touch head redesign**
   - A6 (touch_amp_head=1) collapsed (f0ratio=3.98)
   - Try: separate amplitude path, no f0 modulation

### Option B: Data Augmentation (MEDIUM PRIORITY)

1. **Curve mixing**
   - Train on all curves (train + val), test on test split
   - Sacrifice val split for more training diversity

2. **Time stretching**
   - Augment attacks with slight time stretch (0.9-1.1×)
   - Force model to learn invariant timing

### Option C: Loss Redesign (MEDIUM PRIORITY)

1. **Attack-specific loss term**
   - Add explicit dtpk loss: `L_attack = |dtpk| / 20 ms`
   - Weight annealing: start low, increase over training

2. **Decouple RMS and attack**
   - Separate w_rms for attack region vs sustain
   - Attack region: [-20, +200] ms around onset

---

## Git Status

```
Branch: orch/v9attack
Commit: 0c93df0
Files added:
  - config_sensor_v9_stageB.yaml
  - docs/V9_STAGEB_LAUNCH.md
  - docs/V9_STAGEB_RESULTS.md (this file)
Outputs:
  - runs/v9_stageB/v9_stageB_A4_s{0,1}/ (checkpoints, logs)
  - runs/v9_stageB/eval_s{0,1}/ (eval reports, renders)
```

---

## Decision Record

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-11-23 | **NO-GO** | 0/5 strict pass both seeds, dtpk 53-134 ms (target ≤20) |
| 2026-11-23 | Pause Stage B | Architectural changes needed before further training |

**Next:** Implement Option A (per-curve attack lag learning) or Option C (attack-specific loss).
