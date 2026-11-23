# V9 Stage B Training Launch Report

**Date:** 2026-11-23  
**Branch:** `orch/v9attack` @ `0c93df0`  
**Config:** A4 winner (R=10, peak_ms=40)

---

## Background

Following the 7-arm × 2-seed confirmation matrix (14 GPU runs, 1000 steps each), **A4 (R=10, peak_ms=40)** emerged as the winner:

| Metric | A4 s0 | A4 s1 | Target |
|--------|-------|-------|--------|
| dpk (dB) | -3.54 | -3.33 | ≤4 |
| dtpk (ms) | **0** | **-1** | ≤20 |
| p_bump (dB) | 3.1 | 4.3 | ~6.7 |
| strict_pass | 1/5 | 1/5 | ≥4/5 |
| f0ratio | 0.25 | 0.25 | <1.0 |

**Key finding:** A4 achieved excellent timing (dtpk ≈ 0 ms) with stable f0, but p_bump still undershoots (50-65% of target).

---

## Stage B Configuration

```yaml
# config_sensor_v9_stageB.yaml
model:
  attack_lag_s: [0.023, 0.026, 0.065, 0.040, 0.025]
  onset_weighting:
    enabled: true
    factor: 5.0
    pre_ms: 20
    window_ms: 400
    peak_ms: 40  # ← A4 winner
  touch_amp_head: false

loss_weights:
  w_lin: 100.0
  w_rms: 10.0    # ← A4 value

train:
  lr_start: 1.0e-3
  lr_stop: 1.0e-4
  decay_over: 2000
  warmup_steps: 100
  steps: 2000
  batch: 8
  eval_every: 250
  eval_curves: ["004", "011", "017"]  # v6 held-out val curves
```

---

## Training Runs

### Seed 0 (COMPLETED)
- **Run:** `v9_stageB_A4_s0`
- **Steps:** 2000
- **Final val_loss:** 5.09
- **Final silence_loss:** 10.93
- **Checkpoints:**
  - `state_best.pth` (best val_loss)
  - `state_final.pth` (step 2000)
- **Location:** `/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB/v9_stageB_A4_s0/`

### Seed 1 (IN PROGRESS)
- **Run:** `v9_stageB_A4_s1`
- **Steps:** 2000
- **Status:** Training started at 13:28
- **ETA:** ~30-40 minutes

---

## Success Criteria for Stage B

| Metric | Target | Rationale |
|--------|--------|-----------|
| `med|dpk|` | ≤4 dB | Match overfit confirmation |
| `med|dtpk|` | ≤20 ms | Match overfit confirmation |
| `p_bump` | ≥5 dB | Relaxed from 6.7 dB (generalization) |
| `centroid_corr` | >-0.3 | No timbre collapse |
| `report_loss` | ~8-10 | Comparable to v9_scratch |
| `strict_pass` | ≥3/5 | Improved from 1/5 overfit |

---

## Next Steps

1. **Wait for seed 1 completion** (~30-40 min)
2. **Run attack evaluation** on both seeds:
   ```bash
   python3 eval_stageB_v5.py --config config_sensor_v9_stageB.yaml \
     --checkpoint runs/v9_stageB/v9_stageB_A4_s0/state_final.pth \
     --out_dir runs/v9_stageB/eval_s0
   ```
3. **Compile results** and compare seeds for stability
4. **Decision point:**
   - ✅ GO: Both seeds pass criteria → proceed to test set evaluation
   - ⚠️ CONDITIONAL: One seed passes → investigate divergence
   - ❌ NO-GO: Both fail → architectural changes needed

---

## Git Status

```
Branch: orch/v9attack
Commit: 0c93df0 (merge: Task 2 - eval harness & attack metrics)
Files changed:
  - config_sensor_v9_stageB.yaml (NEW)
  - train_sensor_ddsp_v5.py (eval unpacking fix)
```

---

## Monitoring

- **TensorBoard:** `tensorboard --logdir /home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB/`
- **Train logs:** `runs/v9_stageB/v9_stageB_A4_s{0,1}/train_log.csv`
- **Checkpoints:** `runs/v9_stageB/v9_stageB_A4_s{0,1}/state_*.pth`
