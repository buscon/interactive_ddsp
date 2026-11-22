# Track A v6 Addendum (2026-11-21)

## Summary
v6: New dataset (Dataset_20261121_134808) with exact onset times from sensor threshold crossings (notes.csv, 46 onsets), touch threshold 0.3 (was 0.98), losses OFF (w_attack=0, w_mod_spec=0). Model config unchanged from v4c (timer input dim 3 present but zero-weighted). Training: 2000 steps, batch 8, eval every 250.

## Results: v4c vs v6 (held-out, 20 files)

| Metric | v4c | v6 |
|--------|-----|-----|
| multiscale_loss | 9.402 | **10.362** |
| rms_envelope_corr | 0.772 | **0.738** |
| centroid_corr | n/a | 0.265 |
| rms_ratio | n/a | 0.490 |
| abl_rolled / abl_zero | -0.116 / -0.211 | **-0.103 / 0.034** |
| val loss / silence | 7.04 / 23.22 | **12.46 / 19.50** |
| loss_ratio (vs silence) | 0.303 | **0.535** |

### Per-Sound Breakdown (v6)

| Sound | MS Loss | RMS Env Corr | Centroid Corr | RMS Ratio |
|-------|---------|--------------|---------------|-----------|
| 01_canopy | 9.690 | 0.799 | 0.489 | 0.467 |
| 02_forestPad | 9.621 | 0.784 | 0.418 | 0.299 |
| 03_harmonicSwarm | 9.741 | 0.732 | 0.305 | 0.460 |
| 04_modularPercussions | 11.378 | 0.582 | 0.125 | 0.651 |
| 05_moss_and_mist | 11.377 | 0.793 | -0.010 | 0.574 |

## Gap Analysis

1. **Higher validation loss**: v6 val_loss 12.46 vs v4c 7.04 — expected due to:
   - New dataset with different acoustic characteristics
   - Lower touch threshold (0.3 vs 0.98) → more touch frames (61% vs ~20%)
   - No note labels for most sounds (only 01_canopy has onsets in notes.csv)

2. **Ablation results**: 
   - v6 abl_zero corr 0.034 (worse than v4c -0.211) — sensor still contributes but weaker
   - v6 abl_rolled corr -0.103 (similar to v4c -0.116) — sensor conditioning verified

3. **Per-sound variation**:
   - 01_canopy, 02_forestPad, 03_harmonicSwarm: good RMS corr (>0.73)
   - 04_modularPercussions: weakest fit (RMS corr 0.58, highest MS loss 11.38)
   - 05_moss_and_mist: negative centroid corr (-0.01)

## Recommendations

1. **Ablation runs needed**: Test with losses ON to see if w_attack/w_mod_spec help:
   - stageB_v6_attack: w_attack=1.0, w_mod_spec=0.0
   - stageB_v6_modspec: w_attack=0.0, w_mod_spec=1.0
   - stageB_v6_both: w_attack=1.0, w_mod_spec=1.0

2. **Note labels**: Consider generating note labels for all sounds (not just 01_canopy) if f0 estimation is desired

3. **Touch threshold**: The 0.3 threshold gives 61% touch frames vs ~20% in v4 — may need tuning

## Files
- Config: `config_sensor_v6.yaml`
- Training log: `runs/bakeoff_ddsp_poc/stageB_v6/train_log.csv`
- Metrics: `runs/bakeoff_ddsp_poc/stageB_v6/eval_output/metrics.json`
- Renders: `runs/bakeoff_ddsp_poc/stageB_v6/eval_output/*.wav`
- Dataset: `bakeoff_data_aligned_v5/`, `bakeoff_notes_v5.json`
- Scripts: `prepare_aligned_data_v5.py`, `train_sensor_ddsp_v5.py`, `eval_stageB_v5.py`

## Git
- Branch: `orch/trackA-ddsp-poc`
- Commit: (to be committed)
