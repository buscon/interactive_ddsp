# V7 Evaluation Report

**Date:** 2026-11-22 · **Branch:** `orch/v7-eval` · **Held-out set:** curves 017–020 (20 files, split=val, 5 sounds × 4 curves)

Models evaluated (all `state_final.pth`, rendered with `eval_stageB_v5.py` → `eval_v7grid/`):

| Model | Run dir | Onset weighting |
|-------|---------|-----------------|
| V6 baseline | `runs/bakeoff_ddsp_poc/stageB_v6` | none |
| V7 control | `runs/bakeoff_ddsp_poc/stageB_v7_control` | OFF |
| V7 main | `runs/bakeoff_ddsp_poc/stageB_v7` | ON (k=5, 500 ms window) |

## Aggregate metrics (mean over 20 files, curves 017–020)

| Model | multiscale_loss ↓ | rms_envelope_corr ↑ | centroid_corr ↑ | silence_loss | rms_ratio | touch_onset_f0_err_cents ↓ | sustained_touch_corr |
|-------|-------------------|---------------------|-----------------|--------------|-----------|----------------------------|----------------------|
| V6 baseline | 11.749 | 0.724 | 0.216 | 23.216 | 0.690 | 1201 | 0.320 |
| V7 control (OFF) | 10.914 | 0.787 | 0.302 | 23.216 | 0.568 | 1056 | 0.390 |
| V7 main (ON, k=5, 500ms) | **9.489** | **0.817** | **0.336** | 23.216 | 0.309 | **790** | 0.240 |

Notes:
- `silence_loss` is a constant reference (23.216, from training log) — identical across models by construction; loss ratio (multiscale/silence) improves for V7: 0.41 vs 0.47 (control) vs 0.51 (V6).
- Additional V7-vs-control deltas: `hf_ratio_pred` −10.12 vs −10.15 (target −12.92; V6 −12.00), `mod_depth_pred` 0.372 vs 0.442 (target 0.526), ablation-zero multiscale 19.44 vs 22.82 (V6 12.08).
- ⚠️ `rms_ratio` (overall pred/target level) drops to **0.309** for V7 vs 0.568 control — V7 renders markedly quieter overall.

## Onset-window RMS ratio (500 ms post / 500 ms pre, onsets from `bakeoff_notes_v7.json`)

30 touch onsets across the 20 held-out files (curves 017/018: 1 onset, 019/020: 2 onsets each, × 5 sounds). Computed per onset for pred and target; see `onset_rms_eval.py` / `onset_rms_results.json`.

| Model | Pred median | %>2x | Target median | pred/target |
|-------|-------------|------|---------------|-------------|
| V6 | 1.42x | 27% | 2.15x | 0.66 |
| control | 1.60x | 30% | 2.15x | 0.75 |
| V7 main | 1.21x | 20% | 2.15x | 0.56 |

Robustness check (exclude 4 onsets with silent pre-baseline, pre-RMS < −80 dBFS; n=26; target median 1.79x):

| Model | Pred median | %>2x | pred/target |
|-------|-------------|------|-------------|
| V6 | 1.31x | 15% | 0.73 |
| control | 1.56x | 19% | 0.87 |
| V7 main | 1.18x | 8% | 0.66 |

Target distribution: median 2.15x, 53% of onsets >2x (raw ratios range 0.96–6.3x, plus 4 silent-baseline outliers).

## Verdict

- **Does V7 main improve onset RMS ratio vs control/V6? ❌** On the key onset-window metric V7 is *not* better: median pred ratio 1.21x vs 1.60x (control) / 1.42x (V6) against a target of 2.15x, and fewer onsets exceed 2x (20% vs 30%/27%). The onset weighting did not translate into a larger post/pre RMS jump at touch onsets on held-out curves.
- **Does V7 preserve/improve the other metrics? ✅ (mixed)** multiscale_loss 9.49 (best), rms_envelope_corr 0.817 (best), centroid_corr 0.336 (best), touch_onset_f0_err 790 cents (best, −25% vs control). `silence_loss` is a fixed reference and unchanged. However overall level (`rms_ratio`) drops to 0.31 and `sustained_touch_corr` drops to 0.240 — V7 trades level and sustain for onset-f0 accuracy.
- **Recommendation: need more work — do not ship V7 as-is.** V7 clearly improves overall spectral/envelope fit and onset pitch accuracy, but it does not improve the onset RMS-ratio target and renders ~45% quieter than control relative to target (compensable with gain, but the onset prominence itself is lower). Next steps: (a) check whether onset weighting is being swamped by the silence/level term (increase k or window), (b) add a level/loudness term to prevent the rms_ratio collapse, (c) re-audit onset alignment on the held-out curves before re-training.

## Reproduction

```bash
cd /tmp/opencode/wt-v7-eval
PY=/home/marcello/Documents/interactive_ddsp/.venv/bin/python
$PY eval_stageB_v5.py --run runs/bakeoff_ddsp_poc/stageB_v6          --ckpt state_final.pth --out runs/bakeoff_ddsp_poc/stageB_v6/eval_v7grid
$PY eval_stageB_v5.py --run runs/bakeoff_ddsp_poc/stageB_v7_control   --ckpt state_final.pth --out runs/bakeoff_ddsp_poc/stageB_v7_control/eval_v7grid
$PY eval_stageB_v5.py --run runs/bakeoff_ddsp_poc/stageB_v7          --ckpt state_final.pth --out runs/bakeoff_ddsp_poc/stageB_v7/eval_v7grid
$PY onset_rms_eval.py   # writes onset_rms_results.json
```
