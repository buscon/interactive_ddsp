# Track A Proof-of-Concept Report: DDSP vs RAVE Bake-off

## TL;DR

- **Dataset**: 100 files (20 curves × 5 sounds) from Sentire recordings, aligned with sensor time = audio − 1.0 s + 0.4 s physical delay
- **Model**: SensorDDSP with sensor/touch MLPs, harmonic+noise synthesis, learnable reverb, curriculum loss, and tilted initialization
- **Training**: 15 runs (13/15 passed core+ablation on training chunk), 2 silence collapses (01_canopy_1, 02_forestPad_2), band-shape corr recomputed on all 15 overfit checkpoints: 0.38–1.00 (median ≈ 0.95)
- **Stage B**: 2000 steps, batch 8, 0 restarts, validation loss 7.34 vs silence 23.22; held-out evaluation shows loss_ratio < 0.6, RMS ratio 0.44–0.78, peak 0.20–0.49, ablation rolled-corr mean 0.065
- **Sensor Conditioning**: Evidence suggests sensor conditioning works, but resolved: roll-by-half artifact on quasi-periodic curve 018; shared protocol = roll-2000
- **Caveats**: Centroid-envelope metric artifact, per-sound ablation table bug in STAGE_B_REPORT.md, loss-ratio denominator convention

## 1. Dataset & Alignment

### Dataset Overview

- **Source**: Sentire recordings from dataset `/home/marcello/sentire/Recordings/Dataset_20261120_110047/`
- **Files**: 100 files (20 curves × 5 sounds) in `bakeoff_data_aligned/`
- **Alignment**: Sensor time = audio − 1.0 s + 0.4 s physical delay

### Alignment Check

- **Method**: Independent analysis of sensor-audio alignment using RMS envelope, log envelope, and spectral centroid features
- **Best Feature Pair**: Sensor level and RMS envelope
- **Results**: cross-correlation lags cluster at ≈ +0.4 s across sounds (04_modularPercussions inconclusive at 15% aligned frames)

## 2. Model

### Model Architecture

- **SensorDDSP**: Sensor/touch MLPs, harmonic+noise synthesis, learnable reverb, curriculum loss, and tilted initialization

## 3. Training Recipe & Collapse Analysis

### Training Recipe

- **Curriculum Loss Ramp**: 150–400 steps
- **Guard**: Step 400
- **Warm-up**: 100 steps

### Collapse Analysis

- **Collapsed Runs**: 2 silence collapses (01_canopy_1, 02_forestPad_2)
- **Non-Collapsed Runs**: Band-shape correlation 0.75–1.0

## 4. Overfit Gate

### Overfit Gate Results

- **Passed Runs**: 13/15
- **Failed Runs**: 2 silence collapses

## 5. Stage B Held-Out Results

### Stage B Configuration

- **Steps**: 2000
- **Batch Size**: 8
- **Restarts**: 0
- **Validation Loss**: 7.34 vs silence 23.22

### Stage B Metrics

- **Loss Ratio**: < 0.6
- **RMS Ratio**: 0.44–0.78
- **Peak**: 0.20–0.49
- **Ablation Rolled-Corr Mean**: 0.065

## 6. Sensor Conditioning Evidence

### Sensor Conditioning Evidence

- **Evidence**: Suggests sensor conditioning works, but resolved: roll-by-half artifact on quasi-periodic curve 018; shared protocol = roll-2000

## 7. Curve-018 Ablation Anomaly

### Curve-018 Ablation Anomaly

- **Per-curve mean rolled-corr**: 017 = -0.251, **018 = +0.609 (5/5 entries > 0.5, max +0.682)**, 019 = -0.015, 020 = -0.082; zero-ablation on 018 is normal (mean -0.177)

- **Curve-018 sensor stats**: mean 0.551, std 0.331, 22.7% of frames pinned at 1.000, quasi-periodic with period ≈ 30–33 s ≈ T/2, autocorr at T/2 = +0.348 → roll-by-half is a partial identity (render corr tracks sensor autocorr at the roll lag: +0.35→+0.6, 30 s→+0.3, 10 s→-0.1, 20 s→-0.25)

- **Variant comparison on 018 (mean over 5 seeds)**: roll_half +0.55…+0.68 (artifact), roll_1000 -0.14…-0.10, roll_2000 -0.29…-0.20, zero -0.48…-0.06, cross-curve swap +0.50…+0.72 (false pass, donors correlated +0.4…+0.6), touch_zero ≈ +1.00 (touch carries nothing), sensor_zero -0.53…-0.26 (sensor carries the envelope)

- **Verdict: metric artifact, not a real weakness** — model is genuinely sensor-conditioned on 018; re-scored with roll-2000 it passes the < 0.5 gate (≈ -0.2)

- **Recommended shared ablation protocol (DDSP + RAVE)**: primary = circular sensor roll by 2000 frames (20 s @100 Hz), touch recomputed with the dataset rule, per-file guard: if |autocorr(2000)| ≥ 0.2 take smallest lag ≥ 2000 with |ac| < 0.1; secondary = sensor=0 & touch=0; rejected = roll-by-T/2 and cross-curve swap unless donor |corr| < 0.1; ≥5 seeds per render, report mean ± std; pass threshold = mean ablation RMS-envelope corr < 0.5 per entry (optional max-over-seeds < 0.6). Observed over all 20 held-out entries with roll-2000: +0.13…-0.29.

## 8. Caveats & Reproducibility

### Caveats

- **Centroid-Envelope Metric Artifact**: Artifact in centroid-envelope metric
- **Per-Sound Ablation Table Bug**: Bug in per-sound ablation table in STAGE_B_REPORT.md
- **Loss-Ratio Denominator Convention**: Loss-ratio denominator convention

### Reproducibility

- **Verification Story**: An independent re-render of the original Stage B checkpoint reproduced the stored metrics within ≤1.01% at entry level; the artifacts were later regenerated in the main tree (re-run checkpoint), and aggregate means match the original report exactly.

## 9. Pointers to Track B

### Pointers to Track B

- **Track B**: Further investigation needed for Curve-018 anomaly

## Conclusion

- **Overall**: Dataset appears correctly aligned, model architecture is well-defined, training recipe is well-defined, overfit gate results are well-defined, Stage B results are well-defined, sensor conditioning evidence is well-defined, Curve-018 ablation anomaly is well-defined, caveats and reproducibility are well-defined, and pointers to Track B are well-defined

## References

- **Alignment Check Report**: `/home/marcello/trackA_poc_reports/alignment_check.md`
- **Model**: `engine_ddsp/sensor_ddsp.py`
- **Overfit Table**: `runs/bakeoff_ddsp_poc/RESULTS.json`
- **Stage B**: `runs/bakeoff_ddsp_poc/stageB_eval/metrics.json` + `/home/marcello/trackA_poc_reports/stageB/`

## Report Generated

- **Date**: 2026-11-20
- **Data Directory**: `/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned`
- **Files Analyzed**: 100

## End of Report
