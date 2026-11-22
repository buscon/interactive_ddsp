# Stage B Evaluation Report

## Run Configuration

- Run directory: runs/bakeoff_ddsp_poc/stageB_v6
- Checkpoint: state_best.pth
- Output directory: runs/bakeoff_ddsp_poc/stageB_v6/eval_output
- Sounds: ['01_canopy', '02_forestPad', '03_harmonicSwarm', '04_modularPercussions', '05_moss_and_mist']
- Curves: ['017', '018', '019', '020']
- Dataset split: val

## Validation Loss vs Silence

| Sound | Multiscale Loss | Silence Loss | Loss Ratio |
|-------|------------------|---------------|------------|
| 01_canopy | 9.690393 | 23.216000 | 0.417402 |
| 02_forestPad | 9.621303 | 23.216000 | 0.414426 |
| 03_harmonicSwarm | 9.741301 | 23.216000 | 0.419594 |
| 04_modularPercussions | 11.378355 | 23.216000 | 0.490108 |
| 05_moss_and_mist | 11.377469 | 23.216000 | 0.490070 |

## Metrics

### Overall Mean

| Metric | Value |
|--------|-------|
| Multiscale STFT Loss | 10.361764 |
| RMS Envelope Correlation | 0.737933 |
| Spectral Centroid Correlation | 0.265383 |
| RMS Ratio (pred/target) | 0.490348 |
| Multiscale Loss (ablation zero) | 16.542518 |
| RMS Corr vs Normal (ablation zero) | 0.034137 |
| Multiscale Loss (ablation rolled) | 15.540427 |
| RMS Corr vs Normal (ablation rolled) | -0.102578 |

### Per-Sound Means

| Sound | MS Loss | RMS Env Corr | Centroid Corr | RMS Ratio |
|-------|---------|--------------|---------------|-----------|
| 01_canopy | 9.690393 | 0.799452 | 0.489066 | 0.467079 |
| 02_forestPad | 9.621303 | 0.784079 | 0.417775 | 0.299446 |
| 03_harmonicSwarm | 9.741301 | 0.731883 | 0.304870 | 0.460235 |
| 04_modularPercussions | 11.378355 | 0.581660 | 0.125078 | 0.650856 |
| 05_moss_and_mist | 11.377469 | 0.792592 | -0.009875 | 0.574126 |

## Ablation Results

| Sound | Ablation Zero MS Loss | Ablation Rolled MS Loss | RMS Corr Zero | RMS Corr Rolled |
|-------|------------------------|--------------------------|---------------|------------------|
| 01_canopy | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 02_forestPad | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 03_harmonicSwarm | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 04_modularPercussions | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 05_moss_and_mist | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Pass/Fail Criteria

- Audible: peak > 1e-3, RMS 0.4-2.0
- Loss ratio < 0.6
- Ablation shuffled < 0.5
- Band-shape > 0.65

## Notes

- 04_modularPercussions expected weaker fit
