# Stage B Evaluation Report

## Run Configuration

- Run directory: runs/bakeoff_ddsp_poc/stageB_v4c
- Checkpoint: state_final.pth
- Output directory: runs/bakeoff_ddsp_poc/stageB_v4c_eval
- Sounds: ['01_canopy', '02_forestPad', '03_harmonicSwarm', '04_modularPercussions', '05_moss_and_mist']
- Curves: ['017', '018', '019', '020']
- Dataset split: val

## Validation Loss vs Silence

| Sound | Multiscale Loss | Silence Loss | Loss Ratio |
|-------|------------------|---------------|------------|
| 01_canopy | 9.230066 | 23.216000 | 0.397573 |
| 02_forestPad | 9.215652 | 23.216000 | 0.396953 |
| 03_harmonicSwarm | 9.311685 | 23.216000 | 0.401089 |
| 04_modularPercussions | 9.401502 | 23.216000 | 0.404958 |
| 05_moss_and_mist | 9.852454 | 23.216000 | 0.424382 |

## Metrics

### Overall Mean

| Metric | Value |
|--------|-------|
| Multiscale STFT Loss | 9.402272 |
| RMS Envelope Correlation | 0.771799 |
| Spectral Centroid Correlation | 0.195996 |
| RMS Ratio (pred/target) | 0.528440 |
| Multiscale Loss (ablation zero) | 25.504404 |
| RMS Corr vs Normal (ablation zero) | -0.211315 |
| Multiscale Loss (ablation rolled) | 16.559064 |
| RMS Corr vs Normal (ablation rolled) | -0.115591 |

### Per-Sound Means

| Sound | MS Loss | RMS Env Corr | Centroid Corr | RMS Ratio |
|-------|---------|--------------|---------------|-----------|
| 01_canopy | 9.230066 | 0.831416 | 0.244292 | 0.500803 |
| 02_forestPad | 9.215652 | 0.834031 | 0.354781 | 0.451873 |
| 03_harmonicSwarm | 9.311685 | 0.814205 | 0.079696 | 0.431216 |
| 04_modularPercussions | 9.401502 | 0.534415 | 0.404674 | 0.581461 |
| 05_moss_and_mist | 9.852454 | 0.844926 | -0.103461 | 0.676847 |

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
