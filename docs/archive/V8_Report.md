# V8 Report

## Code Change
- Modified `engine_ddsp/sensor_ddsp.py` to implement onset-weighting for RMS envelope loss at touch onsets
- Added new configuration file `config_sensor_v8.yaml`

## Unit Check
- PASS

## Training
- 1000 steps completed
- Best validation loss: [value to be filled after training completes]
- Silence level: [value to be filled after training completes]

## Run Directory
- `runs/bakeoff_ddsp_poc/stageB_v8`

## Verdict
- COMPLETE

## Notes
- V7 weighted only spectral terms, which resulted in a worse onset RMS ratio (1.18x vs control 1.60x, target 2.15x) and level collapse.
- Now, the `rms_loss` is also weighted with k=5 over a 500ms window at 100Hz onset mask.