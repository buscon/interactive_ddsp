# V9 trainer changes for dataset v6

Status: code ready on branch `orch/v9train`; smoke-tested (30 steps) only. NO real training has been run. Results gate by Opus: conditional GO (all blocking findings fixed, see below).

## Key fact (corrected during the work)
The v6 npz audio is ALREADY on the sensor clock (sensor = ch4 of the same aiff; touch-layer onset is 2-45 ms after the touch_time onset, unshifted). `offset_s` only relates the recording to the planned curve. An earlier plan to shift audio by offset_s was wrong and was reversed: `align_audio` defaults to False (the shift+trim code remains only for legacy data). The model's `attack_lag_s` therefore only holds the real synth lag (+ the 21.3 ms limiter delay of the master target), no longer the 0.4 s recording artifact.

## Changes
- `engine_ddsp/sensor_data_v5.py`: provenance whitelist (v5/v8 root and v6 root), `align_audio` kwarg (default False), `T_eff` consistent in `__getitem__`/`_count_chunks`/`full()` when enabled, `list.index` crash in deterministic mode fixed (`enumerate`). Old dataset output byte-identical (verified).
- `engine_ddsp/sensor_ddsp.py`: `attack_lag_s` kwarg and `set_attack_lag()`; `onset_weighting` now passes ALL keys (before, `peak_ms`/`pre_ms` were silently dropped, so v8c ran with peak 350/pre 100, not the configured values).
- `train_sensor_ddsp_v5.py` / `eval_stageB_v5.py`: index and val curves from the dataset (v6 val 004/011/017, test 007/014/020), attack lag re-applied and asserted after any checkpoint load, paths relative to the script, eval f0 autocorrelation via FFT (27-45 s -> 0.01 s per call, same result to 1e-12), eval accepts `--split test`.
- `render_sensor_ddsp.py`, `make_onset_demo.py`: data root / notes from config, v5 loader.
- `config_sensor_v9.yaml`: from scratch (no finetune block), `attack_lag_s [0.023, 0.026, 0.065, 0.040, 0.025]` (touch-layer lag 2/5/44/22/4 ms + 21.3 ms), onset weighting pre 25 / peak 90 / window 400 ms, `align_audio: false`.
- `analyze_v6_lag.py`, `docs/V9_LAG_MEASUREMENT.md`: measurement (its headline about the shifted timeline is superseded; the unshifted tables are valid).

## Launch (when the user says go; from the main checkout after merging)
`python3 train_sensor_ddsp_v5.py --config config_sensor_v9.yaml --name <run> --root runs/bakeoff_ddsp_poc --steps 2000 --batch 8`
The `train:` block of the YAML is NOT read by the trainer (CLI defaults are steps 800, batch 1; LR = warm-up then constant 1e-3).

## Open decisions (not done)
- D4: canopy additive vibrato `depth_add_t = clamp(0.5*t, 20)` is applied to all 5 sounds (config says only canopy has it). Suggested: multiply by `(lfo_depth_add_base > 0)`. Model behaviour change, so not applied.
- D6: validation loss / `state_best` use only the first 10 s of each val file (4-5 s of silent lead-in, 1-2 of 4-6 onsets). Suggested: chunk over the full file.
- Known/left: `frame_weights` "sustained touch > 2 s" fires after ~69 ms (channel 0 is log1p(t)); eval ablation "touch recomputation" warning compares the wrong arrays; `make_onset_demo` does not pass `max_f0_octaves`; `test_onset_weighting.py` test 1 is flaky (random init, also fails at f9e78d0); `test_v8c.py` needs a checkpoint that is not in git.
