# RAVE arm decision rule (pre-written before mapper training)

## Gate results (pretrained .ts recon of held-out val curves 017-020, 20 files, Stage B metrics)
| model | loss_ratio (vs silence) | rms_ratio | peak | band_shape_corr |
|-------|------------------------|-----------|------|-----------------|
| birds_dawnchorus_b2048_r48000_z8 | 1.047 | 0.087 | 0.216 | 0.913 |
| magnets_b2048_r48000_z8 | 0.578 | 0.414 | 0.703 | 0.958 |
| water_pondbrain_b2048_r48000_z16 | 0.749 | 0.984 | 1.017 | 0.952 |

Details: runs/rave_gate/gate_results.json (per-file). Input normalization: npz audio is int16-scale, divided by 32768.

## Decision rule
- If Arm P (pretrained encoder + sensor->latent mapper) held-out loss_ratio < 1.0 vs silence AND band-shape plausible (> 0.65), Arm P is the primary RAVE arm.
- Otherwise Arm S (Sentire-domain RAVE) becomes primary.
- Report both arms if both ran.

## Verdict
- water_pondbrain and magnets pass the gate as recon backbones (loss_ratio 0.75 / 0.58 < 1.0, band-shape ~0.95).
- birds fails recon (loss_ratio 1.05, rms 0.09) -> excluded.
- No original .ckpt/config found under RAVE_SC* (checked 2026-11-20), so Arm S would mean train-from-scratch with a hard cap; only run it if Arm P fails the rule.
- Primary arm: Arm P with water_pondbrain encoder/decoder (best loudness match), magnets as secondary.
- Label Arm P results "out-of-domain RAVE": the pretrained decoder cannot reproduce Sentire timbre beyond its training domain.

## Shared ablation protocol (from Track A curve-018 diagnostic)
Circular sensor roll by 2000 frames (20 s @ 100 Hz), touch recomputed with the dataset rule,
per-file guard: if |autocorr(2000)| >= 0.2 take smallest lag >= 2000 with |ac| < 0.1.
Secondary: sensor=0 & touch=0. >=5 seeds per render, mean +/- std.
Pass threshold: mean ablation RMS-envelope corr < 0.5 per entry.
