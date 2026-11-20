# Bake-off comparison: Track A (DDSP) vs Track B (RAVE) — sensor-driven synthesis on Sentire

## Shared protocol
- Dataset: `bakeoff_data_aligned/` (100 files, 5 sounds x 20 curves; sensor time = audio - 1.0 s, verified).
- Held-out: curves 017-020 x 5 sounds (20 files). Same metric code (eval_stageB.py).
- Shared sensor-ablation protocol (from curve-018 diagnostic): circular sensor roll 2000 frames (20 s),
  >=5 seeds, pass = mean ablation RMS-envelope corr < 0.5; secondary sensor=0 & touch=0.

## Tier 1 — target fidelity (held-out means)
| system | loss_ratio | rms_ratio | peak | band_shape |
|--------|-----------|-----------|------|-----------|
| DDSP Track A (Stage B) | 0.38-0.46 | 0.44-0.78 | 0.20-0.49 | 0.93-1.00 |
| RAVE Arm P (pretrained + mapper) | 0.84-1.07 | 0.04-0.62 | 0.02-0.25 | 0.78-0.96 |
| RAVE pretrained recon only (water_pondbrain) | 0.749 | 0.984 | 1.017 | 0.952 |

## Tier 2 — sensor dependence (the bake-off question)
| system | abl_roll2000 mean | abl_zero mean | spearman(sensor, RMS env) | gate < 0.5 |
|--------|------------------|---------------|---------------------------|-----------|
| DDSP Track A | 0.065 (roll-half; roll-2000: +0.13..-0.29) | ~-0.32 | strong | PASS |
| RAVE Arm P | 0.74-0.97 | 0.87-0.99 | -0.09..0.27 | FAIL |

## Tier 3 — practical
- DDSP: SensorDDSP trains in minutes (2000 steps on RTX 5090); not real-time-tested here.
- RAVE: pretrained .ts inference ~62 ms per 10 ms block on CPU (not real-time on CPU; GPU real-time
  by design), but the pretrained decoder is out-of-domain for Sentire timbre.

## Conclusion
DDSP Track A is the only arm that achieves verified sensor conditioning on held-out curves
(ablation corr ~0.07 vs gate 0.5; no silence collapse: min peak 0.20). RAVE Arm P reproduces
timbre through the pretrained decoder but the sensor->latent mapper does not pick up the sensor
(abl 0.74-0.97): out-of-domain latents carry little gesture information. A domain-trained RAVE
(Arm S) is the only route to a conditioned RAVE; no .ckpt available, from-scratch declared out
of scope for this PoC. The curve-018 anomaly in Track A was a roll-by-half protocol artifact,
not a model weakness (see docs/trackA_poc_issue1.md section 7 on branch orch/bakeoff-report).
