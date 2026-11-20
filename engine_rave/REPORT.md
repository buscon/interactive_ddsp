# Track B (RAVE) — Arm P results and bake-off position

## Gate (task 4, engine_rave/gate_pretrained.py, runs/rave_gate/gate_results.json)
Pretrained `.ts` recon of 20 held-out val files (Stage B metrics):
- water_pondbrain_b2048_r48000_z16: loss_ratio 0.749, rms 0.984, band 0.952 -> PASS
- magnets_b2048_r48000_z8: loss_ratio 0.578, rms 0.414, band 0.958 -> PASS
- birds_dawnchorus_b2048_r48000_z8: loss_ratio 1.047, rms 0.087 -> FAIL (excluded)
No original .ckpt/config found under RAVE_SC* -> Arm S (domain RAVE) would be train-from-scratch, capped.

## Arm P: sensor->latent mapper on frozen water_pondbrain (engine_rave/armP.py, 80k params)
Three configs tried (2000-4000 steps, latent hop 2087 samples / 23 Hz):
1. Latent MSE only (raw z): held-out mean loss_ratio 1.071, rms_ratio 0.623, band_shape 0.955,
   abl_roll2000 0.944, abl_zero 0.970, spearman 0.270 -> timbre/loudness OK, NOT sensor-conditioned
2. Latent MSE + envelope head (z-norm): collapsed quiet (rms 0.02-0.05), abl 0.9+
3. Latent MSE, z-normalized: mean loss_ratio 0.835, rms 0.038 (quiet), abl_roll2000 0.739, spearman -0.09
   (current runs/rave_armP/metrics.json)

## Verdict vs pre-written decision rule
Rule: Arm P primary iff held-out loss_ratio < 1.0 AND band plausible. Best Arm P run has
loss_ratio 0.835-1.07 but sensor-ablation corr 0.74-0.97 (gate: < 0.5) -> Arm P FAILS the
sensor-dependence gate. The mapper does not use the sensor: out-of-domain latents carry
little loudness/gesture information, and small mappers regress to the mean latent.
Arm S (Sentire-domain RAVE from scratch) is the only route to a conditioned RAVE; no .ckpt
available; from-scratch on 100 clips declared out of scope for this PoC (hard cap, high risk).

## Bake-off conclusion (vs Track A DDSP)
- DDSP Track A: held-out loss_ratio 0.38-0.46, rms 0.44-0.78, band-shape 0.93-1.00,
  abl_roll2000 mean 0.065 (roll-half artifact on curve 018 diagnosed & fixed by protocol)
  -> verified sensor-conditioned, audible, timbre-matched.
- RAVE Arm P (pretrained + mapper): good decoder timbre but NOT sensor-conditioned
  (abl 0.74-0.97). Pretrained recon alone is not an answer to sensor-driven synthesis.
