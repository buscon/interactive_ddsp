# V8c Results Brief — for Opus verification (project rule: results gate)

**Branch:** `orch/v8c` @ aad3738 · **Date:** 2026-11-22

## Runs completed (all fine-tuned from stageB_v8b, 3000 steps, batch 8, seed 0, lr 3e-4, fast_head @10x)

| Run | Changes | val_loss (best) |
|-----|---------|-----------------|
| stageB_v8c_a | loss freq-norm fix + SC vibrato lag + lagged fast_head + 10x lr; notes v7 blips | 6.72 |
| stageB_v8c_b | same + held per-touch pitch notes (bakeoff_notes_v8c.json) | 6.64 |

## Eval metrics (held-out curves 017–020, 20 files, from raw metrics.json)

| metric | v6 | v7_control | **v8c_a** | **v8c_b** |
|--------|-----|-----------|-----------|-----------|
| multiscale_loss ↓ | 11.75 | 10.91 | 10.31 | **9.56** |
| rms_envelope_corr ↑ | 0.724 | 0.787 | 0.875 | **0.887** |
| rms_ratio (level) | 0.690 | 0.568 | **0.777** | 0.736 |
| touch_onset_f0_err (cents) ↓ | 1201 | 1056 | 1051 | 1298 |
| sustained_touch_corr | 0.320 | 0.390 | **−0.243** ⚠️ | **−0.250** ⚠️ |
| centroid_corr | 0.216 | 0.302 | 0.160 | 0.114 |

(v7 main was: multiscale 9.49, rms_corr 0.817, rms_ratio 0.31 — level collapse now fixed by the loss norm fix.)

## Demo clips (listening_demo/, 3 events with 3 s gaps per sound)

Attack step metric = dB(+400..450ms) − dB(+250..350ms), LP<1.5kHz, per event (ev2 has silent pre in target → ignore):

| sound | target ev1/ev3 | v6 | v8b | v8c_a | v8c_b |
|-------|---------------|-----|-----|-------|-------|
| canopy | +12.1/+15.1 | −3.5/−3.0 | +0.9/+1.0 | −1.8/−1.4 | +2.1/+2.3 |
| forestPad | +8.2/+4.4 | +1.8/+1.8 | −1.0/−0.2 | **+6.0/+6.9** | +0.6/+1.3 |
| swarm | +2.9/+2.5 | +0.5/+0.4 | +1.9/+2.4 | +1.8/+2.8 | +1.6/+2.4 |
| modPerc | −7.6/+4.5 | +1.3/+1.5 | +2.3/+2.8 | +2.7/+3.6 | −1.5/−0.3 |
| moss | +11.0/+6.1 | −0.2/+0.0 | +3.8/+4.3 | +3.2/+4.3 | **+6.1/+7.5** |

## Questions for Opus (verify against raw artifacts yourself)

1. Verify the metrics table by recomputing means from the raw metrics.json files. Any metric that looks miscomputed?
2. Demo attack steps: v8c arms clearly beat v6/v8b on forestPad/moss/swarm/modPerc, but **canopy is still weak** (−1.8..+2.3 vs target +12/+15). Why would canopy lag? (Its lag D=0.38 was measured; its touch layer is a sine "drop" at the note pitch.)
3. **sustained_touch_corr went negative** (−0.24) after the vibrato-lag change. Is that a real regression in sustained modulation, or a metric artifact (metric masks touch_time>2s; with 0.98 data the sustained region starts later)? How to check cheaply?
4. centroid_corr dropped (0.30 → 0.16/0.11). Real brightness loss or artifact?
5. Arm B (held pitch) beats A on multiscale/rms_corr but loses on onset_f0_err (1298 vs 1051) and some demo steps. Is B's pitch input helping or hurting the attack, and why would onset_f0_err worsen?
6. Overall verdict: does v8c satisfy "strong touch attack" (≥+6 dB step on ≥2/3 events)? What is the single highest-value next change?

## Artifacts
- runs: `runs/bakeoff_ddsp_poc/stageB_v8c_{a,b}/` (state_final.pth, train_log.csv, eval_v8c/metrics.json)
- demos: `listening_demo/<sound>_{target,stageB_*}.wav`
- labels: `bakeoff_notes_v8c.json`, `attack_lag_report.json`, `make_note_labels_v8c.py`
- code: branch orch/v8c (loss norm, vibrato lag, fast_head lag+10x lr)
