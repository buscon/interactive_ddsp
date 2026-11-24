# V9 lag-fix listening set

**Run:** `v9_stageB_lagfix_s0` (seed 0, 2000 steps), `state_final.pth`, code `ae95734`
**Source:** `~/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/eval_lagfix_s0/`
(the files are byte-identical copies, dry, 48 kHz)

This uses the same config as `listening_v9_best/` (`v9_stageB_fixed_s0`). The only change is
`attack_lag_s`: [0.38, 0.38, 0.42, 0.39] → [0.023, 0.026, 0.065, 0.025] s.
Results: `docs/V9_LAGFIX_RESULTS.md`.

## How to compare

Play the same curve three times: `target`, the new pred here, and the old pred in
`../listening_v9_best/`. Listen for:

- **Attack:** does the touch bump come earlier and louder? The numbers say it is about 40 ms
  earlier and 3 dB louder on average, but still about 100 ms late and 8 dB quiet.
- **Dynamics:** these should not be more squashed than `listening_v9_best/`.
- **forestPad:** its peak is louder, but its timing got *worse* (205 ms vs 194 ms).
- **Pitch** is still too low. The lag fix does not touch pitch.

## Files

16 validation curves, each with `_pred.wav` and `_target.wav`:
canopy 001/007/008/016, forestPad 003/007/013/014, swarm 001/008/016/019, moss 001/009/012/015.
