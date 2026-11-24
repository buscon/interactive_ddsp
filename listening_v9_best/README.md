# V9 best listening set

**Run:** `v9_stageB_fixed_s0` (seed 0, 2000 steps), `state_final.pth`
**Source:** `~/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/eval_fixed_s0/`
(the files are byte-identical to the eval renders, dry, 48 kHz)
**Code state:** `orch/v9data-fix` at `45a61ce` (the f0 freeze was reverted)

This is the best version by ear so far: dynamics work (`touch_amp_head: true`) and the swarm
attack is clean (`lfo_depth_mul_base = 0`). It is still far from the targets.

## Known defects to listen for

- **Pitch is too low** (about ×0.25 of the note for swarm). The f0 head saturates.
- **The attack is late and weak**: overall dtpk +155 ms and dpk −10.9 dB (`attack_summary.json`).
  The likely cause is `attack_lag_s` = 0.4 s, inherited from v8. See
  `docs/V9_STATUS_AND_NEXT_STEPS.md` §3.1.

## Files

16 validation curves (4 per sound), each with `_pred.wav` (model) and `_target.wav` (recording).

| Sound | Curves |
|---|---|
| 01_canopy | 001, 007, 008, 016 |
| 02_forestPad | 003, 007, 013, 014 |
| 03_harmonicSwarm | 001, 008, 016, 019 |
| 05_moss_and_mist | 001, 009, 012, 015 |

Play the target first, then the pred.

Older sets (A4 corrected, the inference-time swarm and dynamics fixes, f0 frozen) are in
`~/sentire/Recordings/Dataset_v6_20261122_aligned/listening_archive/`.
