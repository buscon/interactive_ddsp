# V9 Status and Next Steps

**Date:** 2026-11-24
**Branch:** `orch/v9data-fix`
**Sources:** `FULL_REPORT_v6_v9.md`, `V9_STAGEB_RESULTS.md`, `V9_DATA_ANALYSIS.md`,
`V9_DATA_FIX_REPORT.md`, `V9_STAGEB_FINAL_REPORT.md`, `V9_F0_FREEZE_POSTMORTEM.md`.
The Stage B attack numbers below were re-read from each run's `attack_summary.json`. The
other numbers come from the docs.

---

## 1. What we did

1. **Touch audit.** The old (v8) dataset had a 0.35-0.40 s lag between sensor and audio,
   caused by SuperCollider server latency. V6-V8 learned a delay the live instrument doesn't
   have.
2. **v6 dataset.** 5 sounds × 20 curves × 60 s, 7 channels on one sample clock: master,
   isolated touch layer, sensor, touch gate and note frequency. The data check passed.
   The audio is already on the sensor clock. The touch-layer onset follows the touch by
   2-44 ms, plus a 21.3 ms limiter delay on the master.
3. **First v9 run.** It trained without errors, but **the percussive attack at touch was not
   learned for any sound.** Opus ran a controlled comparison on one fixed batch the model was
   overfit to (training data only). The attack appeared when the loudness-envelope (RMS)
   loss weight went from 1 to 10 and the onset weighting peaked at 40 ms. Fixing the pitch
   did not help the attack.
4. **Stage B (held-out curves): NO-GO.** The attack timing that worked on training data did
   not carry over: attacks were 53-134 ms late, peaks 5-6 dB too quiet, 0/5 sounds passed.
5. **Data fix.** Removed `04_modularPercussions` (now 4 sounds: 0 = canopy, 1 = forestPad,
   2 = swarm, 3 = moss). Re-split train/val/test so each split has a similar mix of short and
   long touches.
6. **Three bugs found:**
   - Swarm modulation from the first frame: fixed with `lfo_depth_mul_base = 0`. Confirmed by
     listening ("very good").
   - Squashed dynamics: `touch_amp_head: true` and `w_attack: 0.1`. Reported as good after
     retraining.
   - Pitch about 2 octaves low: **not fixed.** The pitch head sits at its −1 limit, where its
     gradient is zero, and no loss pulls it back.
7. **Freezing the pitch head (f0 = note) was reverted.** Pitch became correct and the attack
   numbers improved, but in the listening test the dynamics were squashed again and the swarm
   modulation was back.

## 2. Current state

The branch is back at the state with good dynamics and broken pitch (`45a61ce` plus docs).
**Nothing passes yet.**

Stage B on the validation split, seed 0, from `attack_summary.json` (targets: |dpk| ≤ 4 dB,
|dtpk| ≤ 20 ms):

| Run | Changes | dpk (dB) | dtpk (ms) | Pass | Listening |
|---|---|---|---|---|---|
| `v9_stageB_A4_s0_corrected` | baseline, new split | −10.0 | 52 | 0/4 | dynamics squashed, swarm modulated |
| `v9_stageB_fixed_s0` | + LFO fix, touch_amp_head, w_attack | −10.9 | 155 | 0/4 | **best by ear**, pitch 2 octaves low |
| `v9_stageB_f0frozen_s0` | + f0 = note | −8.1 | 98 | 0/4 | squashed again, modulation back |

The best results on held-out curves are still about 100-155 ms late and 8-11 dB too quiet
at the attack. The numbers and the listening do not agree on which run is best.

## 3. Problems in the current setup

### 3.1 `attack_lag_s` is back at the old v8 value of 0.4 s (new finding, likely important)

Commit `d3db5ef` set `attack_lag_s = [0.38, 0.38, 0.42, 0.39]` in all v9 configs and in the
model's default buffer. Its reason was that the measured values (0.02-0.06 s) were
"refinement deltas". That reason is wrong:

- Those 0.38-0.42 values come from `attack_lag_report.json`, which was measured on the
  **v8** dataset. That dataset had the 0.4 s server-latency offset the v6 re-record was made
  to remove.
- `V9_LAG_MEASUREMENT.md` measured the v6 touch layer directly and recommends
  `[0.023, 0.026, 0.065, 0.025]` s for the master (percussion removed).
- `V9_TRAINER.md` and `FULL_REPORT_v6_v9.md` §5 both state that v6 audio is on the sensor
  clock.

So in every Stage B run since `d3db5ef`, the gated attack path (`fast_head`) opens about
0.35 s after the real attack in the target. That fits the large positive dtpk (late attacks)
seen in every run, and it would also explain why `fast_head` contributes almost nothing.
This has not been tested yet. It is the first thing to check.

### 3.2 The freeze postmortem's explanation is unproven

The postmortem says the pitch head was "secretly helping dynamics" through shared features.
That rests on a single run with a single seed, and earlier seeds differed a lot (53 vs
134 ms). Two simpler explanations have not been ruled out:

- At the correct (about 4× higher) pitch, the harmonic synth has a different spectral
  balance, so the loss pushes amplitude differently.
- Normal seed-to-seed variation.

The swarm modulation coming back is also odd, given that `lfo_depth_mul_base = 0` was in the
config. The listening fix (`render_swarm_fix2.py`) zeroed both `lfo_depth_mul_base` **and**
the learned `lfo_depth_mul_emb`. The config only zeroes the base, so the embedding can
re-learn the modulation during training. That is the more likely cause.

### 3.3 Pitch

- The note data and the audio disagree. For swarm, `note_hz` gives about 150-178 Hz, but the
  audio pitch tracker gives about 247 Hz, roughly 5 semitones apart. Both come from the same
  recording, so the tracker may be locking onto a harmonic. This needs settling before any
  pitch loss is added.
- "Raw Hz input" is not the root cause. The pitch was already stuck before the note input
  was added: ×3.97 in the first v9 run and ×0.25 in the overfit test. The real problem is
  the clamp/tanh limit with nothing anchoring the pitch.
- Known loss bug: the f0-regression mask is computed after `nan_to_num`, so unvoiced frames
  are treated as 100 Hz.

### 3.4 Evaluation

- The attack metrics don't catch squashed dynamics. The frozen run scored better and
  sounded worse.
- The validation loss only covers the first 10 s of each file (open item D6), so the saved
  "best" checkpoint is weak evidence.
- Only one seed was run per arm.

## 4. How to improve it

In priority order. Each step is an experiment plan that needs Opus review before dispatch
(AGENTS.md).

**Step 1: Restore the measured attack lag (small, do first)**
- Set `attack_lag_s: [0.023, 0.026, 0.065, 0.025]` in the v9 configs and the model default.
- Retrain `v9_stageB_fixed` with 2 seeds, changing nothing else.
- Success: dtpk falls clearly from 155 ms toward ≤ 20 ms, and `fast_head` contributes more
  than 0.5 dB at onsets.

**Step 2: Make the swarm LFO fix stick in training**
- Zero and freeze `lfo_depth_mul_emb` for swarm, or add the time ramp on `depth_mul` that
  `03_harmonicSwarm_fixed/README.md` proposed, so the modulation can't be re-learned.

**Step 3: Fix pitch without disturbing the other outputs**
- Set `f0 = note_hz × 2^(d)`, with `d` limited to about ±0.25 octave and no hard clamp.
- Feed the head `log2(note / f0_base)` instead of raw Hz, and start its last layer at zero.
- Add a small L2 penalty on `d` instead of a pitch loss on the audio, and fix the NaN mask.
- First run pYIN on the isolated touch layer (channels 2/3) to settle the note-vs-audio
  mismatch.

**Step 4: Fix how experiments are run**
- At least 2 seeds per arm, and change one thing at a time.
- Add measures of dynamics to the eval (crest factor, envelope dynamic range, attack peak
  over sustain), so a squashed result shows up in the numbers and not only by ear.
- Validate on whole files with onset-aware scoring (D6), so the saved "best" checkpoint
  actually reflects the attack.

**Step 5: Close the gap between training and held-out curves on the attack**
- Apply the change that worked in the overfit comparison (RMS weight 10, onset peak 40 ms)
  to the full held-out run. Also lower the learning rate or let it decay.
- **Use the touch layer as a second training target.** Channels 2/3 hold the isolated touch
  layer, which is exactly the missing +10-15 dB bump. One option is two branches: a base
  sound driven by the sensor, plus a touch-layer branch gated by touch and trained on
  channels 2/3, with the master as their sum. This separates the attack from the base sound
  and makes each part easier to learn.
- Add training variety: slight time-stretching around attacks and random crops centred on
  onsets.

## 5. Listening

The current best file set is in `listening_v9_best/` (see its README). Earlier listening
sets (A4 corrected, the inference-time swarm and dynamics fixes, and the f0-frozen run) were
moved to `~/sentire/Recordings/Dataset_v6_20261122_aligned/listening_archive/`.
