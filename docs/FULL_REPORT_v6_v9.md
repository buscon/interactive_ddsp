# Full report: touch-event audit, v6 re-record, v9 trainer and first v9 run

Period: 2026-11-22 to 2026-11-23. Orchestrator Claude Sonnet 5.5, all programming by Haiku (claude-haiku-5-5), plans and results reviewed by Opus (AGENTS.md gate). Branches: `orch/touch-audit`, `orch/touch-sync`, `orch/v6data` (recorder/prepare/validator), `orch/v9train` (trainer). Nothing pushed or merged; `orch/v8c` untouched (f9e78d0, 14 untracked files protected).

## 1. Goal
Models V6-V8c produced a weak percussive attack at touch events. First question: is the touch event fully and correctly in the training data? (touch = sensor crossing >= 0.98 from below; Sentire's touch trigger is strict `>` on the post-scaling touchDetectionSignal, no hysteresis.)

## 2. Audit of the existing (v8) dataset (`docs/TOUCH_DATASET_AUDIT.md`)
- Touch onsets are complete and exact in the old dataset (230 crossings, labels, touch_time consistent).
- Audio lagged the sensor by 0.35-0.40 s per file. Cause: `ctl.set` goes through `NodeProxy.set -> server.sendBundle(server.latency, ...)` with serverLatency 0.4 s, while `rec.resume` is sent immediately. Offset = 0.4 - lateness of the resume (64-sample blocks).
- Consequence: the old models learned a 0.4 s delay, mixed into `attack_lag_s` [0.38,0.38,0.42,0.245,0.39] and the onset-weight window (pre 100 / peak 400 / window 800 ms). The live instrument has no such delay.
- Several Haiku claims in the first audit were corrected by Opus (e.g. analyzer offset "4.885 s" included the 4.51 s curve lead; real 0.379 s).
- Your clarifications: the soundcard setting is not a problem; target/prediction timing looked synced (Opus: that only shows the model learned the 0.4 s delay).

## 3. TouchSync test (`docs/TOUCH_SYNC_RESULTS.md`)
10-file 7-channel re-record with the audio-rate touch proxy (`touchProxy`) recorded on the same sample clock as audio. Findings: the gate rises exactly 64 samples (1.33 ms) after the 0.98 crossing; touch-layer onset medians canopy ~3-5, forestPad ~2.5-3.5, swarm ~43-45, modPerc ~21-22.5, moss ~2.3-3.5 ms; master = touch layer + 1024 samples (21.33 ms, limiter lookahead). The OSC `/sentire/touch` is polled at 0.1 s, so the audio-rate proxy is the real trigger.

## 4. v6 dataset (`docs/V6_DATASET.md`)
- 5 sounds x 20 curves x 60 s, same seeded curves for all sounds (seeds 3001..3020), ~80% v5-style random mix + ~20% designed probe events (touch from 0, touch from level 0.6, non-touch 0->0.9 steps with >= 2.5 s rest). Splits: val 004/011/017, test 007/014/020, rest train (70/15/15).
- Channels (aiff float, 48 kHz): ch0,1 master; ch2,3 isolated touch layer; ch4 sensor customSignal; ch5 touch gate; ch6 note frequency. Raw: `~/sentire/Recordings/Dataset_v6_20261122/` (8.3 GB). Converted npz: `~/sentire/Recordings/Dataset_v6_20261122_aligned/` (1.2 GB, 100 files, plus `bakeoff_notes_v6.json`, `bakeoff_onsets_v6.json`). 2 h recording, ran headless.
- Issues met and fixed: includes/identity compare in SC, odd curves not ending at 0, key-name mismatches between recorder and readers, prepare crash on silent takes, driver exit code, Monitor false alarms. One real defect: `03_harmonicSwarm_curve_005` (ch4 dropout 44.51-44.80 s) was quarantined and re-recorded. The resume run overwrote `events.csv`/`manifest.csv` with partial data; rebuilt from the 100 sidecars (backups `*.partial_bak`). The resume wrapper ended with shell exit 124 (timeout) after the file was written; the data were fine.
- Validator: final verdict GO after relaxing three checks that Opus classed as artifacts, not defects (F3 length -300..+50 ms plus >= 4.5 s tail: files up to 205 ms short, all keep >= 4.79 s past curve end; O1 offset 0.15..0.45 s: two files 0.236/0.273 from late resumes; L3 lead RMS SD 6 dB: modularPercussions idles noisily). I measured the length/tail distribution myself before widening.
- Opus read-only check of the npz: GO, no data defects (audio bit-identical to mean(ch0,ch1), sensor bit-identical to ch4[::480], touch_time onsets = crossings, 72/16/14 onsets per sound for train/val/test).

## 5. v9 trainer changes (`docs/V9_TRAINER.md`)
Loader (provenance whitelist, optional shift, deterministic-mode crash fix), model (`attack_lag_s` kwarg + `set_attack_lag`, full `onset_weighting` pass-through: v8c had silently run peak 350/pre 100 instead of its configured values), trainer/eval (val/test curves from the dataset instead of hard-coded 017-020, which would have crashed; lag re-applied after any checkpoint load; paths relative to the script; FFT f0 autocorrelation 27-45 s -> 0.01 s), render/demo paths, `config_sensor_v9.yaml` (from scratch, no finetune block).
**My planning error:** I planned to shift the audio by `offset_s` and Opus's plan review agreed. Task 5's measurement showed the v6 audio is already on the sensor clock (sensor = ch4 of the same aiff); shifting put the touch layer ~0.2 s early. Reversed (align_audio default False). Opus re-confirmed from raw aiffs.
Config values: `attack_lag_s` [0.023, 0.026, 0.065, 0.040, 0.025] (touch-layer lag 2/5/44/22(-> 40 after Opus frame analysis)/4 ms + 21.3 ms limiter), onset weighting pre 25 / peak 90 / window 400 ms, factor 5.

## 6. First v9 run (2000 steps, batch 8, from scratch, 13 min, `runs/bakeoff_ddsp_poc/v9_scratch/`)
- Valid run: no NaN, checkpoints finite, lag buffer = config, best checkpoint step 1750 (val 7.50), final 7.77, plateau since ~step 1000, no overfitting.
- Eval (val / test mean MS loss 12.01 / 11.94): RMS corr 0.88, centroid corr -0.27, RMS ratio 0.64; zeroing the inputs roughly doubles the loss (model uses the sensor).
- **The percussive attack is NOT learned for any sound.** Target: near silence then a sharp rise to ~-6 dB at +30..80 ms and a sustain ~10 dB lower by +300 ms. Model: rises 80-100 ms BEFORE the onset with the approaching hand and reaches a flat plateau with no peak. Peak in the first 300 ms: 8-13 dB below the target; rise from baseline ~10 dB vs 24-43 dB; model peak at 138-225 ms vs target 27-80 ms. Timing numbers look fine only because the plateau is reached at about the right time.
- Eval-script issues: "Loss Ratio" invalid (silence loss hard-coded 23.216 at line 614), per-sound ablation table in STAGE_B_REPORT.md all zeros (real values in metrics.json), "touch recomputation" warnings are false alarms (compares rolled vs unrolled arrays).

## 7. Open findings and candidate causes (nothing applied)
- D4: canopy-style additive vibrato `clamp(0.5*t, 20)` applies to all 5 sounds.
- D6: validation loss / best checkpoint use only the first 10 s of each val file (1-2 of 4-6 onsets) so checkpoint selection is weak evidence.
- Training schedule: the YAML `train:` block is ignored; LR is warm-up then constant 1e-3, 2000 steps.
- Loss: onset weighting 5x over ~100 ms may be weak against a 20-40 dB transient; `frame_weights` "sustained touch > 2 s" fires after ~69 ms (channel 0 is log1p(t)).
- Other: make_onset_demo does not pass `max_f0_octaves`; test_onset_weighting test 1 is flaky (also at f9e78d0); test_v8c needs a checkpoint not in git.
- Untested hypothesis: the fast-attack path (fast_head with gate after lag) can't create a peak, or is outweighed by the sensor path that already rises before the touch.

## 7b. Root-cause analysis (Opus, read-only, scratch scripts in `scratchpad/opus_rca/`)
Verified by Opus experiments (not independently re-run by a second reviewer yet):
1. **Pitch path broken (ranked #1 by Opus):** in v9 `state_final` the f0 deviation sits at its +/-3 clamp in 100% of frames (f0/note = 3.97; at init 1.05); the gradient through `clamp` is 0 so it cannot return, and nothing anchors it (`f0 = note * 2^(2*tanh(clamp(dev,+/-3)))`, max_f0_octaves 2.0). The touch layer is tonal and low (peak at note x1 or x2, nearly nothing above 4 kHz), so a harmonic stack at 4x note cannot produce it. Loss gradients at the onsets: the RMS term wants louder (+20..+400 ms), the 100x linear term wants quieter, the log term is noisy; the net is ~0, so nothing pushes toward the attack. Opus's own overfit networks also drifted (x3.0, x0.25).
2. **The architecture/synth/loss CAN represent the attack:** free 100 Hz controls through the same synth and `loss_curriculum`, with f0 = true note_hz, reproduce the attack within 250-500 steps (peaks within 1-3 dB, sustain within 0.4 dB). The same chunks through the real network fail (peaks 4-8 dB low, loss oscillates on a fixed batch). Caveat: that comparison changes two things at once (network bypassed AND pitch fixed); the clean test (real network with f0 anchored, nothing else changed) had not been run.
3. **fast_head (the dedicated transient path) is unused:** total-amp contribution < 0.5 dB, training only fast_head leaves it unchanged; it has no per-sound conditioning. Gate timing itself is correct (touch-layer onset in the master +20/20/60/38/24 ms after the onset frame for canopy/forestPad/swarm/percussion/moss; gate opens at o+2/2/6/3-4/2).
4. **Why train_loss rises:** the curriculum ramps the log-spectral weight over steps 150-400 together with an f0-regression term, which is 2.78 of 5.13 of the train loss at the end but carries almost no gradient (stuck f0). Bug: `mask = ~isnan(f0_t)` is computed after `nan_to_num`, so unvoiced target frames count as 100 Hz.
5. **Onset weighting weak and late:** 3.0 at onset, 5.0 only at +90 ms; onset windows are 6.4% of frames and 19% of the loss weight; a peak-flattened target costs only ~6% of the non-f0 loss. Secondary.
6. **The rise before the touch is mostly legitimate:** the target's base sound follows the sensor (about -41 dB at sensor 0.05-0.3 to -20 dB at 0.9) and starts 30-50 ms before the touch frame; the model runs 20-50 ms ahead. The real defect is the missing +10..+15 dB touch-layer bump over the base (canopy, swarm, moss; +3 forestPad, +7 percussion) at +20..80 ms. The earlier "24-43 dB rise" figure was measured from pre-touch silence and mostly reflects the base sound.
7. **Input finding:** `bakeoff_notes_v6.json` labels switch one frame late (at the onset frame the previous note is still held; matches note_hz in 37% of onsets, 81% at o+1); the npz `note_hz` switches exactly at the onset.
8. **Ruled out:** data exposure, attack predictability (level std 0.7-3.6 dB), lag values (master vs touch layer 1022-1024 samples = 21.3 ms), limiter clipping, noise branch, reverb (wet 0.005), gradient clipping.
Untested hypotheses: that the f0 drift is driven by the lin/log terms early in training, and that pitch error also causes the output being 2-6 dB too quiet and the negative centroid correlation.

**User pushback (2026-11-23):** a wrong pitch would only play a wrong note; the loudness jump at touch should still appear, so the attack is the focus and pitch is nice-to-have. Assessment: fair, because the causal claim about pitch was not isolated (see item 2 caveat). A clean A/B was started with the same Opus analyst: same fixed chunks and budget through the real network and real loss; arms: base; f0 anchored to note_hz; f0 untouched but loss rebalanced toward the attack (more RMS weight / lower lin / onset peak ~40 ms); both; f0 anchored + sound-conditioned fast_head; plus f0 frozen at init. Decisive question: does the attack appear without anchoring the pitch? **Result pending when this was written; the section will be updated when it arrives.**

Candidate fixes from the RCA (not applied), in the analyst's order: anchor pitch / fix the f0-regression term; use npz `note_hz` instead of the late JSON labels; sound-conditioned fast_head; onset-weight peak ~40 ms starting at the per-sound lag plus a higher RMS weight / lower lin; lower or decaying LR; whole-file onset-aware validation.

## 8. Process notes (things that went wrong)
- Plan error above (audio shift). Watcher bug: my wait loop used `pgrep -f` that matched itself, so I missed that the run had finished 30 minutes earlier. Monitor false alarm during recording (benign OSError lines). Haiku writer reports were occasionally optimistic (e.g. truncated failure lists in the validator), so every diff and claim was re-checked.
- Opus gates: plan reviews (found the two blockers: hard-coded val curves, dropped onset-weight keys), data gate (GO), trainer gate (conditional GO, 7 defects), results gate (valid run, attack not learned).

## 9. Delegation log
| # | Task | Agent | Model | Outcome |
|---|---|---|---|---|
| audit 1-2 | structure and alignment audit | Haiku | claude-haiku-5-5 | corrected by Opus, merged |
| sync 1-2 | TouchSync recorder and analyzer | Haiku | claude-haiku-5-5 | merged |
| v6 1-5 | recorder, prepare, validator, fixes | Haiku | claude-haiku-5-5 | merged on orch/v6data |
| v6 6 | validator thresholds | Haiku | claude-haiku-5-5 | commit 76f6c2f |
| v9 1-5 | loader, model, trainer+config, render, lag | Haiku | claude-haiku-5-5 | merged on orch/v9train |
| gates | plans, data, trainer, training results | Opus | claude-opus-5-5 | see sections 4-6 |
| leader | orchestration, recording runs, checks | Sonnet | claude-sonnet-5-5 | |
