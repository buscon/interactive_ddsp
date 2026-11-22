# V6 Attack Degradation: Claude Review

**Date:** 2026-11-22
**Reviews:** `docs/trackA_v6_claude_brief.md`
**Code checked:** `engine_ddsp/sensor_ddsp.py` (`attack_loss`, `multiscale_loss` frame weights)

## Verdict

Partly agree with the Qwen diagnosis. The brief skips a check that should come first, and the recommended fix would inherit a bug already in the code.

## Findings from the code

1. **`attack_loss` mostly never sees the onsets** (`sensor_ddsp.py:609-613`). It downsamples the 1-frame onset impulse by picking one touch frame per STFT frame (`indices = arange*T//T_stft`). Each STFT frame spans 5 touch frames, so about 80% of onsets fall between the sampled indices and are dropped. The loss therefore weights the wrong frames or none at all. This explains "no gain" better than gradient conflict does. The `w_attack=1` result shows this implementation fails, not that attack losses fail.

2. **Qwen's proposed fix would repeat the bug.** Its `downsample(touch_onset)` is unspecified. If it is the same strided pick, it fails the same way. The snippet also has shape problems: `.diff()` shortens the tensor by one frame, and `[..., :1]` leaves a trailing dim. The model already derives `touch_onset` in `forward` (lines 162-169), so the snippet duplicates existing logic.

3. **Qwen's claims, assessed.**
   - *Temporal mismatch:* partly right. The rates differ (100 Hz vs 20 Hz), but this is a resampling bug, not a conceptual mismatch.
   - *Gradient conflict:* unsupported guess. The attack loss is linear magnitude only. Its weight ramps in over 500 steps and it is a mean over a mostly silent spectrogram, so it is probably tiny relative to `100*lin_loss`. A tiny loss gives "no effect", not "conflict".
   - *Energy ≠ sharpness:* valid. A 200 ms window (9600 samples) cannot resolve a 20-50 ms rise. The code comment says "50% overlap" but the overlap is 0.75.

4. **The multiscale loss already weights frames by sustained touch.** `frame_weights` gives 1.5x to frames with touch_time > 2 s and nothing to onsets. This may explain why V6 improved continuous modulation while attacks got worse.

## Gaps in the brief

- **No metric measures attacks.** The decision rests on listening plus whole-file RMS correlation. Add an onset-window metric: log-STFT distance and envelope rise time over the 0-150 ms after each onset, pred vs target.
- **V4c vs V6 comparability is unconfirmed.** Chunking (10 s vs 30 s) and the onset definition differ, and V4c's multiscale loss is better (9.40 vs 10.36). Confirm both were evaluated on identical audio and windows.
- **Sensor/audio alignment is unverified.** If the audio onset precedes or lags the 0.3 sensor crossing, "exact" onsets are exactly wrong and the model learns to smooth over the mismatch. This is the best candidate explanation for the paradox. Plot target RMS against the sensor around about 20 onsets.
- **Noise level is unknown.** There are 20 samples and one seed, so the 0.74 to 0.63 and 0.65 swings may be partly noise. Use 2-3 seeds or bootstrap the held-out set.
- **Mixed ablation.** `w_mod_spec` targets sustained modulation, so it is not a transient fix. Drop it from this question.

## Recommended experiments (in order, stop when one works)

1. **Diagnostics only, no training.**
   - Check onset alignment as described above.
   - Compute the onset-window metric on existing v4c and v6 outputs.
   - Count how many onsets the current `attack_loss` actually hits.
2. **Fix the downsampling and retry the attack loss.**
   - Max-pool the onset mask over each hop instead of striding.
   - Use a log-magnitude term with a short STFT (win 512-1024) instead of the 200 ms window.
   - Keep `w_attack` at 0.25-0.5 and drop the ramp.
3. **Onset-weighted multiscale** (Qwen's idea, after step 2).
   - Multiply the existing `frame_weights` by `1 + k*onset_mask`.
   - Dilate the mask over about 100 ms, with correct pooling.
   - Sweep k in {3, 5}.
4. **Only if 1-3 fail:** boost transients in the targets, or add a separate transient pathway.

## Success criteria

- Onset-window log-STFT distance at least 15% better than v6 baseline.
- Rise-time error within 10 ms.
- Overall multiscale loss no worse than 10.4.
- RMS correlation at least 0.74.
- Continuous-modulation metric and noise floor unchanged.
- Blind A/B listening on 10 onsets: the fixed model is preferred over v6 and over v4c.

## Timeline

Diagnostics take hours. Each training run must pass the earlier collapse point (step 1750), so run 3k-5k steps per config with 2 seeds.
