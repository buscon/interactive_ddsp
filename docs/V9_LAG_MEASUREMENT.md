# V9 onset lag measurement on Dataset v6

Branch `orch/v9train-5-lag` (Task 5). Script: `analyze_v6_lag.py`. Raw output: `v6_lag_measurement.json`.
Data: `bakeoff_data_aligned_v6/*.npz`, 100 files (train 70, val 15, test 15), 20 files per sound, 510 touch onsets.

## Headline

1. **The brief's alignment does not hold for these arrays.** With `audio_aligned[n] = audio[n + round(offset_s*48000)]`, no onset passes the quiet check (0 of 510 usable). The shifted touch layer sits about 390 ms before the touch onset, which is physically impossible for a touch-triggered sound. Without the shift (raw arrays as stored) the touch-layer lags match the brief's reference values for all five sounds (table B). **This needs Opus verification (AGENTS.md results gate) before any training change.**
2. **Touch-layer onset lags reproduce the brief's reference values** in the unshifted timeline (within 2.5 ms for every sound).
3. **Audible onset lag on the audio target is not measurable with this method.** Medians are negative, clamped at the search-window edge and move by up to 35 ms between thresholds. The `attack_lag_s` vector from the requested rule is therefore not recommended.
4. **Peak lag is threshold-independent and is the usable metric.** Median over sounds of the per-sound median audio peak lag is 88.5 ms, so `peak_ms = 90`. The modPerc value (307 ms) is an outlier that dominates per-sound differences.

## Method

- **Timeline modes.** `aligned`: array[n] = raw[n + round(offset_s*48000)], as in the brief. `unshifted`: raw arrays as stored. Both modes run in one script invocation.
- **Touch onset.** First `touch_time > 0` frame i after a zero. Time t = (i-1)*0.01 s, following the brief. The gate edge in the raw aiff (ch5) falls at 16628.6 ms for the first onset of `01_canopy_curve_001`, between the (i-1) convention (16620 ms) and the i convention (16630 ms). Switching to i*0.01 would subtract 10 ms from every lag reported here.
- **Envelope.** 4th-order Butterworth high-pass at 200 Hz (sosfiltfilt), 2 ms RMS window, 1 ms hop, in dB. Times are window centres relative to the touch onset.
- **Onset lag.** First envelope frame in [-50, +600] ms above max(noise + X dB, -70 dBFS). Noise = median dB over the preceding 300 ms. X = 6, 12 (primary), 18.
- **Peak lag.** Envelope maximum in [0, +600] ms.
- **Decay.** First frame after the peak at peak - 10 dB, searched to +2000 ms. Reported both as absolute time and as time after the peak.
- **Quiet rule.** Skip an onset if any `touch_time > 0` in the preceding 300 ms, or if the maximum touch-layer envelope in the preceding 300 ms exceeds -50 dBFS. The decision is taken on touch_layer, because the audio target is not silent before onsets in every case (see caveats), and the same onset set is used for both layers.

## A. Unshifted timeline (primary result), per sound, X = 12 dB

Onsets: 102 per sound. Used: 90 per sound (12 per sound skipped as not quiet, 0 edge). Times in ms.

**Audio target**

| Sound | n | Not detected | Onset median | Onset IQR (q25..q75) | Onset min..max | Peak median | Peak IQR | Peak min..max | Decay median (abs) | Decay after peak | Censored decay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_canopy | 90 | 16 | -20.0 | -22.8..-13.0 | -35..142 | 66.0 | 37.0..203.0 | 29..577 | 111.5 | 5.0 | 0 |
| 02_forestPad | 90 | 6 | -34.5 | -50.0..1.5 | -50..84 | 57.0 | 50.0..85.5 | 31..228 | 73.0 | 13.0 | 3 |
| 03_harmonicSwarm | 90 | 0 | -23.0 | -24.0..49.5 | -35..77 | 88.5 | 78.0..98.8 | 69..158 | 103.0 | 12.0 | 0 |
| 04_modularPercussions | 90 | 2 | -14.0 | -45.0..10.5 | -50..590 | 307.0 | 212.5..466.0 | 0..600 | 316.0 | 5.0 | 0 |
| 05_moss_and_mist | 90 | 0 | -50.0 | -50.0..33.0 | -50..48 | 89.5 | 69.3..133.5 | 44..320 | 146.0 | 39.0 | 0 |

**Touch layer (reference)**

| Sound | n | Onset median | Onset IQR | Onset min..max | Peak median | Peak IQR | Decay median (abs) | Decay after peak |
|---|---|---|---|---|---|---|---|---|
| 01_canopy | 90 | 2.0 | 0.0..3.0 | -6..9 | 19.0 | 16.0..22.8 | 76.5 | 57.5 |
| 02_forestPad | 90 | 5.0 | 3.0..7.0 | -3..12 | 44.0 | 29.0..64.5 | 56.0 | 9.0 |
| 03_harmonicSwarm | 90 | 44.0 | 43.0..45.0 | 36..49 | 64.5 | 56.0..71.8 | 77.0 | 13.0 |
| 04_modularPercussions | 90 | 22.0 | 20.0..24.0 | 15..27 | 34.0 | 31.3..37.0 | 41.5 | 5.5 |
| 05_moss_and_mist | 90 | 4.0 | 2.0..6.0 | -3..8 | 91.5 | 63.5..132.8 | 135.0 | 32.0 |

## B. Touch-layer onset lag versus the brief's reference values

| Sound | Measured median (ms) | Brief reference (ms) | Difference (ms) |
|---|---|---|---|
| 01_canopy | 2.0 | 3.0 | -1.0 |
| 02_forestPad | 5.0 | 2.5 | +2.5 |
| 03_harmonicSwarm | 44.0 | 43 | +1.0 |
| 04_modularPercussions | 22.0 | 21 | +1.0 |
| 05_moss_and_mist | 4.0 | 2.3 | +1.7 |

## C. Threshold sensitivity (unshifted, audio target)

| Sound | Median onset, +6 dB | +12 dB (primary) | +18 dB | Not detected at +6 / +12 / +18 | Peak median (all thresholds) |
|---|---|---|---|---|---|
| 01_canopy | -18.0 | -20.0 | -20.0 | 0 / 16 / 23 | 66.0 |
| 02_forestPad | -40.5 | -34.5 | -48.0 | 0 / 6 / 23 | 57.0 |
| 03_harmonicSwarm | -23.0 | -23.0 | -23.0 | 0 / 0 / 0 | 88.5 |
| 04_modularPercussions | -42.0 | -14.0 | -13.0 | 0 / 2 / 15 | 307.0 |
| 05_moss_and_mist | -50.0 | -50.0 | -50.0 | 0 / 0 / 0 | 89.5 |

Touch-layer onset medians are unchanged across thresholds (the -70 dBFS floor dominates on these near-silent pre-touch layers). The audio onset medians are not stable, so they are not used for recommendations.

## D. Aligned timeline (brief's shift)

0 of 510 onsets pass the quiet check. Every onset is rejected because the shifted touch-layer activity lies inside the preceding 300 ms. Checked on `01_canopy_proximity_curve_001`: the touch-layer loud event starts at 16232 ms, while the touch onset is at 16620 ms (-388 ms). The same offset appears on all four onsets in that file. In the raw aiff, the gate (ch5) rises at 16628.6 ms and the unshifted touch layer starts at about 16632 ms, which matches touch_time. The raw audio is about 21 ms behind the touch layer, consistent with the limiter delay in the brief. The shift therefore moves the audio by 0.4 s in the wrong direction for these arrays. This is the main open issue.

## E. Recommendation

**attack_lag_s.** The requested rule (median audio onset lag, rounded to ms) gives [-0.020, -0.034, -0.023, -0.014, -0.050] s. **Not recommended**: the values are negative, clamped at the search edge and threshold-unstable. Candidate alternative, measured on the touch layer (median onset lag, unshifted): [0.002, 0.005, 0.044, 0.022, 0.004] s. For the master target, add the brief's ~21.3 ms limiter delay: [0.023, 0.026, 0.065, 0.043, 0.025] s. The 21 ms master delay was checked only on the first file of four sounds by cross-correlation (21 ms in four of five; modPerc gave 598 ms, which looks like an artifact and is unverified).

**Onset-weight kernel (v9):**

| Parameter | Value | Basis |
|---|---|---|
| pre_ms | 25 | Midpoint of 20-30 ms as requested. The data do not support a specific value: audio onset lags reach -50 ms (P5 -50 ms). |
| peak_ms | 90 | Median over sounds of per-sound median audio peak lag = 88.5 ms, rounded to 10 ms. |
| window_ms | 400 | As requested. |

**Per-sound peak lag versus peak_ms = 90:**

| Sound | Measured peak median (ms) | Difference from 90 (ms) |
|---|---|---|
| 01_canopy | 66.0 | -24.0 |
| 02_forestPad | 57.0 | -33.0 |
| 03_harmonicSwarm | 88.5 | -1.5 |
| 04_modularPercussions | 307.0 | +217.0 |
| 05_moss_and_mist | 89.5 | -0.5 |

Across all 450 used onsets the peak median is 92 ms (IQR 59..179 ms).

## Caveats

- **Alignment (critical).** Section D. Until the alignment question is settled, the aligned target results are not meaningful.
- **Audio onset measurement is unreliable.** The -70 dBFS floor, digital silence before some onsets (for example moss onsets 5250 and 5570, which are silent for 300 ms before touch), and pre-touch audio starting 50-70 ms early on moss (audio is about -84 dB at -70 ms; touch layer begins at 0 ms) all bias the onset lag negative. I did not resolve whether this is a real pre-touch sound or a sensor/audio timing offset.
- **Peak lag mixes the touch event with sustained content.** For modPerc the envelope maximum typically falls 200-470 ms after touch, so the 600 ms window captures build-up or other material. Peak lag is threshold-independent but not necessarily the touch-triggered transient.
- **Onset convention.** Results use t = (i-1)*0.01 s. The gate edge sits closer to i*0.01 s, so lags could be 10 ms smaller. The touch-layer agreement with the brief's values favours (i-1).
- **Resolution.** 1 ms hop, window-centre times; the touch-layer medians are accurate to about 1 ms, not the sample level.
- **Skips.** 12 per sound (60 of 510) were skipped as not quiet, all for a touch within 300 ms. None were skipped for edge reasons.
- **Verification.** These results have not been checked by Opus (AGENTS.md results gate). The alignment finding in particular contradicts the brief and must be reviewed before any change to the training alignment.
