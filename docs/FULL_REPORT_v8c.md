# Full Report — Touch-Event Restoration in Sensor DDSP (V6 → V8c)

**Date:** 2026-11-22 · **Branch:** `orch/v8c` @ `e783d81` · **Author:** project leader, with Claude Opus 5.5 reviews (mandatory gate, see `AGENTS.md`)

---

## 1. Goal

The shipped V4c model had "soft attack" and no continuous touch modulation. **V6** improved modulation + noise floor but made touch events "almost not perceptible". The goal of this whole effort: **make the touch event a clear, short, percussive attack** (as in the target recordings), while keeping V6's modulation/noise-floor gains. Sustained-touch ramp behavior is a *separate* parked issue.

Ground truth about the sounds (user + SC sources): a touch fires a short percussive "drop" (Env.perc(0.01, 0.3–0.5)) at a **random scale note**; staying-in-touch modulation (LFO rate/depth) ramps with touch time and becomes audible only after ~1.5–3 s. Measured: the audible attack energy lands **+0.24–0.44 s after the 0.98 sensor crossing** (per-sound lag D), +10..+20 dB step.

---

## 2. What was done (chronological)

| Phase | What | Result |
|-------|------|--------|
| V6 ablation | attack/modspec losses ON → all degraded (rms_corr 0.74→0.63); ship V6 baseline | root cause = loss conflict |
| V7 | onset-weighted multiscale loss (k=5, 150–500 ms) | spectral metrics best (9.49/0.817) but onset RMS ratio *worse* (1.18x vs control 1.60x), level collapsed (rms_ratio 0.31) |
| Alignment audit | sensor onsets vs audio flux | REBUILD verdict — but wrong approach (this is continuous sound, not percussive) |
| Onset source fix | OSC touch events are NOT logged separately; touch = **sensor ≥ 0.98 crossing** (thresholdTouch in SC). `bakeoff_notes_v7.json`: 230 onsets | correct labels |
| V8b | data rebuilt at threshold **0.98** (`bakeoff_data_aligned_v8`, 230/230 onset match PASS) + code fixes per Opus: causal gamma onset kernel (peak +350 ms, max-normalized → w reaches k=5), max-pool to all STFT scales, RMS term weighted too, fast-decay head bypass (0.03/0.1/0.3 s, zero-init, ckpt-compatible), eval threshold from metadata | val_loss 6.55 (best so far); demo flux 561 vs target 734 (v6 243, v7 42) but peak still 0.17 vs 0.55 |
| V8c arm A | + loss freq-normalization fix (was ~100x sum over bins), SC-formula lagged vibrato (depth=0.5t, rate=1+clip(0.3t)), fast_head delayed by attack lag, fast_head 10x LR | val 6.72; rms_ratio 0.78 (collapse fixed); rms_corr 0.875; attack steps up on 4/5 sounds |
| V8c arm B | arm A + held per-touch pitch labels (`bakeoff_notes_v8c.json`, f0 estimated at attack, snapped to minor pentatonic) | val 6.79; multiscale 9.56, rms_corr 0.887 (best); helps canopy/moss attack, hurts forestPad/modPerc (single pitch track pulled off pad pitch) |

---

## 3. Achieved ✅

- **Correct onset labels & data alignment**: touch = 0.98 crossing; `bakeoff_data_aligned_v8` (0.98) with 230/230 onset match; per-sound attack lag measured (canopy 0.38, forestPad 0.38, swarm 0.42, modPerc 0.245, moss 0.39 s).
- **Real bugs found & fixed** (via Opus gate): backwards onset window, dropped onsets at 4096 scale, RMS-mask crash (V8 never trained), loss 100x freq-sum bug, frame_weights axis bug, decay-feature gating, frame-0 false onsets, render indentation, checkpoint raw-state_dict loading.
- **Level/envelope recovery**: rms_ratio 0.31 (v7) → 0.78 (v8c_a); rms_envelope_corr 0.887 (best ever); multiscale 9.56 (best ever).
- **Attack strength improved on 4/5 sounds** in demo clips (forestPad +6/+6.9 dB vs target +8.2/+4.4; moss +6.1/+7.5; swarm ≈ target; modPerc +2.7/+3.6).
- **Reproducible pipeline**: branches, configs, tests (`test_onset_weighting_v8b.py`, `test_v8c.py` ALL PASS), demo generator (`make_onset_demo.py`: 3 events, 3 s gaps).

## 4. Not achieved ❌

- **Canopy attack still weak**: −1.8..+2.3 dB vs target +12/+15 (Opus: the synth has a single pitch track; the attack "drop" is at the random note pitch while the pad keeps its pitch — arm A can't produce it, arm B pulls the pad off pitch).
- **Success criterion not met**: ≥ +6 dB step on ≥2/3 events — only some forestPad/moss events pass (and Opus's stricter recompute says even fewer; ev2 is unusable due to silent pre).
- **Per-sound vibrato bug**: canopy's FM formula is applied to all 5 sounds (forestPad/moss/percussion should have none) — correctness bug, not yet fixed.
- **Eval metric bugs** (Opus): −1.0 placeholders averaged into means; onset window starts before the +0.38 s attack; pitch search capped at 500 Hz (swarm label ~600 Hz); `np.resize` wrap bug; centroid scored on near-silent frames; old runs scored on 0.3-mask data (and `bakeoff_data_aligned_v5` was accidentally overwritten with 0.98 data — restored a 0.3 copy at `bakeoff_data_aligned_v5_03`).
- Arm B's held-pitch helps canopy/moss but hurts forestPad/modPerc (single-pitch limit) and worsens onset_f0_err metric (partly metric artifact).

---

## 5. My review (project leader)

The biggest wins were **data/label alignment** (0.98 threshold) and **loss normalization** — both found only because Opus caught them. The remaining blocker for a *perceptible* attack is architectural: **one pitch track can't play pad + percussive drop at once**, and the 100 Hz heads get only a smeared envelope. The demo-clip protocol (3 events, 3 s gaps) made the residual problem measurable and listenable. Metrics alone repeatedly misled (level collapse, negative sustained corr artifact) — listening + event-locked measurement must stay primary.

## 6. Opus review (results gate, session ses_dfc3788b6ffexm6MpW8nvNphHV)

Verbatim key findings:

1. Metric means arithmetic correct, but **−1.0 placeholders are averaged in** (sustained_touch_corr −0.24 is an artifact; on the old 0.3 mask v8c arms score 0.48/0.43 vs v7c 0.39). Old runs used 0.3-mask data; v5 data had been overwritten with 0.98 — cross-run comparisons of sustained/mod_depth/onset_f0 are invalid.
2. Arm A beats B on val (6.7185 vs 6.788); B's val unstable (spikes 8.78 @1500).
3. Demo steps: Opus's recompute is 0.5–1 dB lower; **canopy v7_control (+4.1/+3.9) beats every v8 model** on canopy.
4. sustained_touch_corr negative = artifact (0.98 mask scores ~0 for all models). centroid drop mostly artifact (loud-frames-only: v8c_a best 0.51); real problem is swarm (−0.09→−0.33) from the all-sounds-vibrato bug.
5. Arm B helps canopy (onset pitch err 1109→615 cents) and moss, hurts forestPad (5.1→1.9) and modPerc — single-pitch limit.
6. **Verdict: v8c does NOT reliably meet "strong attack"**; canopy 9–13 dB short; criterion should be relative to target's step.

**Opus's prescribed next change:** add a **second, touch-triggered "attack voice"** — a sine (few harmonics) at the held note pitch whose loudness follows the delayed fast-attack envelope, alongside the pad voice (keeps arm A note input). Plus per-sound vibrato gate (correctness fix). Before any new run: commit indexing fix (done, e783d81), restore 0.3 v5 data (done, v5_03), fix eval script (skip −1 placeholders, onset window + lag, pitch limit 1 kHz, loud-frames centroid, fix np.resize), re-evaluate all arms on one setup.

---

## 7. Artifacts

- Branches: `orch/v6-ablation` (69f68a4), `orch/v7-onsetw`, `orch/v7-fixes`, `orch/v8b-eval`, **`orch/v8c` @ e783d81**
- Data: `bakeoff_data_aligned_v8/` (0.98, current), `bakeoff_data_aligned_v5_03/` (restored 0.3), notes: `bakeoff_notes_v7.json` (onsets), `bakeoff_notes_v8c.json` (held pitch)
- Runs: `runs/bakeoff_ddsp_poc/stageB_v6|v7|v7_control|v8b|v8b_control|v8c_a|v8c_b/`
- Demos: `listening_demo/<sound>_*.wav` (3 events, 3 s gaps, per model + target)
- Docs: `docs/v7_plan.md`, `docs/v8_conceptual_issue_brief.md`, `docs/v8c_brief_for_opus.md`, `docs/v8c_results_brief.md`, this report
- Rules added: `AGENTS.md` (mandatory Opus gate for plans+results), LogSeq `Rules___Interactive_Ddsp.md`, model rules (Mistral for coding/review, SAIA out of tokens)
