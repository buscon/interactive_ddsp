# V9 f0 Freeze Postmortem

**Date:** 2026-11-23  
**Branch:** `orch/v9data-fix`  
**Status:** ❌ Reverted — made audio quality worse

---

## Executive Summary

Attempted to fix broken pitch (f0) by freezing f0_head at zero. The fix **corrected pitch** but **broke dynamics and modulation**. User listening test confirmed regression. Reverted to previous state.

| Aspect | Before Freeze | After Freeze | Verdict |
|--------|---------------|--------------|---------|
| f0 (pitch) | Broken (base freq) | ✓ Fixed (follows note) | ✅ Improvement |
| Dynamics | ✓ Good (touch_amp_head) | ❌ Squashed | ❌ Regression |
| Swarm modulation | ✓ Clean attack | ❌ Modulation artifact | ❌ Regression |
| Overall quality | Better | Worse | ❌ **Reverted** |

---

## What We Tried

### The f0 Problem

**Symptom:** All sounds output constant base frequency instead of following note curve:

| Sound | Target f0 | Pred f0 | Error |
|-------|-----------|---------|-------|
| swarm | 247 Hz | 101 Hz | -146 Hz (2 octaves!) |
| canopy | ~180 Hz | 110 Hz | -70 Hz |

**Root cause:** f0_head receives raw Hz input (100-400 Hz). Small weights produce huge outputs:
```
weight_note = -0.039 × note ≈ 250 Hz = -9.75
f0_deviation → -∞, tanh → -1
f0 = note × 2^(2×-1) = note × 0.25 (2 octaves down)
```

### The Fix Attempt

**Code change** (`engine_ddsp/sensor_ddsp.py` line 307):
```python
# BEFORE (broken):
f0_deviation = self.f0_head(torch.cat([mlp_out, touch_onset, touch, note], dim=-1))

# AFTER (frozen):
f0_deviation = torch.zeros_like(note)  # f0 = note_f0 exactly
```

**Commit:** `5526c4c` — "fix: freeze f0_head at zero (f0 = note_f0 exactly)"

### Training & Evaluation

**Run:** `v9_stageB_f0frozen_s0` (seed 0, 2000 steps)

**Metrics:**
| Metric | Before (broken f0) | After (f0 frozen) | Change |
|--------|-------------------|-------------------|--------|
| f0 accuracy | 25% of note | 100% of note | ✅ Fixed |
| dtpk (median) | 155 ms | 98 ms | ✅ 1.6× better |
| dpk (median) | -10.9 dB | -8.1 dB | ✅ 2.8 dB better |
| strict_pass | 0/5 | 0/5 | ❌ Still failing |

**Pitch verification:**
```
03_harmonicSwarm_001:
  Before: pred f0 = 101 Hz (stuck at base)
  After:  pred f0 = 150-178 Hz (follows note data)
  Target: f0 = 247 Hz (from audio)
```

**Note:** Note data doesn't match target audio (~5 semitone difference), but model now correctly follows note.

---

## Listening Test Results

**Location:** `listening_test_v9_f0frozen/`

**User feedback:**
> "the files go worst, the dynamic problem is back (very squashed) and the modulation problem of 03 is back to."

### What Went Wrong

**Hypothesis:** Freezing f0_head removed an unintended side effect.

The f0_head input includes `mlp_out` (512 dims from GRU). Even though the head was trained for pitch, the **mlp_out pathway may have been contributing to other outputs** through:

1. **Shared gradients** — f0_head gradients affected mlp_out during training
2. **Feature interaction** — mlp_out encodes more than pitch; freezing removed this signal
3. **Unintended coupling** — the broken f0_head was accidentally helping dynamics/modulation

**Evidence:**
- Dynamics were good with `touch_amp_head: true` in previous runs
- After freeze, dynamics regressed to "squashed" state
- Swarm modulation artifact (fixed by `lfo_depth_mul_base: [0,0,0,0]`) returned

---

## Git History

```
2cc073b revert: f0_head freeze (made dynamics and modulation worse)
5526c4c fix: freeze f0_head at zero (f0 = note_f0 exactly)
4f53111 docs: V9 Stage B final report
45a61ce fix: f0_head now uses note input; add fixed config
```

**Current state:** Reverted to `45a61ce` + docs (f0_head connected but broken)

---

## Lessons Learned

### 1. Coupled Outputs Are Dangerous

The f0_head wasn't isolated — it affected dynamics and modulation. This suggests:
- **Shared representation** in mlp_out/gru_out
- **No output isolation** — changing one head affects others
- **Training interdependence** — heads learn together, can't freeze one without side effects

### 2. Metrics Don't Capture Everything

Metrics improved (dtpk, dpk), but audio quality got worse. This means:
- **Attack metrics don't measure dynamics** — squashed audio can still have good timing
- **Need perceptual metrics** — crest factor, dynamic range compression detection
- **Listening tests are essential** — can't rely on metrics alone

### 3. The Real Problem: Note Input Scale

The root cause is **raw Hz input to f0_head**:

```python
# Current (broken):
note: 100-400 Hz (raw frequency)
f0_head: Linear(517, 1) → outputs -6.4 (wrong!)

# Should be (normalized):
note_log2_ratio: log2(note / f0_base) ≈ -1 to +2 (normalized)
f0_head: Linear(517, 1) → outputs ~0 (correct!)
```

**Why this matters:**
- Raw Hz: small weight changes → huge output swings
- Normalized: weight changes → proportional output changes
- Zero initialization works with normalized, not raw

---

## Next Steps (Proper Fix)

### Option A: Normalize Note Input (Recommended)

**Change:** Feed log2 ratio instead of raw Hz

```python
# In sensor_ddsp.py forward():
# Compute normalized note input
f0_base_sound = self.f0_base[sound_id]  # (B,)
note_f0 = torch.where(note > 0, note, f0_base_sound.unsqueeze(1).unsqueeze(2))
note_log2_ratio = torch.log2(note_f0 / f0_base_sound.unsqueeze(1).unsqueeze(2) + 1e-8)

# f0_head input
f0_head_in = torch.cat([mlp_out, touch_onset, touch, note_log2_ratio], dim=-1)
f0_deviation = self.f0_head(f0_head_in)

# Initialize f0_head to output ~0
# (weight initialization: small values, bias = 0)
```

**Benefits:**
- f0_head learns correct mapping (deviation → 0 for f0 = note_f0)
- No side effects on dynamics/modulation
- Can add f0 loss later for expressive pitch

### Option B: Add f0 Loss

**Change:** Supervise pitch directly

```python
# In training loop:
f0_target = note_f0  # or extract from audio with CREPE/pYIN
f0_loss = (f0_predicted - f0_target).abs().mean()
total_loss += w_f0 * f0_loss  # w_f0 ≈ 0.1-1.0
```

**Benefits:**
- Direct pitch supervision
- Model learns correct f0 mapping
- Can learn expressive variation (not just frozen)

**Costs:**
- Code change in training script
- May need to re-balance loss weights

### Option C: Isolate f0 Pathway

**Change:** Separate mlp/gru for f0 only

```python
# Dedicated f0 pathway (no shared mlp_out)
f0_gru = nn.GRU(..., num_layers=2)
f0_head = nn.Linear(f0_gru_hidden, 1)
# Don't share weights with harmonic/noise heads
```

**Benefits:**
- Complete isolation — freezing f0 won't affect others
- Clean architecture

**Costs:**
- Major code refactor
- More parameters, slower training

---

## Recommended Path Forward

1. **Short term:** Keep current state (f0_head connected, broken pitch)
   - Dynamics and modulation are good
   - Pitch is wrong but consistent (can be fixed in post?)

2. **Medium term:** Implement Option A (normalize note input)
   - Small code change (2-3 lines)
   - Re-train Stage B (2000 steps, ~30 min)
   - Should fix pitch without side effects

3. **Long term:** Add f0 loss (Option B)
   - Enable expressive pitch variation
   - Better match to target audio

---

## Files Changed

| File | Change | Status |
|------|--------|--------|
| `engine_ddsp/sensor_ddsp.py` | f0_head freeze → revert | ✅ Reverted |
| `docs/V9_STAGEB_FINAL_REPORT.md` | Comprehensive report | ✅ Kept |
| `docs/V9_F0_FREEZE_POSTMORTEM.md` | This postmortem | ✅ New |
| `listening_test_v9_f0frozen/` | Listening test files | ✅ Archive |

---

## Appendix: Commands

### Re-train with normalized note input (future)

```bash
# After implementing note normalization in sensor_ddsp.py
python3 train_sensor_ddsp_v5.py \
  --config config_sensor_v9_stageB_fixed.yaml \
  --name v9_stageB_note_normalized_s0 \
  --root /home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed \
  --steps 2000 --seed 0
```

### Evaluate

```bash
python3 eval_stageB_v5.py \
  --run /home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/v9_stageB_note_normalized_s0 \
  --ckpt state_final.pth \
  --out /home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/runs/v9_stageB_fixed/eval_normalized_s0 \
  --split val
```

---

**Report written:** 2026-11-23  
**Author:** ML audio engineering session  
**Status:** Reverted, awaiting proper fix (normalized note input)
