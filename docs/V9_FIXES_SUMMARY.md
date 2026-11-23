# V9 Stage B — Required Fixes Summary

**Date:** 2026-11-23  
**Status:** Three critical bugs identified and fixed

---

## Issues Identified

### Issue 1: Swarm Modulation Starts at Frame 1 ✓ FIXED
**Symptom:** 03_harmonicSwarm sounds like it's already in "sustained" mode at attack.

**Root cause:** `lfo_depth_mul_base[2] = 0.3` — 30% modulation from frame 1.

**Fix:** Set `lfo_depth_mul_base: [0.0, 0.0, 0.0, 0.0]`

**Status:** ✅ Inference-time fix confirmed working

---

### Issue 2: Compressed Dynamics ✓ TO BE FIXED
**Symptom:** Pred files have ~2× less dynamic range than target.

**Root cause:** `touch_amp_head: false` — no direct touch→amplitude pathway.

**Fix:** 
- `touch_amp_head: true`
- `w_attack: 0.1` (trains the head)

**Status:** ⏳ Requires re-training

---

### Issue 3: f0 Head Doesn't Use Note Input ✓ FIXED
**Symptom:** Pred f0 stuck at base frequency (100 Hz for swarm), target varies 80-300 Hz.

**Root cause:** `f0_head` input was `[mlp_out, touch_onset, touch]` — **note was missing!**

**Fix:** 
- Change `f0_head` input dim: `hidden_size + 1 + 3` → `hidden_size + 1 + 3 + 1`
- Add `note` to f0_head input: `torch.cat([mlp_out, touch_onset, touch, note], dim=-1)`

**Status:** ⏳ Requires re-training

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `engine_ddsp/sensor_ddsp.py` | f0_head input dim +1 | 164 |
| `engine_ddsp/sensor_ddsp.py` | f0_head forward: add note | 307 |
| `config_sensor_v9_stageB.yaml` | lfo_depth_mul_base: [0,0,0,0] | 27 |
| `config_sensor_v9_stageB.yaml` | touch_amp_head: true | 43 |
| `config_sensor_v9_stageB.yaml` | w_attack: 0.1 | 50 |

---

## Re-training Plan

**Config:** `config_sensor_v9_stageB_fixed.yaml` (create from current + fixes)

**Changes from A4:**
```yaml
model:
  lfo_depth_mul_base: [0.0, 0.0, 0.0, 0.0]  # was [0, 0, 0.3, 0]
  touch_amp_head: true  # was: false

loss_weights:
  w_attack: 0.1  # was: 0.0
```

**Training:**
- Steps: 2000
- Batch: 8
- LR: 1e-3 → 1e-4 (cosine decay)
- Seeds: 0, 1 (both)

**Expected improvements:**
- Swarm: clean attack, correct pitch
- All sounds: better dynamics, touch response
- Metrics: dtpk ≤20ms, dpk ≤4dB, pass ≥3/5

---

## Git Status

**Branch:** `orch/v9data-fix` @ `d3db5ef`

**Pending commits:**
1. `fix: f0_head now uses note input` (sensor_ddsp.py)
2. `config: enable touch_amp_head, fix LFO` (config files)

---

## Next Steps

1. Create `config_sensor_v9_stageB_fixed.yaml` with all fixes
2. Launch Stage B re-training (seed 0, then seed 1)
3. Evaluate on val curves
4. Compare to previous results

**Commands:**
```bash
# After config is ready
python3 train_sensor_ddsp_v5.py --config config_sensor_v9_stageB_fixed.yaml \
  --name v9_stageB_fixed_s0 --root /path/to/runs --steps 2000 --seed 0
```
