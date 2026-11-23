# V9 Data Fix Report

**Date:** 2026-11-23  
**Branch:** `orch/v9data-fix` @ `81deb07`  
**BASE:** `orch/v9attack` @ `c373ef7`

---

## Problem

V9 Stage B training failed with both seeds:
- `dtpk = 53-134 ms` late (target ≤20 ms)
- `dpk = -5 to -6 dB` (target ≤4 dB)
- `strict_pass = 0/5` sounds (target ≥3/5)

**Root cause analysis identified two issues:**

1. **04_modularPercussions timbral mismatch:** Percussive sound contaminating sustained-sound training
2. **Train/val split imbalance:** All validation curves had sustained touches (≥11s), while 4/14 train curves had short percussive touches (<5s)

---

## Changes Made

### 1. Exclude 04_modularPercussions

**Rationale:** Percussive timbre fundamentally different from sustained sounds (canopy, forestPad, swarm, moss). Stage B results showed best dpk (-1.8 dB) but worst timing consistency (-92/+26 ms).

**Impact:**
- `n_sounds: 5` → `4`
- sound_ids remapped: `0=canopy, 1=forestPad, 2=swarm, 3=moss` (was `4=moss`)
- Total files: 100 → 80

### 2. Stratified Split by Touch Time

**Rationale:** Original val split (curves 004, 011, 017) had only sustained touches (≥11s). Model trained on short-touch curves couldn't generalize.

**New split:**
| Split | Files | Short (<5s) | Medium (5-15s) | Long (≥15s) | Avg Touch |
|-------|-------|-------------|----------------|-------------|-----------|
| Train | 48 | 8 | 16 | 24 | 16.6s |
| **Val** | **16** | **4** | **4** | **8** | **15.6s** |
| Test | 16 | 4 | 4 | 8 | 14.2s |

**Key improvement:** Val now has 4 short-touch curves (25%), enabling proper validation of attack behavior.

### 3. Code Updates

| File | Change |
|------|--------|
| `train_sensor_ddsp_v5.py` | Derive `sound_names_ordered` from index.json/config; f0_base fallback 4-element |
| `engine_ddsp/sensor_ddsp.py` | Default `attack_lag_s` buffer 4-element |
| `config_sensor_v9_stageB.yaml` | n_sounds=4, all arrays 4-element, updated eval_curves |
| `config_sensor_v9_fix.yaml` | n_sounds=4, all arrays 4-element, updated eval_curves |

---

## Verification

### Consistency Checks
```
✓ Config n_sounds (4) = index sound_ids (4)
✓ f0_base length (4) = n_sounds
✓ attack_lag length (4) = n_sounds
✓ lfo_* arrays length (4) = n_sounds
✓ No duplicate file paths in index.json
✓ sound_ids contiguous: [0, 1, 2, 3]
```

### 1-Batch Forward Pass
```
Model: 3,591,953 parameters
Input: (B=2, T=10) sensor, touch, sound_id
Output: (B=2, T×480=4800, 1) audio
✓ Forward pass successful
```

---

## Git Summary

**BASE:** `orch/v9attack` @ `c373ef7`  
**Current:** `orch/v9data-fix` @ `81deb07`

**Commits:**
```
81deb07 data: exclude 04_modularPercussions, stratified split by touch_time
fe50b80 fix(eval): unpack 5 values from evaluate_model()
c373ef7 docs: Stage B launch report and NO-GO results
```

**Files changed:**
| File | Lines + | Lines - |
|------|---------|---------|
| `config_sensor_v9_fix.yaml` | 10 | 9 |
| `config_sensor_v9_stageB.yaml` | 10 | 9 |
| `engine_ddsp/sensor_ddsp.py` | 2 | 1 |
| `train_sensor_ddsp_v5.py` | 19 | 5 |

**Data files (external, not versioned):**
- `/home/marcello/sentire/.../bakeoff_data_aligned_v6/index.json` (regenerated)

---

## What the User Should Check

1. **Verify index.json:** Confirm the new split looks correct
   ```bash
   python3 -c "import json; idx=json.load(open('/home/marcello/sentire/.../index.json')); print('Val curves:', idx['val_curves'])"
   ```

2. **Ready to re-train Stage B?**
   ```bash
   cd /home/marcello/Documents/interactive_ddsp
   python3 train_sensor_ddsp_v5.py --config config_sensor_v9_stageB.yaml --name v9_stageB_A4_s0 --root /home/marcello/sentire/.../runs/v9_stageB --steps 2000 --seed 0
   ```

3. **Expected improvement:**
   - Val dtpk should improve (short-touch curves now in val)
   - dpk may improve (no percussion contamination)
   - strict_pass should increase (better generalization)

---

## Delegation Log

| # | Task | Model | Outcome |
|---|------|-------|---------|
| 1 | Verify data directory | — | bakeoff_data_aligned_v6 confirmed |
| 2 | Analyze touch distribution | — | Stratified split designed |
| 3 | Generate new index.json | — | 80 files, 4 sounds, stratified |
| 4 | Update trainer script | codestral | sound_names_ordered dynamic |
| 5 | Update configs | codestral | n_sounds=4, arrays 4-element |
| 6 | Update model defaults | codestral | attack_lag_s 4-element |
| 7 | Verification | — | Config/index/model consistent, forward pass OK |
| 8 | Integration & commit | — | orch/v9data-fix @ 81deb07 |

**Review:** Claude Sonnet (ses_df7671a4cffeK8cIujVN8qmr5C) — ⚠️ Approved with 10 conditions (all addressed)

---

## Next Steps

1. **Re-train Stage B** with new data split
2. **Evaluate on val curves** (now includes short-touch)
3. **Compare to previous results:**
   - Old: dtpk=53-134ms, dpk=-6dB, pass=0/5
   - Target: dtpk≤20ms, dpk≤4dB, pass≥3/5

**Commands to review:**
```bash
git log --oneline orch/v9attack..orch/v9data-fix
git diff orch/v9attack...orch/v9data-fix
```
