# V9 Stage B Failure: Data Distribution Shift

**Date:** 2026-11-23  
**Analysis:** Train/Val split imbalance in touch characteristics

---

## Critical Finding

**The validation curves have fundamentally different touch characteristics than training curves.**

| Split | Touch Time Range | Short-Touch Curves (<5s) | Mean Touch Time |
|-------|------------------|--------------------------|-----------------|
| **Train** (14 curves) | 1.50s - 36.15s | **4 curves** (001, 003, 005, 006) | 14.88s |
| **Val** (3 curves) | **11.13s - 26.72s** | **0 curves** | 18.56s |
| **Test** (3 curves) | 13.49s - 26.97s | **0 curves** | 18.24s |

**Problem:** All validation curves have sustained touches (≥11s), while 4/14 train curves have short percussive touches (<5s).

---

## Per-Curve Breakdown (01_canopy)

| Curve | Split | Touch Time | Gate Sum | Character |
|-------|-------|------------|----------|-----------|
| 001 | train | **1.51s** | 457 | Short percussive |
| 002 | train | 19.34s | 2419 | Sustained |
| 003 | train | **1.50s** | 461 | Short percussive |
| **004** | **val** | **11.13s** | 1765 | Sustained |
| 005 | train | **1.50s** | 387 | Short percussive |
| 006 | train | **2.88s** | 829 | Short percussive |
| 007 | test | 14.27s | 1866 | Sustained |
| 008 | train | 26.63s | 3039 | Sustained |
| 009 | train | 12.10s | 1982 | Sustained |
| 010 | train | 28.44s | 3389 | Sustained |
| **011** | **val** | **17.84s** | 2239 | Sustained |
| 012 | train | 19.19s | 3864 | Sustained |
| 013 | train | 36.15s | 3885 | Sustained |
| 014 | test | 26.97s | 3791 | Sustained |
| 015 | train | 17.52s | 3798 | Sustained |
| 016 | train | 23.72s | 4439 | Sustained |
| **017** | **val** | **26.72s** | 3965 | Sustained |
| 018 | train | 10.83s | 2151 | Sustained |
| 019 | train | 7.00s | 1014 | Sustained |
| 020 | test | 13.49s | 1764 | Sustained |

---

## Why This Matters

### 1. Attack Energy vs Sustain
- **Short-touch curves:** Sharp attack, quick release → model learns to prioritize attack transient
- **Sustained-touch curves:** Long press, slow evolution → model learns to prioritize steady-state

### 2. Training Signal Imbalance
- Model trained on mix of short (4/14 = 29%) and long touches
- But validation is 100% long touches
- Model's attack-optimized behavior (from short-touch training) penalized on val

### 3. Explains Stage B Results
| Symptom | Explanation |
|---------|-------------|
| dtpk = 53-134 ms late | Model expects sharp attack (from short-touch training), but val has gradual onset |
| dpk = -5 to -6 dB | Attack energy mismatch: model over-predicts transient |
| 04_modularPercussions best (dpk=-1.8 dB) | Percussive sound has clearer attacks even in sustained playing |

---

## Root Cause

**The v6 aligned dataset was split by curve ID without stratification by touch characteristics:**
- Curves 001-006: Early recordings, more percussive exploration
- Curves 007-020: Later recordings, more sustained exploration
- Split assignment: 004, 011, 017 → val (all sustained)

This is a **dataset design flaw**, not a model architecture issue.

---

## Solutions

### Option 1: Rebalance Train/Val Split (RECOMMENDED)

**Stratified split by touch_time:**
```python
# Group curves by touch duration
short = ['001', '003', '005', '006']  # <5s
medium = ['009', '018', '019']  # 5-15s
long = ['002', '004', '007', '008', '010', '011', '012', '013', '014', '015', '016', '017', '020']  # >15s

# Balanced split (example)
train = ['001', '002', '003', '007', '008', '009', '010', '012', '014', '015', '018', '020']
val = ['004', '005', '011']  # Mix of short/medium/long
test = ['006', '013', '017']  # Hold-out
```

### Option 2: Train on Short-Touch Only

**Train only on curves 001, 003, 005, 006:**
- Model learns percussive attack behavior
- Evaluate on same distribution
- Separate model for sustained playing

### Option 3: Touch-Conditioned Model

**Add touch_time as input feature:**
- Model learns different attack behavior based on touch duration
- Requires architecture change (touch embedding)

### Option 4: Data Augmentation

**Synthesize short-touch variants of long curves:**
- Time-compress attack region
- Artificially create percussive training data

---

## Recommended Next Steps

1. **Verify pattern across all 5 sounds** (currently checked 01_canopy only)
2. **Rebalance train/val split** with stratification by touch_time
3. **Re-run Stage B training** with balanced split
4. **Evaluate separately** on short-touch vs sustained-touch test sets

---

## Files

- Dataset: `/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6/`
- Index: `index.json` (train/val/test split definition)
- Notes: `bakeoff_notes_v6.json` (onset annotations)
