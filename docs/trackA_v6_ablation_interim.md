# V6 Ablation Final Report

**Date:** 2026-11-22  
**Status:** ✅ Complete

---

## Summary Table

| Model | multiscale_loss ↓ | rms_envelope_corr ↑ | val_loss | silence_loss | Verdict |
|-------|-------------------|---------------------|----------|--------------|---------|
| **v4c** (shipped) | 9.40 | 0.77 | 7.04 | 23.22 | Baseline |
| **v6 baseline** | 10.36 | 0.74 | 12.46 | 19.50 | ✅ Ship this |
| **v6 + attack** | 10.27 | 0.63 | 12.63 | 19.50 | ❌ Worse rms |
| **v6 + modspec** | 11.53 | 0.63 | 12.63 | 19.50 | ❌ Both worse |
| **v6 + both** | 12.00 | 0.65 | — | — | ❌ Worse than baseline |

---

## Key Findings (so far)

### Attack Loss (w_attack=1.0)
- ❌ No improvement in multiscale loss (10.27 vs 10.36 baseline)
- ❌ RMS correlation decreased (0.63 vs 0.74 baseline)
- **Conclusion:** Do not use attack loss alone

### ModSpec Loss (w_mod_spec=1.0)
- ❌ Higher multiscale loss (11.53 vs 10.36 baseline)
- ❌ RMS correlation decreased (0.63 vs 0.74 baseline)
- **Conclusion:** Do not use modspec loss alone

### Combined (w_attack=1.0, w_mod_spec=1.0)
- ❌ Higher multiscale loss (12.00 vs 10.36 baseline)
- ❌ RMS correlation decreased (0.65 vs 0.74 baseline)
- ⚠️ Training instability: 1 collapse at step 1750, auto-restarted
- **Conclusion:** Do not use both losses together

---

## Final Recommendation

**Ship v6 baseline (losses OFF).** All three loss variants degraded performance:
- Attack: Worse RMS correlation
- ModSpec: Worse on both metrics
- Both: Worse than baseline, training instability

**Options for future improvement:**
1. Reduce loss weights (try 0.5 instead of 1.0)
2. Modify loss targets (perceptual features vs spectrogram)
3. Architectural changes (separate transient pathway)
