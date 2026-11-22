## V7 Target Audibility Report

### Summary
- **Onsets analyzed:** 23
- **Files analyzed:** 10
- **Touch threshold:** 0.98 (sensor crossing)
- **Analysis window:** 300.0ms before, 300.0ms after onset

### Quantitative Results

| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Log-RMS change (300.0ms) | -0.0003 | 0.0089 | 0.0000 | -0.0198 | 0.0180 |
| Spectral centroid change | 36.45 Hz | 277.50 Hz | 0.00 Hz | -575.03 Hz | 1183.83 Hz |

**Onsets with increased loudness:** 8 / 23 (34.8%)  
**Onsets with increased brightness:** 10 / 23 (43.5%)

### Per-File Statistics

```
File                                        N   RMS_mean    RMS_std    RMS_min    RMS_max    Cent_mean
----------------------------------------------------------------------------------------------------
01_canopy_proximity_curve_004               3     0.0022     0.0049    -0.0024     0.0091       -11.06
01_canopy_proximity_curve_014               3    -0.0122     0.0087    -0.0198     0.0000        68.78
01_canopy_proximity_curve_015               3     0.0072     0.0078    -0.0000     0.0180      -198.72
01_canopy_proximity_curve_019               2     0.0009     0.0009     0.0000     0.0018       -29.70
02_forestPad_proximity_curve_010            3    -0.0065     0.0085    -0.0185     0.0000        40.76
02_forestPad_proximity_curve_013            1     0.0000     0.0000     0.0000     0.0000         0.00
02_forestPad_proximity_curve_018            1     0.0178     0.0000     0.0178     0.0178         1.22
05_moss_and_mist_proximity_curve_006        2     0.0013     0.0018    -0.0005     0.0030       627.10
05_moss_and_mist_proximity_curve_011        3     0.0008     0.0029    -0.0022     0.0046       -13.92
05_moss_and_mist_proximity_curve_020        2    -0.0015     0.0020    -0.0036     0.0005        -7.56
```

### Interpretation

**Touch audibility: MARGINALLY AUDIBLE**

Mean log-RMS change of -0.000 with only 35% positive

**Change character: VARIABLE/GRADUAL**

High variance relative to mean suggests inconsistent or gradual changes

The SuperCollider sound definitions show that touch modulation builds gradually over time:
- **canopy:** `rate=1+clip(0.3t,...)`, `depth=0.5t` (modulation rate and depth increase with touch time)
- **swarm:** `rate=0.1+0.03t` (slow rate increase)

This means the sonic effect of touch is not a sharp attack but a gradual evolution over seconds. The 213ms IQR jitter found in the earlier onset alignment audit (only 21% within ±47ms) is consistent with this gradual change pattern.

### Audio Clips

Generated 23 audio clips in `audibility_clips/` for qualitative listening check.

Each clip is 2 seconds: 1s before onset to 1s after onset.

Clips:
- `05_moss_and_mist_proximity_curve_006_03976.wav`
- `05_moss_and_mist_proximity_curve_006_05918.wav`
- `01_canopy_proximity_curve_015_00849.wav`
- `01_canopy_proximity_curve_015_01455.wav`
- `01_canopy_proximity_curve_015_02054.wav`
- `01_canopy_proximity_curve_004_01334.wav`
- `01_canopy_proximity_curve_004_04181.wav`
- `01_canopy_proximity_curve_004_04539.wav`
- `05_moss_and_mist_proximity_curve_020_01220.wav`
- `05_moss_and_mist_proximity_curve_020_03290.wav`

### Recommendation

**Window size: 300-500ms or longer**

Less than 50% of onsets show significant change in 300ms window.
This is consistent with gradual touch modulation (as in SC sound defs: rate=1+clip(0.3t,...), depth=0.5t).
The touch effect builds over seconds, not in a sharp 150ms attack.
Recommend extending the loss weighting window to 300-500ms or using a time-weighted approach.

### Next Steps for V7

1. **Extend loss weighting window** from 150ms to 300-500ms to capture gradual touch modulation
2. **Consider time-weighted loss** that increases with touch duration (matching SC behavior)
3. **Verify with listening tests** using the generated audio clips
4. **Re-run onset alignment audit** with new window to check if jitter improves

---

*Generated: Analysis script at `/tmp/opencode/wt-audibility/analyze_audibility.py`*
