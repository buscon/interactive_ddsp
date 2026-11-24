# Dataset v6: Channel Structure & Usage

**Date:** 2026-11-23

---

## Original 7-Channel Recording Structure

Each `.aiff` file in `Dataset_v6_20261122/` contains 7 channels:

| Ch | Name | Description | Range | Used in Training? |
|----|------|-------------|-------|-------------------|
| **0** | Master L | Main audio output (left) | float | ❌ No |
| **1** | Master R | Main audio output (right) | float | ❌ No |
| **2** | Touch L | Touch layer synth (left) | float | ❌ No |
| **3** | Touch R | Touch layer synth (right) | float | ❌ No |
| **4** | Sensor | Proximity sensor signal | 0..1 | ✅ **YES** (primary input) |
| **5** | Touch gate | Binary touch trigger (0/1) | 0 or 1 | ⚠️ Derived (touch_time) |
| **6** | Note freq | Note frequency (Hz) | float | ✅ **YES** (note_hz) |

---

## Pre-processed NPZ Files

The aligned dataset (`bakeoff_data_aligned_v6/`) contains `.npz` files with:

| Key | Shape | Source | Usage |
|-----|-------|--------|-------|
| `audio` | (N,) int16 | Ch 0+1 (master, mono mix?) | ✅ Target for synthesis |
| `sensor` | (T,) float32 | Ch 4 | ✅ Model input |
| `touch_time` | (T,) float32 | Derived from Ch 4+5 | ✅ Model input (normalised) |
| `touch_layer` | (N,) int16 | Ch 2+3 (touch synth) | ❌ Stored but unused |
| `gate` | (T,) uint8 | Ch 5 | ⚠️ Used for onset detection |
| `note_hz` | (T,) float32 | Ch 6 | ✅ Used for f0 base |
| `sound`, `curve`, `split` | scalar | Metadata | ✅ Dataset organization |
| `start_s`, `offset_s` | scalar | Alignment | ✅ Audio/sync correction |

---

## What SensorDDSP v9 Actually Uses

### Model Inputs
```python
sensor: (B, T, 1)   # Ch 4, proximity 0..1
touch: (B, T, 3)    # Derived from touch_time:
                    #   [log1p(t_sec), exp(-t_sec/0.5), clamp(t_sec, 0, 40)]
sound_id: (B,)      # 0=canopy, 1=forestPad, 2=swarm, 3=moss (percussion excluded, see below)
note: (B, T)        # From note_hz (Ch 6), used for f0 base
```

### Model Output
```python
audio: (B, T*480, 1)  # Synthesized audio at 48kHz
```

### Loss Computation
```python
# Compare synthesized audio vs target (audio from NPZ)
loss_multiscale = STFT_loss(pred, target)      # 6 scales: 4096, 2048, 1024, 512, 256, 128
loss_lin = L1_loss(pred, target)               # Waveform L1
loss_rms = L1_loss(RMS(pred), RMS(target))     # Envelope match
loss_onset_rms = weighted_rms_loss(...)        # Attack region weighting
```

---

## Unused Channels

### Channels 0-3 (Master + Touch Layer Audio)
**Why unused:**
- Master (Ch 0,1) is the **target** for synthesis, not an input
- Touch layer (Ch 2,3) is a **separate synth voice** (not used as target)
- Model learns to recreate master from sensor + touch + note

### Potential Future Uses
1. **Touch layer as auxiliary target:** Multi-output model (master + touch)
2. **Stereo synthesis:** Use Ch 0+1 as stereo target (current model is mono)
3. **Timbre disentanglement:** Touch layer has different timbre → could help separate articulation from instrument

---

## Excluding 04_modularPercussions

**Recommendation:** ✅ **Exclude from training**

**Rationale:**
1. **Timbral mismatch:** Percussive vs sustained (canopy, forestPad, swarm, moss)
2. **Different attack profile:** Sharp transients vs gradual onsets
3. **Stage B results:** Best dpk (-1.79/-1.02 dB) but worst timing consistency (-92/+26 ms)
4. **Contamination risk:** Model tries to learn both percussive and sustained → compromises both

**New sound mapping:**
| Sound ID | Name | Include? |
|----------|------|----------|
| 0 | 01_canopy | ✅ |
| 1 | 02_forestPad | ✅ |
| 2 | 03_harmonicSwarm | ✅ |
| ~~3~~ | ~~04_modularPercussions~~ | ❌ |
| 3 | 05_moss_and_mist | ✅ (reindexed) |

**Changes required:**
- `n_sounds: 5` → `n_sounds: 4`
- `f0_base_hz_override: [110, 210, 100, 200, 110]` → `[110, 210, 100, 110]`
- `attack_lag_s: [...]` → remove percussion entry
- Update `index.json` to exclude 04_modularPercussions files

---

## Recommended Actions

1. **Exclude 04_modularPercussions** from training
2. **Stratified train/val split** by touch_time (see `docs/V9_DATA_ANALYSIS.md`)
3. **Consider adding touch_layer audio** as auxiliary target (optional, future work)
4. **Verify master audio channel:** Confirm if `audio` in NPZ is Ch 0, Ch 1, or (L+R)/2
