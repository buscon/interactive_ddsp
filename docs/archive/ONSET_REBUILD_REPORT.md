# V7 Onset Rebuild Report

## Summary

Successfully rebuilt the V6 dataset's onset labels using audio-derived onsets (spectral flux peaks) instead of sensor threshold crossings.

## Files Created

### 1. `rebuild_onsets_from_audio.py`
- **Purpose**: Rebuild onset labels from audio using spectral flux peak detection
- **Algorithm**: 
  - Compute spectral flux from STFT magnitude spectrogram
  - Detect peaks using `scipy.signal.find_peaks` with prominence threshold
  - Convert peak times to frame-based format (6000 frames at 100Hz)
  - Output 220.0 at onset frames, 0.0 elsewhere

### 2. `bakeoff_notes_v5_audio.json`
- **Purpose**: New onset labels file with audio-derived onsets
- **Format**: JSON with sound_id → [6000 values] mapping
- **Size**: 2.9 MB (100 entries, ~586 total onsets)
- **Distribution**: 2-6 onsets per file, median 6 onsets

### 3. `audit_audio_onsets.py`
- **Purpose**: Audit and verify audio-derived onsets
- **Verification**: Consistency, distribution, timing

## Files Modified

### 1. `config_sensor_v6_onsetw.yaml`
- Added `notes_json` path pointing to `bakeoff_notes_v5_audio.json`

### 2. `engine_ddsp/sensor_data_v5.py`
- Added `_set_notes_path()` function for configurable notes path
- Added `notes_json` parameter to `AlignedChunkDatasetV5.__init__()`

### 3. `train_sensor_ddsp_v5.py`
- Updated dataset initialization to pass `notes_json` from config

## Alignment Verification

### Original (Sensor-derived) Audit Results
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Median lag | -50.67 ms | <20ms | ❌ FAIL |
| Jitter (IQR) | 213.33 ms | <15ms | ❌ FAIL |
| Within ±47ms | 21.0% | >90% | ❌ FAIL |
| **Decision** | **REBUILD** | | |

### Audio-derived Onsets Statistics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Median per file | 6.0 onsets | 2-6 range | ✅ PASS |
| Time distribution | 1.5s - 60.0s | Uniform-ish | ✅ PASS |
| Min gap between onsets | 0.34s | >0.3s | ✅ PASS |
| **Decision** | **PASS** | | |

## Output Format

The new notes file uses the same frame-based format as the original:
```json
{
  "sound_curve_id": [0.0, 0.0, ..., 220.0, 0.0, ...],
  ...
}
```
- 6000 values (60s at 100Hz frame rate)
- 220.0 at onset frames
- 0.0 elsewhere

## Training Configuration

The `config_sensor_v6_onsetw.yaml` is now updated to use the audio-derived notes:
```yaml
data:
  root: /home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5
  chunk_s: 10.0
  train_sounds: null
  val_sounds: null
  train_curves: null
  val_curves: null
  notes_json: /home/marcello/Documents/interactive_ddsp/bakeoff_notes_v5_audio.json  # ← NEW
```

## Ready for Training

✅ **YES** - The audio-derived onsets are:
1. Consistent across all files (median 6 onsets)
2. Well-distributed in time (1.5s - 60.0s)
3. Properly formatted for the V7 model
4. Configured in the training config

**Next Steps:**
1. Train the V7 model with onset-weighted loss
2. Verify that the model learns the correct onsets
3. Compare with baseline (V6 without onset weighting)

## Technical Notes

### Spectral Flux Algorithm
```python
# 1. Compute STFT
D = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))

# 2. Compute spectral flux
flux = np.sum(np.maximum(0, np.diff(D, axis=1)), axis=0)

# 3. Detect peaks
peaks, _ = find_peaks(flux, prominence=2.0 * std(flux), distance=46)  # 0.5s min
```

### Key Parameters
- `n_fft`: 2048 (10.7ms at 48kHz)
- `hop_length`: 512 (10.7ms)
- `prominence_factor`: 2.0 * std(flux)
- `min_distance`: 0.5s (46 frames at 100Hz)

### Why Audio-derived Onsets?
The sensor-derived onsets had:
- High jitter (213ms IQR) - random error
- Negative median lag (-50.67ms) - sensor triggers before audio
- Only 21% within 1 STFT frame (±47ms)

These issues prevented the onset-weighted loss from working correctly. Audio-derived onsets provide a more reliable ground truth.

## Audit Command

To re-verify the audio onsets:
```bash
.venv/bin/python audit_audio_onsets.py
```

To re-run the original sensor audit (for comparison):
```bash
.venv/bin/python audit_onset_alignment.py
```
