# Sensor Sonification Plan

**Date**: 2026-11-19  
**Status**: Active Planning Phase  
**Project**: Extension of `engine_ddsp` for general time-series sonification

---

## Executive Summary

This project extends the current `EngineDDSP` (engine sound synthesis from RPM/torque) to a **general-purpose sensor sonification system** that can take arbitrary time-series data and generate audio in real-time. The approach is **hybrid**: combining learned mappings with hand-designed controls, inspired by the Sentire system.

---

## Project Goals

### Primary Goal
Build a system for **real-time sonification of sensor data** where:
- Multiple sensor channels (proximity, touch, orientation, etc.) control sound parameters
- Learning refines hand-designed mappings
- System generalizes to novel sensor inputs
- Output is suitable for performance (low latency, musical quality)

### Key Requirements
- **Variable input dimensions**: Accept 2-N sensor channels
- **Real-time capable**: Block-based processing, low latency
- **Hybrid control**: Hand-designed mappings + learned refinements
- **Modular**: Pluggable synthesis backends

---

## Architecture Overview

### Current State (EngineDDSP)

```
Input: [f0, rpm, torque] → MLPs → GRU → Decoder → [Harmonic + Noise + Reverb] → Audio
```

- Fixed 3 inputs (f0, rpm, torque)
- Single synthesis type (engine-like: harmonic + noise)
- Physics-grounded (f0 derived from RPM)

### Target State (SensorSonify)

```
Input: [sensor1, sensor2, ..., sensorN] → Configurable → [Neural/Hand-designed] → [Pluggable Synth] → Audio
```

- Variable N inputs (configurable per session)
- Both neural and direct mapping modes
- Multiple synthesis backends (granular, additive, FM, noise)

---

## Sensor Signal and Touch

- Sentire gives **one channel**, 0 to 1, from the distance between two persons.
- The value reaching **0.98 (`thresholdTouch`) is the touch event**. While the persons stay in touch, the sound keeps changing with the seconds since the touch began; the change goes on for about 30 s.
- A "trajectory" in this plan means only this value over time (one sensor curve per recording), not a spatial path.
- **Consequence for the model:** it trains on 2 s chunks, so a chunk inside a long touch cannot show how long the touch has lasted. Add a derived input `touch_time` (seconds the value has stayed at or above 0.98, else 0), computed in preprocessing and computed the same way live. The GRU alone would average over the touch duration.
- **Consequence for the dataset:** it needs both short touches (touch sound only) and long holds (stay-in-touch change). Touches in the recording script are short (0.3-3 s) or long (5-45 s), 50/50.
- **Touch threshold:** Sentire defines a touch as the value reaching `thresholdTouch: 0.98`. So `touch_time` counts the seconds the value has stayed at or above 0.98, and is 0 below it. The recording script's random ramps stay at or below 0.95, so they never trigger a touch by accident; touches are made by an explicit rise to 1.

---

## Dataset Recording Script

`sc/record_dataset.scd` records the dataset from Sentire:
- 20 random, reproducible sensor curves of 60 s (seed in the settings), replayed through each of the sounds. 5 sounds = 100 files, about 115 min.
- The curves are written to `trajectories/trajectory_NNN.csv` at 100 Hz (`time_s,sensor`), with `manifest.csv`, `events.csv` and `dataset_info.txt` next to the audio.
- Curve start inside each audio file is measured in the language (a few ms). The sync check in the replay-dataset issue must confirm it.
- Not yet run against a live Sentire setup: do a short test first (`nTrajectories: 2`, `trajectoryDur: 10`, one sound).

---

## Phase 0: Prototype Bake-off (DDSP vs RAVE)

**Purpose**: choose the backbone (DDSP in this repo, or RAVE with a sensor-to-latent mapper) from evidence before committing to the full plan.
**Time-box**: about 10 h, finished by 6 Dec 2026. Deadline for the whole project: 10 Jan 2027.

### Shared test set (build once, use for both tracks)
- 2-3 of the 10 sounds: at least one pitched/harmonic and one noisy or textural.
- The one Sentire sensor channel plus the derived time-in-touch input (see "Sensor Signal and Touch").
- About 20-30 unique sensor curves, with at least 3 held out for validation.
- Same trajectory-based split and the same held-out evaluation clips for both tracks.

### Track A: DDSP (about 4 h)
1. Copy `model.py` to a prototype version with N sensor inputs plus a sound-ID embedding.
2. Train briefly (1-2k steps) on the shared set.
3. Render the held-out trajectories.

### Track B: RAVE (about 5 h)
1. Load an existing pretrained RAVE model (no training). Prefer one whose material resembles your sounds (e.g. textural if your target is textural). It was not trained on your 10 sounds, so this track tests sensor-to-latent control, controllability and CPU latency, not reproduction of your sounds.
2. Hand-map the sensors to a few latent dimensions and render the held-out trajectories offline.
3. Check decoder latency on CPU against the block size.
4. If time remains: train a small sensor-to-latent mapper on the shared set.

### Comparison (about 1 h)
| Criterion | How to measure |
|-----------|----------------|
| Fit to target sounds | DDSP only: multi-resolution STFT loss on held-out clips and a listening check against the recorded targets. RAVE (pretrained): judge whether the output is plausible and has enough timbral range; a fit to your sounds needs fine-tuning |
| Sensor responsiveness | Do the sensors audibly and predictably change the sound? |
| Timbre coverage | Does it handle the noisy/textural sound as well as the pitched one? |
| Controllability | Can you say what each sensor does (DDSP parameters vs opaque latents)? |
| Real-time feasibility | CPU latency per block; export path |
| Compute and effort | Training time, GPU need, code still to write |

### Decision rule
- Pick DDSP if it fits the pitched sounds well, responds clearly to the sensors, and the noisy sound is acceptable or can be dropped.
- Pick RAVE if DDSP clearly fails on the textural sound and RAVE runs within the latency budget.
- If neither is clearly better, pick DDSP: less compute, explicit control, and the code already exists.
- Record the result and reasoning as a short decision note before starting Phase 1.

### Caveats
- This is a quick comparison, not a fair benchmark. The RAVE track is untrained for these sounds, so a poor fit there says little about a fine-tuned model.
- Listening judgement is yours; the loss numbers only support it.
- Open questions about the 10 sounds still need answers to choose the test sounds.

---

## Implementation Roadmap

### Phase 1: Flexible Input Architecture (1-2 days)

**Goal**: Make model accept arbitrary sensor dimensions

**Tasks**:
1. Refactor `model.py`:
   - Change from fixed 3 inputs to configurable `sensor_names` list
   - Store `mean`/`std` per sensor channel in config
   - Support optional "ground truth" channel

2. Update `config.yaml`:
   ```yaml
   sensors:
     - name: sentire        # the single 0..1 channel (distance; 1 = touch)
       channel: 0
       mean: 0.5
       std: 0.3
   derived:
     - name: touch_time     # seconds the value has stayed >= 0.98 (thresholdTouch), 0 otherwise
   ```
   (Sentire has one channel, so the example now lists that channel plus the derived input. The list format still allows more sensors later.)

3. Update `preprocess.py`:
   - Read sensor names from config
   - Compute per-channel statistics

**Output**: `sensor_sonify.py` with configurable inputs

---

### Phase 2: Pluggable Synthesis Backends (2-3 days)

**Goal**: Support multiple sound types

**Tasks**:
1. Create `engine_ddsp/synth_registry.py`:
   ```python
   SYNTH_REGISTRY = {
       "harmonic_noise": EngineDDSP,
       "granular": GranularSynth,
       "additive": AdditiveSynth,
       "fm": FMSynth,
   }
   ```

2. Implement backend-specific classes:
   - **Granular**: Grain density, size, envelope from sensors
   - **Additive**: Harmonic amplitudes/frequencies from sensors
   - **FM**: Carrier/modulator ratios from sensors

3. Create `generate.py` variant that accepts synth type

**Output**: Multiple synthesis options selectable via config

---

### Phase 3: Hybrid Learning + Hand-Designed Mapping (Ongoing)

**Goal**: Combine neural learning with interpretability

**Strategy**:
1. **Start with hand-designed mappings**:
   - Proximity → grain density (0→100 grains/s)
   - Touch → grain size (0→50ms)
   - Orientation → spatial position

2. **Train neural network** to refine these mappings:
   - Small network on top of hand-designed parameters
   - Loss: audio reconstruction + physics regularization

3. **Extract learned corrections**:
   - Per-sensor scaling factors
   - Cross-sensor interaction weights

**Output**: Interpretable, tunable sonification

---

### Phase 4: Real-Time OSC Integration (1 week)

**Goal**: Connect to SuperCollider/TouchDesigner

**Tasks**:
1. Add `realtime_forward()` method in model
2. Create OSC listener (using `python-osc` library)
3. Process per-frame: OSC message → tensor → synthesis → audio output

**Output**: Live performance capable system

---

## Dataset Requirements

### Minimum Dataset Size
- **Recommended**: 200-500 high-quality segments (count unique sensor trajectories; see Multi-Sound Replay below)
- Each segment: 2 seconds of audio + corresponding sensor time-series
- At 48kHz: 96,000 audio samples per segment

### Data Characteristics

#### Must-Have Coverage
- **Sensor range**: Full range 0.0-1.0, including touches (value = 1) of short and long duration
- **Motion variety**: Static + slow + fast movements
- **Audio quality**: Clean, no clipping

#### Preferred Characteristics
- **Synchronized**: Sensor and audio perfectly aligned
- **Diverse gestures**: Multiple patterns per session
- **Dynamic range**: Soft to loud, quiet to intense

---

## Data Augmentation Strategies

| Augmentation | Purpose | Implementation |
|-------------|---------|----------------|
| Time stretching | Simulate different gesture speeds | ±10-20% speed |
| Gaussian noise | Robustness to sensor noise | σ=0.01-0.05 |
| Amplitude scaling | Simulate different recording levels | 0.8-1.2× |
| Random truncation | Train on variable-length segments | Random 1-3s chunks |

Sensor-channel permutation was dropped from this list. It would break the meaning of each channel.

---

## Multi-Sound Replay (Dataset Multiplication)

The same recorded sensor trajectories can be replayed through about 10 different sounds, and each replay is recorded. This is the main augmentation.

### What it gives
- About 10x audio diversity from the same sensor sessions.
- It makes "plug in different sounds" trainable: the sensors drive control and a separate input selects the timbre.

### What it does not give
- It adds no sensor diversity. 300 trajectories x 10 sounds = 3,000 segments but still only 300 unique gestures.
- Count dataset size by **unique sensor trajectories**. That limits generalisation to new gestures.

### Required design changes
1. **Sound-identity conditioning.** Without it, one sensor input maps to 10 different targets and the model averages them (muddy output).
   - Step 1: learned embedding per sound ID, concatenated with the sensor features before the GRU.
   - Step 2 (later): replace the ID with an audio embedding (e.g. CLAP) of a short reference clip, so unseen sounds can be conditioned on. An ID only covers the 10 trained sounds.
2. **Split by trajectory, not by segment.** The same gesture with sound A in train and sound B in validation is leakage.
   - Hold out whole trajectories for validation.
   - Hold out 1-2 sounds entirely to test generalisation to unseen sounds.
3. **Know what produced the 10 sounds.** If they are hand-designed SuperCollider synths driven by the sensors, the model is a neural surrogate of those 10 mappings. If only those 10 sounds are needed, running the SC synths directly is simpler and exact. Learning pays off for new sounds and blends between sounds, which is what the held-out-sound test measures.
4. **Check the synth backend against the sounds.** The harmonic+noise synth assumes a pitched source. Granular, percussive or noise-heavy sounds will fit poorly whatever the conditioning. Inspect the 10 sounds before choosing backends.

### Open questions
- What are the 10 sounds (SC synths, samples, other)?
- How many unique sensor trajectories, and what total duration?
- Is playback-and-record sample-accurate, so sensor and audio stay aligned across replays?

---

## Training Curriculum

### Week 1-2: 100-200 segments
**Goal**: Test pipeline works

- Start with the one Sentire channel plus the time-in-touch input
- Use direct mapping first (no neural network)
- Verify you can synthesize from sensor inputs
- Then add neural refinement

### Week 3-4: 200-500 segments
**Goal**: Train meaningful model

- Train with full dataset
- Evaluate: generalization, artifacts
- Add augmentation if needed

### Week 5+: 500-1000 segments
**Goal**: Full training, fine-tuning

- Train full model
- Hand-tune mappings where neural struggles
- Deploy for real use

---

## Success Criteria

### Technical Metrics
- [ ] Audio quality: SNR > 20 dB
- [ ] Latency: < 50 ms end-to-end
- [ ] Generalization: Works on unseen gestures
- [ ] Real-time: 48kHz, 480-block (100 Hz control rate)

### Musical Qualities
- [ ] Expressive: Clear sensor→sound relationship
- [ ] Responsive: Immediate feedback to gestures
- [ ] Musical: pleasant, intentional sound

---

## Related Work

### Existing Implementations
- **acids-ircam/ddsp_pytorch**: Original DDSP implementation, real-time Pd external
- **ddsp_pytorch issues**: Issue #13 discusses block_size constraints for real-time

### What's Missing
- **Variable input dimensions**: All DDSP variants assume fixed 2 inputs
- **Sensor sonification**: No public implementations for arbitrary sensor time-series
- **Hybrid learning**: Most approaches do full neural OR full rule-based

---

## Files and Directories

```
st8_procedural_engine/
├── engine_ddsp/
│   ├── model.py              # Current EngineDDSP (to be extended)
│   ├── synth.py              # Synthesis primitives
│   ├── data.py               # Preprocessing and Dataset
│   ├── sensor_sonify.py      # NEW: SensorSonify architecture
│   └── synth_registry.py     # NEW: Pluggable backends
├── config.yaml               # Update for sensor config
├── train.py                  # Update for sensor training
├── generate.py               # Update for sensor generation
├── preprocess.py             # Update for sensor preprocessing
└── SENSOR_SONIFICATION_PLAN.md  # This file
```

---

## Next Steps

### Immediate (Today)
1. [ ] Save this plan to Logseq
2. [ ] Create `sensor_sonify.py` starting point
3. [ ] Update `config.yaml` with sensor template

### This Week
1. [ ] Implement flexible input architecture
2. [ ] Create simple 2-sensor test case
3. [ ] Test pipeline with minimal dataset

### Next Week
1. [ ] Train first model
2. [ ] Evaluate results
3. [ ] Iterate on architecture

---

## Notes

### Why This Approach?
1. **Hybrid learning** provides interpretability + performance
2. **Start with hand-designed mappings** ensures you know what each sensor does
3. **Neural refinement** captures subtle interactions you might miss
4. **Modular design** lets you swap sound types easily

### Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| Insufficient data | Start with 2 sensors, use augmentation |
| Complex model | Start with direct mapping, add neural later |
| Training instability | Use physics regularization, monitor loss |
| Latency issues | Optimize for block-based processing |

---

## Logseq Links

- [[Projects/Postdoc]] — Tuning Ecologies connection
- [[Projects/CH-Proj-M]] — Data Sonification Course
- [[Projects/RestorationModelling]] — Existing ML infrastructure

---

## Version History

- **v1.0 (2026-11-19)**: Created from initial planning session
  - Hybrid learning approach
  - Variable input architecture
  - Pluggable synthesis backends
  - Real-time OSC integration target