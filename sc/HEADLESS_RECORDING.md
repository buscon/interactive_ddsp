# Headless Dataset Recording

## Quick Start

### Recommended: Use the Driver Script

The driver script handles everything automatically:

```bash
cd ~/interactive_ddsp/sc
./run_both.sh
```

Or directly:
```bash
pw-jack sclang ~/interactive_ddsp/sc/run_both.scd
```

**What it does:**
1. ✅ Boots Sentire headless
2. ✅ Waits for initialization (polls g.controller)
3. ✅ Loads the recording script
4. ✅ Waits for recording to complete
5. ✅ Exits sclang automatically

---

## Configuration

Edit `record_dataset.scd` to customize the recording:

```supercollider
var settings = (
    // Sounds to record (name used in filenames, sound environment key)
    sounds: [
        ["01_canopy", \whisperingCanopy],
        ["02_forestPad", \forestPad],
        ["03_harmonicSwarm", \harmonicSwarm],
        ["04_modularPercussions", \modularPercussiveCloud],
        ["05_moss_and_mist", \mossAndMist]
    ],
    
    // Recording parameters
    nProximityCurves: 20,    // number of unique sensor curves
    proximityCurveDur: 60,   // duration in seconds
    thresholdTouch: 0.98,    // touch detection threshold
    touchHoldShort: [0.3, 3.0],   // short touch duration range
    touchHoldLong: [5.0, 45.0],   // long touch duration range
    pLongTouch: 0.5,         // probability of long touch
    
    // Output settings
    folder: nil,             // nil = auto-generate folder name
    skipExisting: true,      // skip files that already exist
    nodes: [\master],        // which nodes to record
);
```

### Quick Test Configuration

For a fast test (takes ~1 minute):

```supercollider
nProximityCurves: 2,
proximityCurveDur: 10,
sounds: [["01_canopy", \whisperingCanopy]],
```

---

## How It Works

### run_both.scd (Driver)

The driver script orchestrates the entire process:

1. **Loads Sentire** - `main-headless.scd` from `~/sentire/`
2. **Polls for readiness** - Waits for `g.controller` to exist (not fixed delay)
3. **Loads recording script** - `record_dataset.scd`
4. **Waits for recording** - Monitors `~recordingSeries` and `~recordingDone`
5. **Exits cleanly** - `0.exit` when complete

### record_dataset.scd (Recording)

The recording script:

1. Generates random proximity curves (reproducible from seed)
2. Replays each curve through every sound
3. Records audio and saves sensor data
4. Sets `~recordingDone = true` when finished

---

## Output

Files are saved to:
```
~/sentire/Recordings/Dataset_YYYYMMDD_HHMMSS/
├── proximity_curves/
│   ├── proximity_curve_001.csv
│   ├── proximity_curve_002.csv
│   └── ...
├── 01_canopy_proximity_curve_001_master.aiff
├── 01_canopy_proximity_curve_002_master.aiff
├── ...
├── manifest.csv          # metadata for all files
├── events.csv            # timing events
└── dataset_info.txt      # configuration used
```

---

## Duration

**Per file:** settle + lead + curve + tail = ~68 seconds (with default settings)

**Total:** curves × sounds × duration
- 20 curves × 5 sounds × 68s = **~113 minutes** (almost 2 hours)

Use the quick test configuration to verify everything works first!

---

## Stopping Early

Press `Ctrl+C` to stop the recording.

### To Resume Later

1. Find the folder path (shown in output)
2. Edit `record_dataset.scd`:
   ```supercollider
   folder: "/home/marcello/sentire/Recordings/Dataset_20261120_093928",
   skipExisting: true,  // already true by default
   ```
3. Run again - existing files will be skipped

---

## Troubleshooting

### "Sentire not found"
Make sure Sentire is installed at `~/sentire`. If elsewhere, edit the `sentireFolder` path in `run_both.scd`.

### "Sentire not ready after 60 seconds"
Sentire may have failed to load. Check that:
- `~/sentire/main-headless.scd` exists
- All required SoundEnvironments are loaded
- No errors in the SuperCollider post window

### "Recording routine did not start"
Check `record_dataset.scd` for syntax errors. Try loading it manually in the IDE first.

### Files are silent
- Check that sound environments are loaded in Sentire
- Verify audio output is working
- Test with a short recording first

### "late" messages
```
late 0.245667146
```
This is normal - just means the scheduler was briefly behind. Audio is still recorded correctly.

---

## Files

| File | Purpose |
|------|---------|
| `run_both.scd` | **Main driver** - orchestrates everything |
| `run_both.sh` | Shell wrapper for driver |
| `record_dataset.scd` | Recording script (called by driver) |
| `record_with_sentire.scd` | Legacy combined script (deprecated) |
| `record_dataset_standalone.scd` | Legacy standalone (deprecated) |
| `HEADLESS_RECORDING.md` | This documentation |

---

## Technical Details

### Why a Driver Script?

SuperCollider's command-line mode (`sclang script.scd`) has quirks:
- Scripts compile but don't always execute top-level code
- Asynchronous operations (like booting servers) need explicit waiting
- sclang exits immediately unless told to stay running

The driver solves this by:
- Using a `Routine` with explicit `wait` calls
- Polling for `g.controller` instead of fixed delays
- Waiting for `~recordingDone` flag before exiting
- Calling `0.exit` to terminate cleanly

### Server Architecture

```
┌─────────────────────────────────────┐
│  run_both.scd (driver)              │
│  - Loads Sentire                    │
│  - Waits for g.controller           │
│  - Loads record_dataset.scd         │
│  - Waits for ~recordingDone         │
│  - Exits                            │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  Sentire (main-headless.scd)        │
│  - Boots server                     │
│  - Creates g.controller             │
│  - Loads sound environments         │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│  record_dataset.scd                 │
│  - Generates proximity curves       │
│  - Replays through sounds           │
│  - Records audio                    │
│  - Sets ~recordingDone = true       │
└─────────────────────────────────────┘
```

---

## Alternative: SuperCollider IDE

If command-line doesn't work, use the IDE:

```supercollider
// 1. Boot Sentire headless
"~/sentire/main-headless.scd".resolvePath.load;

// 2. Wait for "Sentire boot complete"

// 3. Load recording script
"~/interactive_ddsp/sc/record_dataset.scd".resolvePath.load;
```

---

## See Also

- `SENSOR_SONIFICATION_PLAN.md` - Full project plan and methodology
- `sc/record_dataset.scd` - Recording script source
- `~/sentire/main-headless.scd` - Sentire headless boot script
