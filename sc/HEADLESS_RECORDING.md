# Headless Dataset Recording

## Quick Start (Recommended)

### Option 1: Combined Script (Best for Command Line)

This script loads Sentire first, then the recording script:

```bash
cd ~/interactive_ddsp/sc
./run_with_sentire.sh
```

Or directly:
```bash
pw-jack sclang ~/interactive_ddsp/sc/record_with_sentire.scd
```

**What it does:**
1. Boots a JACK server
2. Loads Sentire headless (`~/sentire/main-headless.scd`)
3. Waits 3 seconds for initialization
4. Loads the recording script automatically
5. Starts recording

---

### Option 2: From SuperCollider IDE

If you prefer the IDE:

```supercollider
// Step 1: Boot Sentire headless
"~/sentire/main-headless.scd".resolvePath.load;

// Wait for initialization (watch post window)

// Step 2: Load the recording script
"~/interactive_ddsp/sc/record_dataset.scd".resolvePath.load;
```

---

## Configuration

Edit the `settings` block at the top of `record_dataset_standalone.scd`:

```supercollider
var settings = (
    sounds: [
        ["01_canopy", \whisperingCanopy],
        // ... add your sounds here
    ],
    
    nProximityCurves: 20,    // number of unique sensor curves
    proximityCurveDur: 60,   // duration in seconds
    // ... more settings
);
```

### Quick Test Configuration

For a quick test (2 curves, 10 seconds, 1 sound):

```supercollider
nProximityCurves: 2,
proximityCurveDur: 10,
sounds: [["01_canopy", \whisperingCanopy]],
```

---

## Stopping Early

To stop the recording before it completes:

**From SuperCollider IDE:**
```supercollider
~stopRecordingSeries.value;
```

**From command line:**
Press `Ctrl+C`

---

## Resuming an Interrupted Run

The script supports resuming. Set the `folder` setting to the existing folder:

```supercollider
folder: "/path/to/existing/Dataset_20261120_143022",
skipExisting: true,  // already true by default
```

Files that already exist will be skipped.

---

## Output

Files are created in:
```
~/sentire/Recordings/Dataset_YYYYMMDD_HHMMSS/
├── proximity_curves/
│   ├── proximity_curve_001.csv
│   ├── proximity_curve_002.csv
│   └── ...
├── 01_canopy_proximity_curve_001_master.aiff
├── 01_canopy_proximity_curve_002_master.aiff
├── ...
├── manifest.csv
├── events.csv
└── dataset_info.txt
```

---

## Troubleshooting

### "Sentire not found"
Make sure Sentire is installed at `~/sentire`. If it's elsewhere, edit the `sentireFolder` variable at the top of the script.

### "Sound environment not loaded"
Make sure all sound environments listed in the `sounds` array are loaded in your Sentire installation.

### Files are silent
Test with a short run first to verify audio is being produced:
```supercollider
nProximityCurves: 2,
proximityCurveDur: 10,
sounds: [["01_canopy", \whisperingCanopy]],
```

### sclang exits immediately
Use the standalone script (`record_dataset_standalone.scd`), not `record_dataset.scd`. The standalone version keeps sclang running.

---

## Files

| File | Purpose |
|------|---------|
| `record_with_sentire.scd` | **Recommended** - Loads Sentire, then recording script |
| `run_with_sentire.sh` | Shell wrapper for combined script |
| `record_dataset.scd` | For IDE use (Sentire must be loaded first) |
| `record_dataset_standalone.scd` | Alternative (may not work on all setups) |
| `HEADLESS_RECORDING.md` | This documentation |

---

## Technical Notes

### Why a Standalone Script?

The regular `record_dataset.scd` is wrapped in parentheses `( )` which means it must be **evaluated** in the IDE, not just loaded.

The standalone version:
- Is executable from command line
- Auto-boots Sentire if needed
- Keeps sclang running via `thisThread.clock.play`
- Contains all the recording logic inline

### How It Works

1. Script loads and checks if Sentire is running
2. If not, loads `main-headless.scd` and waits 3 seconds
3. Uses `AppClock.sched` to continue after Sentire initializes
4. Calls `thisThread.clock.play` to keep sclang alive
5. Executes the full recording script

---

## See Also

- `SENSOR_SONIFICATION_PLAN.md` - Full project plan
- `sc/record_dataset.scd` - Original recording script (IDE only)