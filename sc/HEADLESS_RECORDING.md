# Headless Dataset Recording

This guide explains how to run the dataset recording script without the GUI.

## Prerequisites

- Sentire installed at `~/sentire` (or adjust paths accordingly)
- SuperCollider installed

---

## Option 1: Use the Auto-Executing Wrapper (Recommended)

The wrapper script automatically detects when Sentire is ready and starts recording:

### From Command Line
```bash
# Load both scripts together (Sentire must be first!)
pw-jack sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset_headless.scd"
```

**Important:** Use `-scdefs` and list Sentire first. This keeps sclang running.

### Alternative Command (if the above doesn't work)
```bash
# The wrapper will boot a server and keep running
pw-jack sclang ~/interactive_ddsp/sc/record_dataset_headless.scd
```

### From SuperCollider IDE
```supercollider
// Just load the wrapper - it handles everything
"~/interactive_ddsp/sc/record_dataset_headless.scd".load;
```

The wrapper will:
1. Check if Sentire is already running
2. If not, load `main-headless.scd`
3. Wait for initialization (polls every 1 second)
4. Automatically start the recording script

---

## Option 2: Manual Two-Step Loading

If you prefer more control, load Sentire and the recording script separately:

```supercollider
// Step 1: Load Sentire headless
"~/sentire/main-headless.scd".load;

// Wait for it to initialize (watch the post window)
// When you see "Sentire boot complete" or similar, continue:

// Step 2: Load the recording script
"~/interactive_ddsp/sc/record_dataset_headless.scd".load;
// OR directly:
"~/interactive_ddsp/sc/record_dataset.scd".load;
```

---

## Option 3: Command Line with sclang

### Using the Shell Script
```bash
cd ~/interactive_ddsp/sc
./run_recording_headless.sh ~/sentire
```

### Manual sclang Command
```bash
# Load both together (order matters - Sentire first!)
pw-jack sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset_headless.scd"
```

---

## Option 4: Modify main-headless.scd to Auto-Run

Add this to the end of `~/sentire/main-headless.scd`:

```supercollider
// Auto-run dataset recording after Sentire boots
AppClock.sched(2.0, {
    "Auto-loading dataset recording script...".postln;
    "~/interactive_ddsp/sc/record_dataset.scd".resolvePath.load;
    nil
});
```

Then just run:
```bash
sclang -scdefs "~/sentire/main-headless.scd"
```

---

## Monitoring Progress

The script outputs status messages to the SuperCollider post window:

```
Dataset: 20 proximity curves x 5 sounds = 100 files, about 115 min
Dataset: recording 01_canopy_proximity_curve_001
  66.0 s, peak -12.3 dBFS  (01_canopy_proximity_curve_001_master.aiff)
Dataset: recording 01_canopy_proximity_curve_002
...
```

---

## Stopping Early

To stop the recording before it completes:

```supercollider
// In SuperCollider:
~stopRecordingSeries.value;
```

This will:
- Stop the current recording
- Save all files recorded so far
- Restore the original sound environment
- Free all buffers

---

## Resuming an Interrupted Run

The script supports resuming. Set the `folder` setting to the existing folder:

```supercollider
// In record_dataset.scd, change:
folder: "/path/to/existing/Dataset_20261120_143022",
skipExisting: true,  // already set to true by default
```

Files that already exist will be skipped.

---

## Troubleshooting

### "g not found" or "g.controller not found"
Sentire isn't booted. Load `main-headless.scd` first and wait for initialization.

### "Sound environment not loaded"
Make sure all the sound environments listed in the `sounds` array are loaded in Sentire.

### Files are silent
Check that the sound environments are actually producing audio. Test with a short run first:
```supercollider
nProximityCurves: 2,
proximityCurveDur: 10,
sounds: [["01_canopy", \whisperingCanopy]],
```

---

## Files Created

- `sc/record_dataset_headless.scd` - Wrapper script for headless mode
- `sc/run_recording_headless.sh` - Shell script for command-line usage
- `sc/HEADLESS_RECORDING.md` - This documentation
