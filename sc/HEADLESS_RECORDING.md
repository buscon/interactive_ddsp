# Headless Dataset Recording

This guide explains how to run the dataset recording script without the GUI.

## Prerequisites

- Sentire installed at `~/sentire` (or adjust paths accordingly)
- SuperCollider installed

---

## Option 1: Load Both Scripts from SuperCollider (Recommended)

### Step 1: Boot Sentire Headless
In SuperCollider, run:
```supercollider
// Load Sentire in headless mode
"~/sentire/main-headless.scd".resolvePath.load;

// Wait a moment for it to initialize
0.5.wait;

// Verify it's running
g.postStatus;  // should show server status
g.controller.postcs;  // should show the controller object
```

### Step 2: Load the Recording Script
Once Sentire is booted:
```supercollider
// Load the recording script
"~/interactive_ddsp/sc/record_dataset.scd".resolvePath.load;
```

The script will automatically start recording.

---

## Option 2: Use the Wrapper Script

The wrapper script `record_dataset_headless.scd` loads both automatically:

```supercollider
// In SuperCollider:
"~/interactive_ddsp/sc/record_dataset_headless.scd".resolvePath.load;
```

This will:
1. Load Sentire headless
2. Wait for initialization
3. Start the recording script

**Note:** Adjust the `sentireFolder` path in the script if your Sentire installation is elsewhere.

---

## Option 3: Command Line with sclang

### Using the Shell Script
```bash
cd ~/interactive_ddsp/sc
./run_recording_headless.sh ~/sentire
```

### Manual sclang Command
```bash
sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset.scd"
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
