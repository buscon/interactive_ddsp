/*
Auto-executing wrapper for headless dataset recording.

This script:
1. Checks if Sentire is already loaded (via -scdefs or manually)
2. If not, loads Sentire headless
3. Waits for initialization
4. Automatically executes the recording script

Usage:
    // Option 1: Load Sentire separately, then this script
    sclang -scdefs "~/sentire/main-headless.scd"
    // Then:
    "~/interactive_ddsp/sc/record_dataset_headless.scd".load;
    
    // Option 2: Load both together (Sentire must be first)
    sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset_headless.scd"
    
    // Option 3: From SuperCollider IDE
    "~/interactive_ddsp/sc/record_dataset_headless.scd".load;
*/

var sentireFolder, mainHeadlessPath, recordingScriptPath, checkAndRun;

// Find the Sentire folder
sentireFolder = Platform.userHomeDir +/+ "sentire";
mainHeadlessPath = sentireFolder +/+ "main-headless.scd";
recordingScriptPath = thisProcess.nowExecutingPath.dirname +/+ "record_dataset.scd";

// Function to check if ready and run recording
checkAndRun = {
    if (g.isNil) {
        "record_dataset_headless: Server g not found. Boot Sentire first.".error;
        ^nil
    };
    
    if (g.controller.isNil) {
        "record_dataset_headless: g.controller not found. Waiting for Sentire...".postln;
        // Schedule another check
        AppClock.sched(1.0, checkAndRun);
        ^nil
    };
    
    // Sentire is ready - run the recording script
    "record_dataset_headless: Sentire is ready.".postln;
    "record_dataset_headless: Starting dataset recording...".postln;
    
    // Execute the recording script by loading it
    recordingScriptPath.load;
    
    nil
};

// Check if Sentire is already loaded
if (g.isNil or: { g.controller.isNil }) {
    "record_dataset_headless: Sentire not ready yet.".postln;
    
    // Try to load Sentire if we have the path
    if (File.exists(mainHeadlessPath)) {
        "record_dataset_headless: Loading Sentire from %...".format(mainHeadlessPath).postln;
        mainHeadlessPath.load;
        // Wait and check again
        AppClock.sched(2.0, checkAndRun);
    } {
        "record_dataset_headless: Sentire not found at %. Please load it first.".format(mainHeadlessPath).error;
        "record_dataset_headless: Usage: sclang -scdefs \"~/sentire/main-headless.scd,%\"".format(thisProcess.nowExecutingPath).error;
    };
} {
    // Sentire is already ready
    "record_dataset_headless: Sentire already running.".postln;
    checkAndRun.value;
};

"record_dataset_headless: Initialization started. Check post window for progress.".postln;
