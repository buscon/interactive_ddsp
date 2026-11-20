/*
Combined script: Boots Sentire + Runs Dataset Recording

This script:
1. Loads Sentire in headless mode
2. Waits for it to initialize (3 seconds)
3. Automatically starts the dataset recording

Usage:
    cd ~/interactive_ddsp/sc
    pw-jack sclang record_with_sentire.scd

Or:
    ./run_with_sentire.sh

Configuration:
    Edit the `settings` block in record_dataset.scd to customize recording.
*/

var sentireFolder, mainHeadlessPath, recordingScriptPath;

// Configuration
sentireFolder = Platform.userHomeDir +/+ "sentire";
mainHeadlessPath = sentireFolder +/+ "main-headless.scd";
recordingScriptPath = thisProcess.nowExecutingPath.dirname +/+ "record_dataset.scd";

"========================================".postln;
"Dataset Recording with Sentire".postln;
"========================================".postln;
postln;

// Check if Sentire exists
if (File.exists(mainHeadlessPath).not) {
    "ERROR: Sentire not found at: %".format(mainHeadlessPath).error;
    "Please adjust sentireFolder path or install Sentire.".error;
    ^this
};

"Step 1: Loading Sentire headless...".postln;
"From: %".format(mainHeadlessPath).postln;
postln;

// Load Sentire headless
mainHeadlessPath.load;

// Wait for Sentire to initialize, then run recording
"Step 2: Waiting for Sentire to initialize (3 seconds)...".postln;

AppClock.sched(3.0, {
    // Check if Sentire is ready
    if (g.isNil) {
        "ERROR: Server g not found. Sentire may have failed to load.".error;
        ^nil
    };
    
    if (g.controller.isNil) {
        "ERROR: g.controller not found. Sentire may not have initialized properly.".error;
        ^nil
    };
    
    "Sentire is ready!".postln;
    postln;
    
    "Step 3: Loading dataset recording script...".postln;
    "From: %".format(recordingScriptPath).postln;
    postln;
    
    // Load the recording script
    recordingScriptPath.load;
    
    "Recording script loaded. Check output above for recording progress.".postln;
    postln;
    
    nil
});

// Keep sclang running so AppClock can execute
"Keeping sclang running...".postln;
thisThread.clock.play;

postln;
"----------------------------------------".postln;
"Script loaded. Waiting for Sentire...".postln;
"Watch the output for recording progress.".postln;
"----------------------------------------".postln;
