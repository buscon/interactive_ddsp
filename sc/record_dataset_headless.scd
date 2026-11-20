/*
Wrapper script to run the dataset recording in headless mode.

This script:
1. Loads Sentire in headless mode (main-headless.scd)
2. Waits for it to initialize
3. Runs the dataset recording script

Usage:
    // From SuperCollider:
    thisProcess.nowExecutingPath.dirname ++ "/record_dataset_headless.scd".resolvePath.load;
    
    // Or from command line with sclang:
    sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset_headless.scd"
*/

var sentireFolder, mainHeadlessPath, recordingScriptPath;

// Find the Sentire folder (adjust if your setup is different)
sentireFolder = thisProcess.nowExecutingPath.dirname ++ "/../../sentire";

// Check if the path exists, if not try alternative locations
if (File.exists(sentireFolder).not) {
    sentireFolder = Platform.userHomeDir +/+ "sentire";
};

if (File.exists(sentireFolder).not) {
    "record_dataset_headless: Sentire folder not found. Please adjust the path in the script.".error;
    ^this
};

"record_dataset_headless: Loading Sentire from %".format(sentireFolder).postln;

// Load Sentire headless
mainHeadlessPath = sentireFolder +/+ "main-headless.scd";
if (File.exists(mainHeadlessPath).not) {
    "record_dataset_headless: main-headless.scd not found at %".format(mainHeadlessPath).error;
    ^this
};
mainHeadlessPath.load;

// Wait for Sentire to initialize (adjust time if needed)
"record_dataset_headless: Waiting for Sentire to initialize...".postln;
AppClock.sched(2.0, {
    // Check if Sentire is ready
    if (g.isNil or: { g.controller.isNil }) {
        "record_dataset_headless: Sentire not ready yet. Try loading again in a few seconds.".error;
    } {
        "record_dataset_headless: Sentire loaded successfully.".postln;
        "record_dataset_headless: Loading recording script...".postln;
        
        // Load the recording script
        recordingScriptPath = thisProcess.nowExecutingPath.dirname +/+ "record_dataset.scd";
        recordingScriptPath.load;
    };
    nil
});

"record_dataset_headless: Initialization scheduled. Check the post window for status.".postln;
