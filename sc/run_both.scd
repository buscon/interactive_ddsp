/*
Headless dataset recording driver.

Boots Sentire headless, waits for it to initialize, then runs the dataset recording.
Exits sclang automatically when recording is complete.

Usage:
    pw-jack sclang ~/interactive_ddsp/sc/run_both.scd
    
    Or:
    cd ~/interactive_ddsp/sc
    ./run_both.sh

Features:
- Waits for g.controller (not fixed delay)
- Waits for recording to start
- Waits for recording to complete
- Exits cleanly with status 0
- Configurable timeouts

Configuration:
Edit the `settings` block in record_dataset.scd to customize:
- nProximityCurves: number of sensor curves
- proximityCurveDur: duration of each curve
- sounds: list of sound environments to record
*/

(
Routine {
    var sentireFolder, sentirePath, recPath;
    var bootTimeout = 60;   // s, max wait for Sentire to be ready
    var startTimeout = 30;  // s, max wait for recording to start
    var waited = 0;

    // Configuration
    sentireFolder = Platform.userHomeDir +/+ "sentire";
    sentirePath = sentireFolder +/+ "main-headless.scd";
    recPath = thisProcess.nowExecutingPath.dirname +/+ "record_dataset.scd";

    "".postln;
    "========================================".postln;
    "Dataset Recording - Headless Mode".postln;
    "========================================".postln;
    "".postln;

    // ========================================================================
    // 1. Load Sentire
    // ========================================================================
    
    if (File.exists(sentirePath).not) {
        "ERROR: Sentire not found at: %".format(sentirePath).error;
        "Please install Sentire or adjust the path.".error;
        1.exit; 3600.wait;  // exit is not immediate: wait to avoid crash
    };
    
    "Step 1: Loading Sentire headless".postln;
    "From: %".format(sentirePath).postln;
    "".postln;
    
    sentirePath.load;

    // Wait for Sentire to initialize (poll g.controller)
    "Step 2: Waiting for Sentire to initialize...".postln;
    waited = 0;
    while { (g.isNil or: { g.controller.isNil }) and: { waited < bootTimeout } } {
        0.5.wait;
        waited = waited + 0.5;
    };
    
    if (g.isNil or: { g.controller.isNil }) {
        "ERROR: Sentire not ready after % seconds".format(bootTimeout).error;
        "Check that Sentire is installed correctly.".error;
        1.exit; 3600.wait;
    };
    
    1.wait;  // Grace period for sound environments / SynthDefs to load
    "Sentire is ready!".postln;
    "".postln;

    // ========================================================================
    // 2. Load Recording Script
    // ========================================================================
    
    if (File.exists(recPath).not) {
        "ERROR: Recording script not found at: %".format(recPath).error;
        1.exit; 3600.wait;
    };
    
    "Step 3: Loading dataset recording script".postln;
    "From: %".format(recPath).postln;
    "".postln;
    
    ~recordingSeries = nil;  // Clear any previous recording routine
    recPath.load;

    // ========================================================================
    // 3. Wait for Recording to Start
    // ========================================================================
    
    "Waiting for recording to start...".postln;
    waited = 0;
    while { ~recordingSeries.isNil and: { waited < startTimeout } } {
        0.5.wait;
        waited = waited + 0.5;
    };
    
    if (~recordingSeries.isNil) {
        "ERROR: Recording routine did not start within % seconds".format(startTimeout).error;
        "Check record_dataset.scd for errors.".error;
        1.exit; 3600.wait;
    };
    
    "Recording started.".postln;
    "".postln;

    // ========================================================================
    // 4. Wait for Recording to Complete
    // ========================================================================
    
    "Step 4: Recording in progress...".postln;
    "Watch the output above for file-by-file progress.".postln;
    "This may take several hours depending on your settings.".postln;
    "".postln;
    
    // Wait for the recording routine to finish (state 0 = not running)
    while { ~recordingSeries.notNil and: { ~recordingSeries.state != 0 } } { 
        5.wait  // Check every 5 seconds
    };
    
    "".postln;
    "========================================".postln;
    "Recording complete!".postln;
    "========================================".postln;
    "Files saved to: ~/sentire/Recordings/".postln;
    "Exiting sclang.".postln;
    "".postln;
    
    // ========================================================================
    // 5. Exit
    // ========================================================================
    
    0.exit;
}.play(AppClock);
)
