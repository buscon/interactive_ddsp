/*
Auto-executing dataset recording script for headless mode.

This is identical to record_dataset.scd but wrapped to execute automatically
when loaded (not just define functions).

Usage:
    sclang -scdefs "~/sentire/main-headless.scd"
    // Then in SuperCollider or after Sentire boots:
    "~/interactive_ddsp/sc/record_dataset_auto.scd".load;
    
    Or combine with the wrapper:
    sclang -scdefs "~/sentire/main-headless.scd,~/interactive_ddsp/sc/record_dataset_auto.scd"
*/

// Wait for Sentire to be ready, then execute the recording script
AppClock.sched(1.0, {
    var sentireFolder, sentireBootPath;
    
    // Find Sentire
    sentireFolder = Platform.userHomeDir +/+ "sentire";
    sentireBootPath = sentireFolder +/+ "main-headless.scd";
    
    // Check if we need to boot Sentire
    if (g.isNil) {
        "record_dataset_auto: Server g not found. Please boot Sentire first.".error;
        "record_dataset_auto: Load % before this script.".format(sentireBootPath).error;
        ^nil
    };
    
    if (g.controller.isNil) {
        "record_dataset_auto: g.controller not found. Sentire may not be fully initialized.".error;
        ^nil
    };
    
    "record_dataset_auto: Sentire is ready. Starting dataset recording...".postln;
    
    // Execute the main recording script
    (thisProcess.nowExecutingPath.dirname +/+ "record_dataset.scd").load;
    
    nil
});

"record_dataset_auto: Script loaded. Will execute in 1 second if Sentire is ready.".postln;
