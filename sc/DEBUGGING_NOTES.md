# SuperCollider Debugging Notes

## Key Lesson: Detecting Routine Completion

**Problem:** Using `Thread.state` or `isPlaying` to detect when a Routine finishes is **unreliable**:
- State is only meaningful while the routine is waiting
- At start, state may be 0 briefly before becoming 1
- At end, state may briefly be non-zero before settling to 0
- This causes false "complete" detection

**Solution:** Use an explicit completion flag:

```supercollider
// In the routine (e.g., record_dataset.scd):
~recordingSeries = fork {
    try {
        // ... do work ...
    } { |error|
        error.reportError;
    };
    cleanup.value;
    ~recordingDone = true;  // Signal completion (finished OR aborted)
};

// In the driver (e.g., run_both.scd):
~recordingDone = nil;  // Clear before starting
recPath.load;          // Load script that starts the routine

// Wait for completion flag
while { ~recordingDone.isNil } {
    5.wait;
};
```

## Key Lesson: Loading Scripts Programmatically

**Problem:** Scripts wrapped in `( )` for IDE evaluation won't execute when loaded via `.load`

**Solution:** Remove outer parentheses from scripts meant to be loaded programmatically:

```supercollider
// ❌ Won't work with .load:
(
var x = 1;
// ... code ...
)

// ✅ Works with .load:
var x = 1;
// ... code ...
```

## Key Lesson: Command-Line sclang Execution

When running `sclang script.scd`:
1. Class library compiles
2. Script is **loaded** (not necessarily executed)
3. sclang exits unless something keeps it alive

To keep sclang running:
- Boot a server: `Server.default.boot`
- Or use: `thisThread.clock.play`

## Routine States

- `state = 0`: Not running (finished or never started)
- `state = 1`: Running
- `state = 2`: Suspended
- `state = 5`: Terminated with error

**Warning:** State is unreliable for completion detection. Use explicit flags.
