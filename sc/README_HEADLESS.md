# Headless Recording - Simple Guide

## The Problem

When you run `sclang script.scd`, it:
1. Compiles the class library
2. Loads your script
3. **Exits immediately** (before AppClock events can run)

## Solution: Use SuperCollider IDE

The most reliable way to run headless recording:

### Step 1: Boot Sentire Headless
In SuperCollider IDE, run:
```supercollider
"~/sentire/main-headless.scd".resolvePath.load;
```

Wait for it to initialize (you'll see messages in the post window).

### Step 2: Load the Recording Script
```supercollider
"~/interactive_ddsp/sc/record_dataset.scd".resolvePath.load;
```

The recording will start automatically.

---

## Alternative: Command Line with Server Boot

If you must use command line, the script now boots a server to keep sclang alive:

```bash
pw-jack sclang ~/interactive_ddsp/sc/record_dataset_headless.scd
```

**However**, this may not work reliably because:
- sclang exits before AppClock events execute
- The server boot happens too late

**Recommended:** Use the SuperCollider IDE method above.

---

## Why `-scdefs` Doesn't Work

The `-scdefs` flag (which loads scripts and keeps running) is only available in newer versions of SuperCollider (3.14+). Your version (3.13.0) doesn't support it.

To check your version:
```bash
sclang -v
```

---

## Files

- `record_dataset_headless.scd` - Wrapper that tries to auto-run (may not work on sclang 3.13)
- `record_dataset.scd` - Main recording script (must be evaluated, not just loaded)
- `record_dataset_auto.scd` - Alternative auto-executing version

**Best approach:** Load `main-headless.scd` first in the IDE, then load `record_dataset.scd`.
