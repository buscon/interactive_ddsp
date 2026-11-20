# GitHub Issues Update Summary

## Overview

All instances of "trajectory" have been replaced with "proximity curve" in the local codebase to better reflect the Sentire system's actual functionality.

**Why this change?**
- The term "trajectory" implies a spatial path through physical space
- The Sentire sensor signal (0..1) represents **proximity** or **distance** between two people over time
- It's a temporal signal indicating touch interactions, not a spatial trajectory

## Files Updated (Local Codebase)

| File | Changes |
|------|---------|
| `SENSOR_SONIFICATION_PLAN.md` | 28 replacements |
| `generate.py` | 4 replacements |
| `sc/record_dataset.scd` | 52 replacements |

## GitHub Issues That Need Manual Update

The following GitHub issues contain "trajectory" in their content and need to be updated on GitHub.com:

### Issue #1: Prototype bake-off: DDSP vs RAVE
**URL:** https://github.com/buscon/interactive_ddsp/issues/1
**State:** Open
**Changes needed:**
- `trajectory-based split` → `proximity-curve-based split`
- `sensor trajectories` → `sensor proximity curves`
- `held-out trajectories` → `held-out proximity curves`

### Issue #4: Record replay dataset with sync check
**URL:** https://github.com/buscon/interactive_ddsp/issues/4
**State:** Open
**Changes needed:**
- `sensor trajectories` → `sensor proximity curves`
- `trajectory ID` → `proximity_curve ID`
- `nTrajectories: 2` → `nProximityCurves: 2`
- `trajectoryDur: 10` → `proximityCurveDur: 10`
- `trajectories/*.csv` → `proximity_curves/*.csv`

### Issue #5: Update preprocess.py: stats, trajectory split, held-out sounds
**URL:** https://github.com/buscon/interactive_ddsp/issues/5
**State:** Open
**Changes needed:**
- `trajectory split` → `proximity_curve split`
- `trajectory` → `proximity_curve` (multiple occurrences)

## Instructions to Update GitHub Issues

### Option 1: Manual Update (Recommended)
1. Visit each issue URL listed above
2. Click "Edit" on the issue
3. Replace the text as indicated above
4. Click "Update issue"

### Option 2: Using the Update Script
A Python script has been created at `/tmp/update_github_issues.py` that can update issues programmatically.

**Prerequisites:**
1. Create a GitHub Personal Access Token at https://github.com/settings/tokens
2. Set the environment variable: `export GITHUB_TOKEN=your_token_here`
3. Run the script: `python3 /tmp/update_github_issues.py`

**Note:** The script requires write access to the repository.

## Summary of Terminology Changes

| Old Term | New Term | Context |
|----------|----------|---------|
| `trajectory` | `proximity curve` | General term for sensor signal over time |
| `Trajectory` | `Proximity curve` | Capitalized version |
| `TRAJECTORY` | `PROXIMITY CURVE` | Uppercase version |
| `trajectory-based` | `proximity-curve-based` | Adjective form |
| `nTrajectories` | `nProximityCurves` | Variable name |
| `trajectoryDur` | `proximityCurveDur` | Variable name |
| `trajectories/` | `proximity_curves/` | Directory name |
| `trajectory_start_s` | `proximity_curve_start_s` | CSV column name |

## Verification

To verify the local codebase is correct:
```bash
cd /home/marcello/Documents/interactive_ddsp
grep -ri trajectory . --exclude-dir=.venv --exclude-dir=.git
```

Expected result: No matches found (after the changes have been committed).