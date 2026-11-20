# Project Timeline - Interactive DDSP

**Last Updated:** 2026-11-19

## Executive Summary

This document provides a Gantt-style timeline for the Interactive DDSP project, showing the planned work across all phases.

## Timeline Overview

```
Nov 19  Dec 10  Dec 17  Dec 24  Dec 31  Jan 07  Jan 10
├───────┼───────┼───────┼───────┼───────┼───────┼───────►
│       │       │       │       │       │       │
│   M0  │   M1  │       │   M2  │   M3  │   M4  │
│       │       │       │       │       │       │
```

## Phase M0: Decision and Setup

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| Decision note: sounds, sensors, sync | 2 h | Nov 19 - Dec 10 | Planned |

## Phase M1: Data Pipeline

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| Sensor schema in config.yaml | 2 h | Dec 7 - Dec 10 | Planned |
| Record replay dataset with sync check | 4 h | Dec 7 - Dec 16 | Planned |
| Update preprocess.py: stats, proximity_curve split, held-out sounds | 2 h | Dec 14 - Dec 17 | Planned |

## Phase M2: Model

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| N-sensor model with sound-ID embedding | 5 h | Dec 14 - Dec 23 | Planned |
| Update train.py for the new model | 4 h | Dec 24 - Dec 31 | Planned |
| Hand-designed baseline mapping (no training) | 4 h | Dec 21 - Dec 27 | Planned |
| Evaluate on held-out gestures and a held-out sound | 3 h | Dec 28 - Jan 3 | Planned |

## Phase M3: Evaluation

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| Prototype bake-off: DDSP vs RAVE | 10 h | Nov 19 - Dec 6 | Planned |

## Phase M4: Real-time and Delivery

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| Real-time path: realtime_forward and OSC listener | 7 h | Jan 1 - Jan 8 | Planned |
| Latency measurement, README and demo | 4 h | Jan 6 - Jan 10 | Planned |

## Stretch Goals

| Task | Duration | Planned Dates | Status |
|------|----------|---------------|--------|
| Residual learning on the hand-designed mapping | 3 h | Only if there is slack | Planned |

## Critical Path

1. **Decision Phase (M0)** - Must complete by Dec 10
2. **Data Pipeline (M1)** - Must complete by Dec 17
3. **Model Development (M2)** - Must complete by Jan 3
4. **Real-time (M4)** - Must complete by Jan 10

## Dependencies

```
M4: Real-time and Delivery
├── M3: Evaluation
│   ├── M2: Model
│   │   ├── M1: Data Pipeline
│   │   │   └── M0: Decision and Setup
│   │   └── (depends on M5: Decision note)
│   └── (depends on M3: Evaluation)
└── (depends on M3: Evaluation)
```

## Notes

- All estimates are in hours
- Dates are tentative and may be adjusted based on progress
- The timeline assumes the DDSP backbone is selected in the prototype bake-off
- If RAVE is selected, the model development tasks would need adjustment

## Gantt Chart (Text-based)

```
Nov 19-26   Nov 26-Dec 3   Dec 3-10   Dec 10-17   Dec 17-24   Dec 24-31   Jan 07-10
    │            │            │            │             │             │
    ▼            ▼            ▼            ▼             ▼             ▼
M0: DECISION
[Prototype bake-off: DDSP vs RAVE]

M1: DATA PIPELINE
        [Sensor schema]
        [Replay dataset w/ sync check]
                          [Update preprocess.py]

M2: MODEL
        [N-sensor model]
                          [Train.py]
          [Baseline mapping]
                                        [Evaluation]

M4: REAL-TIME
                                                        [Real-time path]
                                                                          [Latency & demo]
```