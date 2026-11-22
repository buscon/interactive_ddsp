# AGENTS.md — interactive_ddsp (project-local rules)

## Mandatory Opus review gate (project-specific rule, added 2026-11-22)

**Always check plans AND results with Claude Opus 5.5 (`claude-agent/opus`) before acting on them.**

This was added after repeated large errors that slipped through (onset-weight window shifted
backwards, onsets dropped at the 4096 scale, V8 RMS mask crash meaning "training completed"
reports were false, touch_time threshold 0.3 vs SC 0.98 mismatch).

Applies to:

1. **Plans** — before dispatching any experiment/change: send the plan (goal, evidence,
   exact changes, success criteria) to Opus and incorporate its review. Do not dispatch
   on a plan Opus has not seen.
2. **Results** — before believing or acting on results (training reports, eval metrics,
   "tests pass", "training complete"): have Opus independently verify claims against the
   raw artifacts (logs, metrics.json, checkpoints, git state). A subagent's summary is
   not evidence — check `train_log.csv`, `metrics.json`, file existence yourself and via Opus.
3. **Disagreements** — if Opus and another model disagree, Opus wins unless the user
   explicitly overrides.

Exceptions: none for training/eval decisions. Pure formatting/typo fixes may skip it.

## Model rules (see also LogSeq Rules___Model_Selection)

- SAIA/Qwen models are OUT OF TOKENS (daily limit) — do not use `saia/*`.
- Coding → `mistral/codestral-2508`; reasoning/review → `mistral/mistral-large-latest`;
  verification tie-breaker → `claude-agent/sonnet`; **plans & results review → `claude-agent/opus`**.
- `/orchestrate` and `/orchestrate-git`: always show the task breakdown table
  (task, agent, model ID, file ownership) at the plan gate before approval.
