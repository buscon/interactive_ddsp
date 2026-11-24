# scripts/

One-off diagnostic and render scripts from the V9 debugging sessions (2026-11-22/23).
They are not part of the training pipeline. Most have absolute paths hard-coded.

- `analysis/`: pitch/f0 checks (`analyze_pitch*`, `debug_f0*`, `verify_note_f0*`,
  `check_*note*`), dynamics analysis, onset audits, the f0-freeze test.
- `render/`: inference-time listening renders (swarm LFO fix, touch-dynamics gain hack).

Run them from the repo root with `PYTHONPATH=.` so `engine_ddsp` can be imported, e.g.

    PYTHONPATH=. python3 scripts/analysis/analyze_pitch.py

The render scripts add the repo root to `sys.path` themselves.
