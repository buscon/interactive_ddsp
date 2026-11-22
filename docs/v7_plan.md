# V7 Plan — Onset-Weighted Multiscale Loss (v2, Mistral Models)

**Date:** 2026-11-22
**Branch:** `orch/v7-fixes` @ `3a73b52`
**Base:** `orch/v7-onsetw` @ `1c88b47`
**Status:** READY FOR REVIEW — awaiting user approval before training

---

## ⚠️ Model Change (SAIA Out of Tokens)

SAIA/Qwen daily limit reached (800/800). All models switched to **Mistral**:

| Role | Model | ID | Provider |
|------|-------|-----|----------|
| Orchestration / reasoning / analysis | Mistral Large (latest) | `mistral/mistral-large-latest` | mistral |
| Coding / implementation | Codestral | `mistral/codestral-2508` | mistral |
| Reviewing / verification | Mistral Large (latest) | `mistral/mistral-large-latest` | mistral |
| Tie-breaker / unsure cases | Claude Sonnet 5.5 | `claude-agent/sonnet` | claude-agent (free) |
| Deep analysis | Claude Opus 5.5 | `claude-agent/opus` | claude-agent (free) |

---

## 1. Onset Source (CORRECTED)

Touch onsets = **sensor crossing 0.98 from below** (this IS the OSC touch trigger).
- 230 onsets across 100 files
- File: `bakeoff_notes_v7.json` (audio-aligned via `start_s`)
- OSC touch events are NOT separately logged — defined by proximity signal itself

---

## 2. Audibility Finding (CORRECTED — Full Dataset)

**Earlier report was wrong** (300ms window, 23 onsets). Full analysis (230 onsets):

| Post-onset window | Median RMS ratio | >2x |
|-------------------|-----------------|-----|
| 50–300ms | ~0.9x (dip) | 2–3% |
| 400ms | 1.14x | 28% |
| **500ms** | **2.58x** | **63%** |
| 700–2000ms | ~3.4x (sustained) | ~70% |

**Conclusion:** There IS a strong attack — it develops over 300–500ms after the
0.98 crossing (SC touchSource uses `Env.perc(0.01, 0.3–0.5)` — fast rise, slow decay).

**→ Window set to 500ms** in `config_sensor_v6_onsetw.yaml`.

---

## 3. V7 Solution (Implemented & Verified)

Onset-weighted multiscale loss inside `loss_curriculum()`:

| Parameter | Value | Source |
|-----------|-------|--------|
| `enabled` | true | — |
| `factor` (k) | 5.0 | Claude+Qwen agreed |
| `window_ms` | **500** | Full-dataset audibility (was 150) |
| `dilate_frames` | 1 | ±1 STFT frame |
| `touch_rate_hz` | 100 | — |

**Implementation verified:**
- ✅ Wired into `loss_curriculum()` (not dead code)
- ✅ Correct tensor axis `(B, freqs, frames)`
- ✅ Normalized weights `sum(w·L)/sum(w)`
- ✅ Fine-tuning: `--init_checkpoint`, 0.3× LR, 1000 steps
- ✅ Smoke test: weighted loss (70143) ≠ unweighted (195) — weighting is active

---

## 4. Execution Plan

| # | Task / Git Issue | Assigned to (agent) | Model (provider/ID) | File Ownership |
|---|------------------|---------------------|---------------------|----------------|
| 1 | **Smoke test training** (100 steps, fine-tune from V6) | Codestral | `mistral/codestral-2508` | Read-only run, logs only |
| 2 | **Full V7 training** (1000 steps fine-tune, batch 8, val every 250) | Codestral | `mistral/codestral-2508` | `runs/bakeoff_ddsp_poc/stageB_v7/` |
| 3 | **Control run** (V6 fine-tuned 1000 steps, weighting OFF) | Codestral | `mistral/codestral-2508` | `runs/bakeoff_ddsp_poc/stageB_v7_control/` |
| 4 | **Evaluation** (curves 017–020, metrics + onset-window metrics) | Mistral Large | `mistral/mistral-large-latest` | `eval_stageB_v5.py` (read-only) |
| 5 | **Review results + listening test prep** | Mistral Large | `mistral/mistral-large-latest` | `docs/v7_results.md` |
| 6 | **Tie-breaker if disagreement** | Claude Sonnet 5.5 | `claude-agent/sonnet` | Read-only |

---

## 5. Training Commands

```bash
cd /home/marcello/Documents/interactive_ddsp

# V7 main run (fine-tune from V6 baseline, onset weighting ON, 500ms window)
python train_sensor_ddsp_v5.py config_sensor_v6_onsetw.yaml \
  --init_checkpoint runs/bakeoff_ddsp_poc/stageB_v6/state_final.pth \
  --out runs/bakeoff_ddsp_poc/stageB_v7

# Control run (same but onset_weighting.enabled=false)
# → copy config, set enabled: false, out: stageB_v7_control
```

---

## 6. Success Criteria

| Metric | V6 baseline | V7 target |
|--------|-------------|-----------|
| multiscale_loss | 10.36 | ≤ 10.5 |
| rms_envelope_corr | 0.74 | ≥ 0.72 |
| silence_loss | 19.50 | ≤ 19.5 |
| **Onset-window RMS ratio (500ms)** | measure | **≥ 2.0 median** (target audio has 2.58x) |
| **Touch events (qualitative)** | almost inaudible | **perceptible attack** |

---

## 7. Open Decisions

| Decision | Default |
|----------|---------|
| PR/merge after training? | Neither (local review first) |
| 2 seeds for training? | Yes (seed 0 + seed 1) |
| Window 500ms vs 700ms? | 500ms (peaks there, 700ms is sustain) |

---

## 8. Git State

| Item | Value |
|------|-------|
| Integration branch | `orch/v7-fixes` @ `3a73b52` |
| Task branches (keep) | `orch/v7-fix-{loss-curriculum,finetune,audibility,cleanup}` |
| Untracked (protected) | old notes files, ONSET_REBUILD_REPORT.md |
| Nothing pushed | ✅ |

**Awaiting approval before dispatching training tasks.**
