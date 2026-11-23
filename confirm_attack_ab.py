#!/usr/bin/env python3
"""
confirm_attack_ab.py — Overfit confirmation harness on Dataset v6 for V9 attack tuning.

1000-step overfit confirmation script on a balanced 10-chunk batch (5 s = 500 frames each,
2 chunks per sound covering early probe and late touch-from-level events).
Tracks onset-locked attack metrics across training steps, saving detailed onset rows
and summary metrics to check stability across steps 800-1000.
"""

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
import yaml
import numpy as np
import torch

# Ensure local repo imports work
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from engine_ddsp.sensor_ddsp import SensorDDSP
from eval_stageB_v5 import compute_attack_metrics


def count_valid_onsets(tt, chunk_frames=500, sr=48000, hop=96, win=240):
    """Count onsets within a chunk that satisfy f0 >= 55 and f0 + 160 <= n_frames."""
    onsets = np.where((tt[1:] > 0) & (tt[:-1] == 0))[0] + 1
    chunk_samples = chunk_frames * 480
    n_frames = (chunk_samples - win) // hop + 1
    cnt = 0
    for o in onsets:
        f0 = int(round(o * 0.01 * sr / hop))
        if f0 >= 55 and f0 + 160 <= n_frames:
            cnt += 1
    return cnt


def select_balanced_batch(v6_root, notes_dict=None, use_npz_note_hz=False, chunk_frames=500):
    """
    Selects 10 chunks (2 per sound, across 5 sounds) of 5 s (500 frames) from v6 train curves.
    Specifically selects chunks that cover BOTH early probe events (4-15 s)
    AND later mixed/touch-from-level events (20-55 s) across curves such as
    001/002/005/010/015, programmatically scanning train curves so EVERY sound
    has >= 4 valid onsets passing the f0 >= 55 and f0 + 160 <= len check.
    """
    index_path = os.path.join(v6_root, "index.json")
    with open(index_path) as f:
        index = json.load(f)

    sounds = ["01_canopy", "02_forestPad", "03_harmonicSwarm", "04_modularPercussions", "05_moss_and_mist"]
    preferred_curves = ["001", "002", "005", "010", "015", "003", "006", "008", "009", "012", "013", "016", "018", "019"]

    batch_meta = []
    chunk_items = []

    print("\nScanning train curves for balanced 10-chunk batch (>= 4 valid onsets per sound)...")
    for sound in sounds:
        sound_files = {f["curve"]: f for f in index["files"] if f["sound"] == sound and f["split"] == "train"}

        # Find early candidate (400 to 1500 frames = 4 to 15s)
        best_early = None
        for c in preferred_curves:
            if c not in sound_files:
                continue
            data = np.load(sound_files[c]["path"])
            tt = data["touch_time"]
            for off in range(400, min(1500, len(tt) - chunk_frames), 10):
                cnt = count_valid_onsets(tt[off:off + chunk_frames])
                if cnt >= 2:
                    best_early = {
                        "sound": sound,
                        "curve": c,
                        "offset": off,
                        "n_onsets": cnt,
                        "path": sound_files[c]["path"],
                    }
                    break
            if best_early:
                break

        # Find late candidate (2000 to 5500 frames = 20 to 55s), curve != early curve
        best_late = None
        for c in preferred_curves:
            if c not in sound_files or (best_early and c == best_early["curve"]):
                continue
            data = np.load(sound_files[c]["path"])
            tt = data["touch_time"]
            for off in range(2000, min(5500, len(tt) - chunk_frames), 10):
                cnt = count_valid_onsets(tt[off:off + chunk_frames])
                if cnt >= 2:
                    best_late = {
                        "sound": sound,
                        "curve": c,
                        "offset": off,
                        "n_onsets": cnt,
                        "path": sound_files[c]["path"],
                    }
                    break
            if best_late:
                break

        if best_early is None or best_late is None or (best_early["n_onsets"] + best_late["n_onsets"] < 4):
            raise RuntimeError(
                f"Failed to find >= 4 valid onsets for {sound}: early={best_early}, late={best_late}"
            )

        batch_meta.extend([best_early, best_late])

    # Print exact (sound, curve, offset_frames, n_onsets) for each chunk
    print("\nSelected 10-chunk batch:")
    for b in batch_meta:
        print(f"  sound={b['sound']}, curve={b['curve']}, offset_frames={b['offset']}, n_onsets={b['n_onsets']}")

    # Load chunk arrays
    audios = []
    sensors = []
    touches = []
    touch_raws = []
    notes = []
    sound_ids = []

    sound_id_map = {s: i for i, s in enumerate(sounds)}

    for b in batch_meta:
        data = np.load(b["path"])
        off = b["offset"]
        audio_chunk = data["audio"][off * 480 : (off + chunk_frames) * 480].astype(np.float32) / 32768.0
        sensor_chunk = data["sensor"][off : off + chunk_frames].astype(np.float32)
        tt_chunk = data["touch_time"][off : off + chunk_frames].astype(np.float32)

        # 3 touch features: log1p(t), (t>0)*exp(-t/0.5), clip(t,0,40)
        t_sec = tt_chunk
        f1 = np.log1p(t_sec)
        f2 = (t_sec > 0) * np.exp(-t_sec / 0.5)
        f3 = np.clip(t_sec, 0.0, 40.0)
        touch_chunk = np.stack([f1, f2, f3], axis=-1)

        # Note labels: from npz if requested and available, else from notes dict
        if use_npz_note_hz and "note_hz" in data.files:
            note_chunk = data["note_hz"][off : off + chunk_frames].astype(np.float32)
        elif notes_dict is not None:
            key = f"{b['sound']}_proximity_curve_{b['curve']}"
            full_note = notes_dict.get(key, [0.0] * (off + chunk_frames))
            note_chunk = np.asarray(full_note[off : off + chunk_frames], dtype=np.float32)
        else:
            note_chunk = np.zeros(chunk_frames, dtype=np.float32)

        if len(note_chunk) < chunk_frames:
            note_chunk = np.pad(note_chunk, (0, chunk_frames - len(note_chunk)), mode="constant")

        audios.append(audio_chunk)
        sensors.append(sensor_chunk)
        touches.append(touch_chunk)
        touch_raws.append(tt_chunk)
        notes.append(note_chunk)
        sound_ids.append(sound_id_map[b["sound"]])

    batch_tensors = {
        "audio": torch.from_numpy(np.stack(audios, axis=0)),
        "sensor": torch.from_numpy(np.stack(sensors, axis=0)).unsqueeze(-1),
        "touch": torch.from_numpy(np.stack(touches, axis=0)),
        "touch_raw": torch.from_numpy(np.stack(touch_raws, axis=0)),
        "note": torch.from_numpy(np.stack(notes, axis=0)).unsqueeze(-1),
        "sound_id": torch.tensor(sound_ids, dtype=torch.long),
        "metadata": batch_meta,
        "sounds": sounds,
    }
    return batch_tensors


def main():
    p = argparse.ArgumentParser(description="Confirm attack loss and architectural changes on Dataset v6")
    p.add_argument("--arm", type=str, required=True, help="Arm label / name")
    p.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    p.add_argument("--L", type=float, default=100.0, help="Linear spectral loss weight (default: 100.0)")
    p.add_argument("--R", type=float, default=1.0, help="RMS loss weight (default: 1.0)")
    p.add_argument("--peak", type=float, default=90.0, help="Onset weighting peak_ms (default: 90.0)")
    p.add_argument("--onset_rms_factor", type=float, default=None, help="Onset RMS factor (default: None)")
    p.add_argument("--touch_amp_head", type=int, default=0, help="Use dedicated touch amp head (default: 0)")
    p.add_argument("--lr_start", type=float, default=1e-3, help="Start learning rate (default: 1e-3)")
    p.add_argument("--lr_stop", type=float, default=1e-4, help="Stop learning rate (default: 1e-4)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p.add_argument("--out_dir", type=str, default="/tmp/opencode/v9attack_confirm", help="Output directory")
    p.add_argument("--use_npz_note_hz", type=int, default=0, help="Use note_hz from npz (default: 0)")
    args = p.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Running confirm_attack_ab on {device} (arm={args.arm}, seed={args.seed})")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load config_sensor_v9.yaml
    cfg_path = os.path.join(REPO_ROOT, "config_sensor_v9.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    v6_root = cfg["data"]["root"]
    notes_json_path = cfg["data"].get("notes_json")
    notes_dict = None
    if notes_json_path and os.path.exists(notes_json_path):
        with open(notes_json_path) as f:
            notes_dict = json.load(f)

    # Select balanced 10-chunk batch
    batch = select_balanced_batch(
        v6_root,
        notes_dict=notes_dict,
        use_npz_note_hz=bool(args.use_npz_note_hz),
        chunk_frames=500
    )

    audio_batch = batch["audio"].to(device)
    sensor_batch = batch["sensor"].to(device)
    touch_batch = batch["touch"].to(device)
    touch_raw_batch = batch["touch_raw"].to(device)
    note_batch = batch["note"].to(device)
    sound_id_batch = batch["sound_id"].to(device)
    batch_meta = batch["metadata"]
    sound_names = batch["sounds"]

    # Model creation
    model_kwargs = cfg["model"].copy()
    model_kwargs.pop("lfo_rate_base", None)
    model_kwargs.pop("lfo_depth_mul_base", None)
    model_kwargs.pop("lfo_depth_add_base", None)
    override = model_kwargs.pop("f0_base_hz_override", None)
    model_kwargs["f0_base_hz"] = list(override) if override else [110.0, 210.0, 100.0, 200.0, 110.0]

    # Configure onset_weighting
    ow = model_kwargs.get("onset_weighting", {}).copy()
    ow["enabled"] = True
    ow["peak_ms"] = float(args.peak)
    model_kwargs["onset_weighting"] = ow

    # Check signature for optional kwargs
    sig = inspect.signature(SensorDDSP.__init__).parameters
    if "w_lin" in sig:
        model_kwargs["w_lin"] = float(args.L)
    if "w_rms" in sig:
        model_kwargs["w_rms"] = float(args.R)
    if "onset_rms_factor" in sig and args.onset_rms_factor is not None:
        model_kwargs["onset_rms_factor"] = float(args.onset_rms_factor)
    if "touch_amp_head" in sig:
        model_kwargs["touch_amp_head"] = bool(args.touch_amp_head)

    # Filter out unknown keys if no var_keyword
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    if not has_var_kw:
        for k in list(model_kwargs.keys()):
            if k not in sig:
                model_kwargs.pop(k, None)

    model = SensorDDSP(**model_kwargs).to(device)
    if hasattr(model, "set_attack_lag") and "attack_lag_s" in cfg["model"]:
        model.set_attack_lag(cfg["model"]["attack_lag_s"])

    # Optimizer & parameter groups (10x LR for fast_head and touch_amp_proj)
    fast_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fast_head.") or name.startswith("touch_amp_proj.") or "fast_head" in name or "touch_amp_proj" in name:
            fast_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr_mult": 1.0},
        {"params": fast_params, "lr_mult": 10.0},
    ]
    optimizer = torch.optim.Adam(param_groups, lr=args.lr_start)

    scales = cfg["train"]["scales"]
    overlap = cfg["train"]["overlap"]

    eval_steps = sorted(set([0, 200, 400, 600, 800, 900, 1000, args.steps]))
    eval_steps = [s for s in eval_steps if s <= args.steps]

    step_history = []
    saved_onset_rows = []
    eval_results_by_step = {}

    def get_lr_for_step(step_idx):
        if step_idx < 100:
            # Linear warmup 0 to lr_start over 100 steps
            factor = float(step_idx + 1) / 100.0
            return args.lr_start * factor
        else:
            # Cosine decay from lr_start to lr_stop
            decay_steps = max(1, args.steps - 100)
            progress = min(1.0, float(step_idx - 100) / float(decay_steps))
            return args.lr_stop + 0.5 * (args.lr_start - args.lr_stop) * (1.0 + math.cos(math.pi * progress))

    print(f"\nStarting {args.steps}-step overfit confirmation run...")
    print(f"Eval steps: {eval_steps}")

    for step in range(args.steps + 1):
        # 1. Evaluate at scheduled eval steps
        if step in eval_steps:
            model.eval()
            with torch.no_grad():
                pred = model(sensor_batch, touch_batch, sound_id_batch, note=note_batch).squeeze(-1)
                rep_loss = float(model.loss_fn(pred, audio_batch, scales, overlap).item())

                # Collect onset metrics and f0ratio across all 10 batch items
                pred_np = pred.cpu().numpy()
                audio_np = audio_batch.cpu().numpy()
                touch_raw_np = touch_raw_batch.cpu().numpy()
                note_np = note_batch.squeeze(-1).cpu().numpy()
                last_f0_np = getattr(model, "_last_f0", None)
                if last_f0_np is not None:
                    last_f0_np = last_f0_np.squeeze(-1).cpu().numpy()

                step_onsets_by_sound = {s: [] for s in sound_names}
                step_all_onsets = []

                for i in range(len(batch_meta)):
                    snd = batch_meta[i]["sound"]
                    crv = batch_meta[i]["curve"]
                    off = batch_meta[i]["offset"]

                    # Compute f0ratio for item i
                    snd_id = int(sound_id_batch[i].item())
                    base_f0 = float(cfg["model"]["f0_base_hz_override"][snd_id])
                    tgt_f0 = np.where(note_np[i] > 0, note_np[i], base_f0)
                    if last_f0_np is not None:
                        pred_f0 = last_f0_np[i]
                        active_mask = (touch_raw_np[i] > 0)
                        if active_mask.sum() > 0:
                            item_f0ratio = float(np.median((pred_f0 / tgt_f0)[active_mask]))
                        else:
                            item_f0ratio = float(np.median(pred_f0 / tgt_f0))
                    else:
                        item_f0ratio = 1.0

                    res = compute_attack_metrics(pred_np[i], audio_np[i], touch_raw_np[i])
                    for on in res["onsets"]:
                        row = {
                            "step": step,
                            "sound": snd,
                            "curve": crv,
                            "chunk_offset": off,
                            "onset_frame": on["onset_frame"],
                            "f0": on["f0"],
                            "dpk_db": on["dpk_db"],
                            "abs_dpk_db": abs(on["dpk_db"]),
                            "dtpk_ms": on["dtpk_ms"],
                            "abs_dtpk_ms": on["abs_dtpk_ms"],
                            "p_bump_db": on["p_bump_db"],
                            "t_bump_db": on["t_bump_db"],
                            "dpre_db": on["dpre_db"],
                            "dp300_db": on["dp300_db"],
                            "f0ratio": item_f0ratio,
                        }
                        step_onsets_by_sound[snd].append(row)
                        step_all_onsets.append(row)
                        if step in [800, 900, 1000]:
                            saved_onset_rows.append(row)

                # Compute per-sound medians
                sound_eval = {}
                strict_pass_count = 0
                for snd in sound_names:
                    s_rows = step_onsets_by_sound[snd]
                    if s_rows:
                        m_dpk = float(np.median([r["dpk_db"] for r in s_rows]))
                        m_abs_dpk = float(np.median([r["abs_dpk_db"] for r in s_rows]))
                        m_dtpk = float(np.median([r["dtpk_ms"] for r in s_rows]))
                        m_abs_dtpk = float(np.median([r["abs_dtpk_ms"] for r in s_rows]))
                        m_pbump = float(np.median([r["p_bump_db"] for r in s_rows]))
                        m_tbump = float(np.median([r["t_bump_db"] for r in s_rows]))
                        m_dpre = float(np.median([r["dpre_db"] for r in s_rows]))
                        m_dp300 = float(np.median([r["dp300_db"] for r in s_rows]))
                        m_f0r = float(np.median([r["f0ratio"] for r in s_rows]))
                        passed = bool(abs(m_dpk) <= 4.0 and m_abs_dtpk <= 20.0)
                        if passed:
                            strict_pass_count += 1
                        sound_eval[snd] = {
                            "dpk": m_dpk,
                            "med_abs_dpk": m_abs_dpk,
                            "dtpk": m_dtpk,
                            "med_abs_dtpk": m_abs_dtpk,
                            "p_bump": m_pbump,
                            "t_bump": m_tbump,
                            "dpre": m_dpre,
                            "dp300": m_dp300,
                            "f0ratio": m_f0r,
                            "n_onsets": len(s_rows),
                            "pass": passed,
                        }
                    else:
                        sound_eval[snd] = {k: float("nan") for k in ["dpk", "med_abs_dpk", "dtpk", "med_abs_dtpk", "p_bump", "t_bump", "dpre", "dp300", "f0ratio"]}
                        sound_eval[snd]["n_onsets"] = 0
                        sound_eval[snd]["pass"] = False

                if step_all_onsets:
                    overall_eval = {
                        "dpk": float(np.median([r["dpk_db"] for r in step_all_onsets])),
                        "med_abs_dpk": float(np.median([r["abs_dpk_db"] for r in step_all_onsets])),
                        "dtpk": float(np.median([r["dtpk_ms"] for r in step_all_onsets])),
                        "med_abs_dtpk": float(np.median([r["abs_dtpk_ms"] for r in step_all_onsets])),
                        "p_bump": float(np.median([r["p_bump_db"] for r in step_all_onsets])),
                        "t_bump": float(np.median([r["t_bump_db"] for r in step_all_onsets])),
                        "dpre": float(np.median([r["dpre_db"] for r in step_all_onsets])),
                        "dp300": float(np.median([r["dp300_db"] for r in step_all_onsets])),
                        "f0ratio": float(np.median([r["f0ratio"] for r in step_all_onsets])),
                        "total_onsets": len(step_all_onsets),
                        "strict_pass_count": strict_pass_count,
                    }
                else:
                    overall_eval = {"total_onsets": 0, "strict_pass_count": 0}

                eval_results_by_step[str(step)] = {
                    "report_loss": rep_loss,
                    "overall": overall_eval,
                    "per_sound": sound_eval,
                }

                # Print evaluation summary
                print(f"\n--- Step {step:4d} | Report Loss: {rep_loss:.4f} | Strict Pass: {strict_pass_count}/5 ---")
                print(f"{'Sound':<22} {'N':>2} {'dpk':>7} {'|dpk|':>7} {'dtpk':>7} {'|dtpk|':>7} {'pbump':>7} {'tbump':>7} {'dpre':>7} {'dp300':>7} {'f0r':>6} {'Pass'}")
                for snd in sound_names:
                    se = sound_eval[snd]
                    p_str = "PASS" if se["pass"] else "FAIL"
                    print(
                        f"{snd:<22} {se['n_onsets']:>2d} {se['dpk']:>7.2f} {se['med_abs_dpk']:>7.2f} "
                        f"{se['dtpk']:>7.1f} {se['med_abs_dtpk']:>7.1f} {se['p_bump']:>7.2f} {se['t_bump']:>7.2f} "
                        f"{se['dpre']:>7.2f} {se['dp300']:>7.2f} {se['f0ratio']:>6.3f}  {p_str}"
                    )

        if step == args.steps:
            break

        # 2. Training step
        model.train()
        cur_lr = get_lr_for_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr * pg.get("lr_mult", 1.0)

        optimizer.zero_grad()
        pred = model(sensor_batch, touch_batch, sound_id_batch, note=note_batch).squeeze(-1)

        # Loss curriculum with kwargs fallback
        try:
            loss_kwargs = {"w_lin": args.L, "w_rms": args.R}
            if args.onset_rms_factor is not None:
                loss_kwargs["onset_rms_factor"] = args.onset_rms_factor
            loss, loss_details = model.loss_curriculum(
                pred, audio_batch, scales, overlap, step,
                touch=touch_batch, sound_id=sound_id_batch,
                return_losses=True, **loss_kwargs
            )
        except TypeError:
            loss, loss_details = model.loss_curriculum(
                pred, audio_batch, scales, overlap, step,
                touch=touch_batch, sound_id=sound_id_batch,
                return_losses=True
            )
            # Rebalance lin and rms terms manually
            loss = (
                loss
                - 100.0 * loss_details["lin"]
                - 1.0 * loss_details["rms"]
                + args.L * loss_details["lin"]
                + args.R * loss_details["rms"]
            )

        loss.backward()

        # Track pre-clip grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        grad_norm_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

        optimizer.step()

        step_record = {
            "step": step,
            "train_loss": float(loss.item()),
            "grad_norm": grad_norm_val,
            "lr": cur_lr,
        }
        if str(step) in eval_results_by_step:
            step_record["report_loss"] = eval_results_by_step[str(step)]["report_loss"]
        step_history.append(step_record)

        if step % 50 == 0:
            print(f"  Step {step:4d} | Train Loss: {loss.item():.4f} | Grad Norm: {grad_norm_val:.3f} | LR: {cur_lr:.2e}")

    # 3. Save CSV of onset rows for steps 800, 900, 1000
    csv_path = os.path.join(args.out_dir, f"ab_{args.arm}_s{args.seed}.csv")
    csv_fields = [
        "step", "sound", "curve", "chunk_offset", "onset_frame", "f0",
        "dpk_db", "abs_dpk_db", "dtpk_ms", "abs_dtpk_ms",
        "p_bump_db", "t_bump_db", "dpre_db", "dp300_db", "f0ratio"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in saved_onset_rows:
            writer.writerow(row)
    print(f"\nSaved onset details to: {csv_path} ({len(saved_onset_rows)} rows)")

    # 4. Compute pooled 800..1000 medians
    rows_800_1000 = [r for r in saved_onset_rows if r["step"] in [800, 900, 1000]]
    pooled_by_sound = {}
    pooled_strict_pass = 0
    for snd in sound_names:
        s_rows = [r for r in rows_800_1000 if r["sound"] == snd]
        if s_rows:
            m_dpk = float(np.median([r["dpk_db"] for r in s_rows]))
            m_abs_dpk = float(np.median([r["abs_dpk_db"] for r in s_rows]))
            m_dtpk = float(np.median([r["dtpk_ms"] for r in s_rows]))
            m_abs_dtpk = float(np.median([r["abs_dtpk_ms"] for r in s_rows]))
            m_pbump = float(np.median([r["p_bump_db"] for r in s_rows]))
            m_tbump = float(np.median([r["t_bump_db"] for r in s_rows]))
            m_dpre = float(np.median([r["dpre_db"] for r in s_rows]))
            m_dp300 = float(np.median([r["dp300_db"] for r in s_rows]))
            m_f0r = float(np.median([r["f0ratio"] for r in s_rows]))
            passed = bool(abs(m_dpk) <= 4.0 and m_abs_dtpk <= 20.0)
            if passed:
                pooled_strict_pass += 1
            pooled_by_sound[snd] = {
                "dpk": m_dpk,
                "med_abs_dpk": m_abs_dpk,
                "dtpk": m_dtpk,
                "med_abs_dtpk": m_abs_dtpk,
                "p_bump": m_pbump,
                "t_bump": m_tbump,
                "dpre": m_dpre,
                "dp300": m_dp300,
                "f0ratio": m_f0r,
                "n_onsets": len(s_rows),
                "pass": passed,
            }
        else:
            pooled_by_sound[snd] = {"n_onsets": 0, "pass": False}

    pooled_overall = {}
    if rows_800_1000:
        pooled_overall = {
            "dpk": float(np.median([r["dpk_db"] for r in rows_800_1000])),
            "med_abs_dpk": float(np.median([r["abs_dpk_db"] for r in rows_800_1000])),
            "dtpk": float(np.median([r["dtpk_ms"] for r in rows_800_1000])),
            "med_abs_dtpk": float(np.median([r["abs_dtpk_ms"] for r in rows_800_1000])),
            "p_bump": float(np.median([r["p_bump_db"] for r in rows_800_1000])),
            "t_bump": float(np.median([r["t_bump_db"] for r in rows_800_1000])),
            "dpre": float(np.median([r["dpre_db"] for r in rows_800_1000])),
            "dp300": float(np.median([r["dp300_db"] for r in rows_800_1000])),
            "f0ratio": float(np.median([r["f0ratio"] for r in rows_800_1000])),
            "total_onsets": len(rows_800_1000),
            "strict_pass_count": pooled_strict_pass,
        }

    # Step 1000 metrics
    step_1000_metrics = eval_results_by_step.get(str(args.steps), {})

    summary_out = {
        "args": vars(args),
        "batch_metadata": batch_meta,
        "step_history": step_history,
        "step_1000": step_1000_metrics,
        "steps_800_1000": {
            "overall": pooled_overall,
            "per_sound": pooled_by_sound,
            "strict_pass_count": pooled_strict_pass,
            "strict_pass_total": len(sound_names),
        },
    }

    summary_path = os.path.join(args.out_dir, f"ab_{args.arm}_s{args.seed}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
