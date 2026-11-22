#!/usr/bin/env python3
"""
make_onset_demo.py — render demo clips with ONLY touch events, 3s silence between.

For a given sound, picks up to 3 touch onsets (one per given curve), and renders:
  [1.0s before onset | 2.5s event]  + 3.0s silence,  repeated 3x
for the TARGET audio and for each model checkpoint. Output: one wav per model + target,
so you can listen and judge whether the attack is strong.

Usage:
  .venv/bin/python make_onset_demo.py --sound 01_canopy --curves 017 019 020 \
      --models stageB_v6 stageB_v7_control stageB_v7 stageB_v8b \
      --out listening_demo
"""
import argparse, json, os, re
import numpy as np
import torch
import soundfile as sf

from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5

SR = 48000
BLOCK = 480
PRE_S, EVENT_S, GAP_S = 1.0, 2.5, 3.0


def onset_frames_for(notes, key):
    return [i for i, v in enumerate(notes.get(key, [])) if v > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sound", default="01_canopy")
    ap.add_argument("--curves", nargs="+", default=["017", "019", "020"])
    ap.add_argument("--models", nargs="+", required=True, help="run dir names under runs/bakeoff_ddsp_poc")
    ap.add_argument("--root", default="bakeoff_data_aligned_v8")
    ap.add_argument("--notes", default="bakeoff_notes_v7.json")
    ap.add_argument("--out", default="listening_demo")
    ap.add_argument("--runroot", default="runs/bakeoff_ddsp_poc")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    notes = json.load(open(args.notes))
    ds = AlignedChunkDatasetV5(args.root, split="val", chunk_s=10.0, train=False, notes_json=args.notes)

    # collect up to 3 onsets: one per curve
    events = []  # (curve, frame)
    for c in args.curves:
        key = f"{args.sound}_proximity_curve_{c}"
        ons = onset_frames_for(notes, key)
        if ons:
            events.append((c, ons[0]))
        if len(events) >= 3:
            break
    assert len(events) >= 1, f"no onsets found for {args.sound} in curves {args.curves}"
    print(f"events: {[(c, f) for c, f in events]}")

    def get_item(curve):
        return ds.full(args.sound, curve)

    # target demo
    tgt_parts = []
    for c, f in events:
        item = get_item(c)
        audio = item["audio"]
        s = f * BLOCK
        a, b = s - int(PRE_S * SR), s + int(EVENT_S * SR)
        a = max(0, a); b = min(len(audio), b)
        tgt_parts.append(audio[a:b])
        tgt_parts.append(np.zeros(int(GAP_S * SR), dtype=audio.dtype))
    tgt = np.concatenate(tgt_parts)
    sf.write(os.path.join(args.out, f"{args.sound}_target.wav"), tgt.astype(np.float32), SR)
    print(f"wrote {args.out}/{args.sound}_target.wav  ({len(tgt)/SR:.1f}s)")

    device = args.device if torch.cuda.is_available() else "cpu"
    for mname in args.models:
        run_dir = os.path.join(args.runroot, mname)
        ckpt = os.path.join(run_dir, "state_final.pth")
        if not os.path.exists(ckpt):
            print(f"skip {mname}: no checkpoint yet")
            continue
        cfg_path = os.path.join(run_dir, "config.yaml")
        import yaml
        cfg = yaml.safe_load(open(cfg_path))
        model = SensorDDSP(
            f0_base_hz=list(cfg["model"]["f0_base_hz_override"]),
            n_sounds=cfg["model"]["n_sounds"],
            hidden_size=cfg["model"]["hidden_size"],
            n_harmonic=cfg["model"]["n_harmonic"],
            n_bands=cfg["model"]["n_bands"],
            onset_weighting=cfg["model"].get("onset_weighting", {"enabled": False}),
        )
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd if "model" not in sd else sd["model"])
        model.to(device).eval()

        parts = []
        for c, f in events:
            item = get_item(c)
            s = f * BLOCK
            a, b = max(0, s - int(PRE_S * SR)), min(len(item["audio"]), s + int(EVENT_S * SR))
            n_blocks = (b - a) // BLOCK
            sensor = torch.from_numpy(item["sensor"][a // BLOCK: a // BLOCK + n_blocks]).unsqueeze(0).unsqueeze(-1).to(device)
            touch1 = torch.from_numpy(item["touch"][a // BLOCK: a // BLOCK + n_blocks]).unsqueeze(0).to(device)  # (1, T, 3) features
            note_np = np.asarray(item.get("note", np.zeros_like(item["sensor"])), dtype=np.float32)
            note = torch.from_numpy(note_np[a // BLOCK: a // BLOCK + n_blocks]).unsqueeze(0).to(device)
            if note.dim() == 2: note = note.unsqueeze(-1)
            sid = torch.tensor([item["sound_id"]], device=device)
            with torch.no_grad():
                pred = model(sensor, touch1, sid, note=note).squeeze(0).squeeze(-1).cpu().numpy()
            pred = pred[: b - a]
            parts.append(pred)
            parts.append(np.zeros(int(GAP_S * SR), dtype=np.float32))
        clip = np.concatenate(parts)
        out = os.path.join(args.out, f"{args.sound}_{mname}.wav")
        sf.write(out, clip.astype(np.float32), SR)
        pk = np.abs(clip).max()
        print(f"wrote {out}  ({len(clip)/SR:.1f}s, peak={pk:.3f})")


if __name__ == "__main__":
    main()
