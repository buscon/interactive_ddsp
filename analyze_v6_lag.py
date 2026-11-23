#!/usr/bin/env python3
"""Measure the audible onset lag of touch events on Dataset v6 audio.

For every touch onset (first touch_time > 0 frame after a zero) of every npz
file, the script measures, on the audio target and on touch_layer, in a
[-50 ms, +600 ms] window around the touch_time onset:

  onset lag : first frame whose high-passed 2 ms RMS (1 ms hop) exceeds
              max(noise_floor_dB + X, -70 dBFS). Noise floor = median dB of the
              preceding 300 ms. X in {6, 12, 18} dB (12 dB is primary).
  peak lag  : time of the envelope maximum within [0, +600 ms].
  decay     : first frame after the peak that is 10 dB below the peak
              (searched up to +2000 ms; censored if never reached).

Two timeline modes are computed in one run:
  aligned   : array[n] = raw[n + round(offset_s*48000)] for audio and
              touch_layer (the training-target alignment from the brief).
  unshifted : raw arrays as stored.

Onsets are skipped (and counted) when the preceding 300 ms are not quiet: a
touch_time>0 frame in that window, or a max touch_layer envelope above
--quiet-db. The quiet decision uses the isolated touch layer, because the audio
target carries a continuously sounding drone and an absolute level test on it
would reject every onset. The same onset set is used for audio and touch_layer.

Output JSON: per mode -> counts, per layer -> per threshold -> per-sound
statistics, per-onset records, touch-layer check against the brief's reference
values, and a recommended attack_lag_s vector plus onset-weight kernel values.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

SR = 48000
SOUND_ORDER = [
    "01_canopy",
    "02_forestPad",
    "03_harmonicSwarm",
    "04_modularPercussions",
    "05_moss_and_mist",
]
# Reference touch-layer lags (ms) quoted in the brief, same sound order.
REFERENCE_TOUCH_LAG_MS = {
    "01_canopy": 3.0,
    "02_forestPad": 2.5,
    "03_harmonicSwarm": 43.0,
    "04_modularPercussions": 21.0,
    "05_moss_and_mist": 2.3,
}
DEFAULT_DATA_DIR = (
    "/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/"
    "bakeoff_data_aligned_v6"
)
HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("aligned", "unshifted")
LAYERS = ("audio", "touch_layer")

WIN = 96            # 2 ms RMS window at 48 kHz
HOP = 48            # 1 ms hop
PRE_MS = 300        # preceding window for noise floor / quiet check
POST_MS = 2000      # envelope extent after onset (decay search)
SEARCH_MIN_MS = -50
SEARCH_MAX_MS = 600
PEAK_MAX_MS = 600
DECAY_DB = 10.0
FLOOR_DBFS = -70.0
HP_HZ = 200.0
THRESH_OFFSETS_DB = (6.0, 12.0, 18.0)
PRIMARY_DB = 12.0
TOUCH_FRAME_DT = 0.01
# touch_time frame i holds the first in-touch value 0.01. Brief convention:
# onset = (i - 1) * 0.01 s (shift -1). Shift 0 gives i * 0.01 s, 10 ms later.
DEFAULT_ONSET_FRAME_SHIFT = -1


def centre_ms(k):
    """Envelope frame k (1 ms hop, 2 ms window) has window-centre time k - PRE_MS + 1 ms."""
    return k - PRE_MS + 1


def hp_filter(x):
    sos = butter(4, HP_HZ, btype="highpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x)


def envelope_db(seg):
    frames = np.lib.stride_tricks.sliding_window_view(seg, WIN)[::HOP]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def measure_onset(hp, on_sample, quiet_db, pre_touch):
    """Measure one onset on one high-passed signal.

    Returns (status, result); status in {ok, not_quiet, edge_pre, edge_post}.
    """
    pre = int(PRE_MS * SR / 1000)
    post = int(POST_MS * SR / 1000)
    if on_sample - pre < 0:
        return "edge_pre", None
    if on_sample + post > len(hp):
        return "edge_post", None
    db = envelope_db(hp[on_sample - pre: on_sample + post])
    k_all = np.arange(len(db))
    cms = centre_ms(k_all)

    pre_mask = (cms >= -PRE_MS) & (cms <= -1)
    noise_db = float(np.median(db[pre_mask]))
    pre_max_db = float(np.max(db[pre_mask]))
    if pre_touch or pre_max_db > quiet_db:
        return "not_quiet", {"noise_db": noise_db, "pre_max_db": pre_max_db}

    search = (cms >= SEARCH_MIN_MS) & (cms <= SEARCH_MAX_MS)
    kp_idx = np.where((cms >= 0) & (cms <= PEAK_MAX_MS))[0]
    kp = kp_idx[np.argmax(db[kp_idx])]
    peak_db = float(db[kp])
    peak_ms = float(cms[kp])
    after = np.where(k_all > kp)[0]
    hit = after[db[after] <= peak_db - DECAY_DB]
    decay_ms = float(cms[hit[0]]) if len(hit) else float("nan")

    out = {"noise_db": noise_db, "pre_max_db": pre_max_db, "peak_db": peak_db,
           "peak_lag_ms": peak_ms, "decay_ms": decay_ms,
           "decay_after_peak_ms": (decay_ms - peak_ms) if not np.isnan(decay_ms) else float("nan"),
           "thr": {}}
    for off in THRESH_OFFSETS_DB:
        thr = max(noise_db + off, FLOOR_DBFS)
        idx = np.where(search & (db > thr))[0]
        out["thr"][str(int(off))] = {
            "thr_db": thr,
            "onset_lag_ms": float(cms[idx[0]]) if len(idx) else float("nan"),
        }
    return "ok", out


def stats(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None,
                "min": None, "max": None}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": round(float(med), 2), "q25": round(float(q25), 2),
            "q75": round(float(q75), 2), "iqr": round(float(q75 - q25), 2),
            "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)}


def analyse(files, mode, quiet_db, onset_frame_shift):
    counts = {s: {"files": 0, "onsets_total": 0, "skipped_not_quiet": 0,
                  "skipped_edge": 0, "onsets_used": 0} for s in SOUND_ORDER}
    records = {layer: {s: [] for s in SOUND_ORDER} for layer in LAYERS}
    noise_db = {s: [] for s in SOUND_ORDER}
    split_counts = {}

    for f in files:
        d = np.load(f)
        sound = str(d["sound"])
        split = str(d["split"])
        split_counts[split] = split_counts.get(split, 0) + 1
        counts[sound]["files"] += 1
        shift = int(round(float(d["offset_s"]) * SR)) if mode == "aligned" else 0
        tt = d["touch_time"]
        raw = {
            "audio": d["audio"].astype(np.float64)[shift:] / 32768.0,
            "touch_layer": d["touch_layer"].astype(np.float64)[shift:] / 32768.0,
        }
        hp = {layer: hp_filter(raw[layer]) for layer in LAYERS}

        on_idx = np.where((tt > 0) & (np.concatenate(([0.0], tt[:-1])) <= 0))[0]
        for i in on_idx:
            counts[sound]["onsets_total"] += 1
            t_on = (i + onset_frame_shift) * TOUCH_FRAME_DT
            on_sample = int(round(t_on * SR))
            pre_touch = bool(np.any(tt[max(0, i - int(PRE_MS / 10)): i] > 0))
            # Quiet/edge decision on touch_layer; audio uses the same onset set.
            status, res_touch = measure_onset(hp["touch_layer"], on_sample, quiet_db, pre_touch)
            if status != "ok":
                key = "skipped_not_quiet" if status == "not_quiet" else "skipped_edge"
                counts[sound][key] += 1
                continue
            _, res_audio = measure_onset(hp["audio"], on_sample, np.inf, False)
            counts[sound]["onsets_used"] += 1
            noise_db[sound].append(res_audio["noise_db"])
            for layer, res in (("audio", res_audio), ("touch_layer", res_touch)):
                rec = {"file": os.path.basename(f), "split": split, "touch_index": int(i),
                       "onset_s": round(t_on, 4), "noise_db": round(res["noise_db"], 2),
                       "peak_db": round(res["peak_db"], 2), "peak_lag_ms": res["peak_lag_ms"],
                       "decay_ms": res["decay_ms"], "decay_after_peak_ms": res["decay_after_peak_ms"]}
                for off in THRESH_OFFSETS_DB:
                    rec[f"onset_lag_ms_{int(off)}"] = res["thr"][str(int(off))]["onset_lag_ms"]
                records[layer][sound].append(rec)

    results = {}
    for layer in LAYERS:
        results[layer] = {}
        for off in THRESH_OFFSETS_DB:
            o = int(off)
            per_sound = {}
            all_onset, all_peak, all_decay = [], [], []
            for s in SOUND_ORDER:
                recs = records[layer][s]
                ol = [r[f"onset_lag_ms_{o}"] for r in recs]
                pl = [r["peak_lag_ms"] for r in recs]
                dl = [r["decay_ms"] for r in recs]
                all_onset += ol
                all_peak += pl
                all_decay += dl
                per_sound[s] = {
                    "n": len(recs),
                    "onset_not_detected": int(sum(np.isnan(x) for x in ol)),
                    "onset_lag_ms": stats(ol),
                    "peak_lag_ms": stats(pl),
                    "decay_ms": stats(dl),
                    "decay_after_peak_ms": stats([r["decay_after_peak_ms"] for r in recs]),
                    "decay_censored": int(sum(np.isnan(x) for x in dl)),
                }
            results[layer][str(o)] = {
                "per_sound": per_sound,
                "all_sounds": {"n": len(all_onset), "onset_lag_ms": stats(all_onset),
                               "peak_lag_ms": stats(all_peak), "decay_ms": stats(all_decay)},
            }

    def med(layer, off, s, key):
        v = results[layer][str(int(off))]["per_sound"][s][key]["median"]
        return np.nan if v is None else v

    touch_check = {}
    for s in SOUND_ORDER:
        m = med("touch_layer", PRIMARY_DB, s, "onset_lag_ms")
        touch_check[s] = {"measured_onset_median_ms": None if np.isnan(m) else round(float(m), 2),
                          "reference_ms": REFERENCE_TOUCH_LAG_MS[s],
                          "difference_ms": None if np.isnan(m) else round(float(m) - REFERENCE_TOUCH_LAG_MS[s], 2)}

    sound_peak = np.array([med("audio", PRIMARY_DB, s, "peak_lag_ms") for s in SOUND_ORDER])
    peak_median_all = float(np.nanmedian(sound_peak)) if np.any(~np.isnan(sound_peak)) else float("nan")
    peak_ms = int(round(peak_median_all / 10.0) * 10) if np.isfinite(peak_median_all) else None
    attack_lag_s = [None if np.isnan(med("audio", PRIMARY_DB, s, "onset_lag_ms"))
                    else round(round(float(med("audio", PRIMARY_DB, s, "onset_lag_ms"))) / 1000.0, 3)
                    for s in SOUND_ORDER]

    sensitivity = {}
    for off in THRESH_OFFSETS_DB:
        o = str(int(off))
        sensitivity[o] = {
            "attack_lag_s": [None if np.isnan(med("audio", off, s, "onset_lag_ms"))
                             else round(round(float(med("audio", off, s, "onset_lag_ms"))) / 1000.0, 3)
                             for s in SOUND_ORDER],
            "peak_median_of_sounds_ms": round(float(np.nanmedian(
                [med("audio", off, s, "peak_lag_ms") for s in SOUND_ORDER])), 2),
            "onset_detected_fraction": {
                s: round(1 - results["audio"][o]["per_sound"][s]["onset_not_detected"]
                         / max(1, results["audio"][o]["per_sound"][s]["n"]), 3)
                for s in SOUND_ORDER},
        }

    finite = np.array([r[f"onset_lag_ms_{int(PRIMARY_DB)}"] for s in SOUND_ORDER for r in records["audio"][s]
                       if not np.isnan(r[f"onset_lag_ms_{int(PRIMARY_DB)}"])], dtype=float)
    p5 = float(np.percentile(finite, 5)) if finite.size else float("nan")

    return {
        "split_counts": split_counts,
        "counts": counts,
        "noise_floor_db_median_per_sound": {
            s: (round(float(np.median(noise_db[s])), 2) if noise_db[s] else None) for s in SOUND_ORDER},
        "results": results,
        "touch_layer_check_vs_brief": touch_check,
        "sensitivity": sensitivity,
        "recommendation": {
            "attack_lag_s": attack_lag_s,
            "sound_order": SOUND_ORDER,
            "onset_weight_kernel": {
                "pre_ms": 25,
                "pre_ms_note": f"midpoint of the 20-30 ms range; P5 of measured audio onset lags "
                               f"(+{PRIMARY_DB:g} dB) is {p5:.1f} ms",
                "peak_ms": peak_ms,
                "peak_ms_note": f"median over sounds of per-sound median peak lag "
                                f"({peak_median_all:.2f} ms), rounded to 10 ms",
                "window_ms": 400,
            },
            "per_sound_peak_vs_peak_ms": {
                s: {"measured_peak_median_ms": None if np.isnan(sound_peak[i]) else round(float(sound_peak[i]), 2),
                    "difference_ms": None if (np.isnan(sound_peak[i]) or peak_ms is None)
                    else round(float(sound_peak[i]) - peak_ms, 2)}
                for i, s in enumerate(SOUND_ORDER)},
        },
        "onsets": records,
    }


def clean(x):
    """Replace NaN/inf with None and numpy scalars with Python scalars, recursively."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=os.path.join(HERE, "v6_lag_measurement.json"))
    ap.add_argument("--quiet-db", type=float, default=-50.0,
                    help="skip onset if max touch_layer envelope dB in preceding 300 ms exceeds this")
    ap.add_argument("--onset-frame-shift", type=int, default=DEFAULT_ONSET_FRAME_SHIFT,
                    help="onset = (i + shift) * 0.01 s for first touch_time>0 frame i (default -1)")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma list of timeline modes: aligned, unshifted")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not files:
        sys.exit(f"no npz files in {args.data_dir}")
    out_modes = {}
    for mode in [m for m in args.modes.split(",") if m]:
        print(f"== mode {mode}: {len(files)} files", file=sys.stderr)
        out_modes[mode] = analyse(files, mode, args.quiet_db, args.onset_frame_shift)
        for s in SOUND_ORDER:
            print(f"  {mode} {s}: {out_modes[mode]['counts'][s]}", file=sys.stderr)

    out = {
        "method": {
            "data_dir": args.data_dir,
            "n_files": len(files),
            "modes": {
                "aligned": "array[n] = raw[n + round(offset_s*48000)] (brief's alignment)",
                "unshifted": "raw arrays as stored",
            },
            "touch_onset": f"first touch_time>0 frame i after a zero; t = (i + {args.onset_frame_shift}) * 0.01 s",
            "envelope": {"highpass_hz": HP_HZ, "filter": "butter4 sosfiltfilt",
                         "window_ms": 2.0, "hop_ms": 1.0, "db": "20log10(rms)",
                         "time_reference": "window centre, ms relative to touch onset"},
            "noise_floor": "median dB over preceding 300 ms",
            "threshold": "max(noise_db + X, -70 dBFS), X in {6, 12, 18}; primary X=12",
            "search_window_ms": [SEARCH_MIN_MS, SEARCH_MAX_MS],
            "peak_window_ms": [0, PEAK_MAX_MS],
            "decay": "first frame after peak at peak_db - 10 dB, searched to +2000 ms",
            "quiet_rule": f"skip if touch_time>0 in preceding 300 ms, or max touch_layer "
                          f"envelope dB in preceding 300 ms > {args.quiet_db} dBFS",
        },
        "modes": out_modes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(clean(out), fh, indent=1, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
