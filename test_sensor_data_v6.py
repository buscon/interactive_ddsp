"""
Tests for the v6 loader path of engine_ddsp/sensor_data_v5.py.

Run: python3 test_sensor_data_v6.py   (or: pytest test_sensor_data_v6.py)
"""
import contextlib
import json
import os
import tempfile

import numpy as np

from engine_ddsp.sensor_data_v5 import AlignedChunkDatasetV5 as DS

V6_ROOT = DS.V6_PROVENANCE_ROOT
OLD_ROOT = DS.OLD_PROVENANCE_ROOT
V6_DIR = "/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_data_aligned_v6"
V6_NOTES = "/home/marcello/sentire/Recordings/Dataset_v6_20261122_aligned/bakeoff_notes_v6.json"
OLD_DIR = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned_v5_03"

RAW_T = 50          # raw sensor frames per synthetic file
OFFSET_S = 0.3      # -> shift = 14400 samples = 30 frames
SHIFT = 14400
T_EFF = (RAW_T * 480 - SHIFT) // 480   # = 20
CLICK_K = 5         # aligned sample index of the click (sensor clock)


@contextlib.contextmanager
def whitelist(n_files_for_v6=None, n_files_for_old=None):
    """Temporarily change the class whitelist (synthetic datasets have few files)."""
    orig = dict(DS.PROVENANCE_ROOTS)
    try:
        if n_files_for_v6 is not None:
            DS.PROVENANCE_ROOTS[V6_ROOT] = n_files_for_v6
        if n_files_for_old is not None:
            DS.PROVENANCE_ROOTS[OLD_ROOT] = n_files_for_old
        yield
    finally:
        DS.PROVENANCE_ROOTS.clear()
        DS.PROVENANCE_ROOTS.update(orig)


def write_dataset(root, source_dir, n_files=3, with_offset=True, raw_t=RAW_T):
    """Write a tiny index.json + npz dataset. File 0 is 'val', the rest 'train'.
    Audio has a single click at raw sample CLICK_K + SHIFT (audio lags sensor)."""
    os.makedirs(root, exist_ok=True)
    files = []
    n = raw_t * 480
    for i in range(n_files):
        curve = f"{i + 1:03d}"
        split = "val" if i == 0 else "train"
        audio = np.zeros(n, dtype=np.int16)
        audio[CLICK_K + SHIFT] = 20000
        sensor = (np.arange(raw_t, dtype=np.float32) + 1000.0 * i)
        touch = np.linspace(0.0, 1.0, raw_t, dtype=np.float32)
        path = os.path.join(root, f"01_canopy_proximity_curve_{curve}.npz")
        kw = dict(audio=audio, sensor=sensor, touch_time=touch,
                  sound=np.array("01_canopy"), curve=np.array(curve),
                  split=np.array(split), start_s=np.float64(0.0))
        if with_offset:
            kw["offset_s"] = np.float64(OFFSET_S)
        np.savez(path, **kw)
        files.append({"path": path, "sound": "01_canopy", "curve": curve,
                      "split": split, "n_samples": n, "T": raw_t})
    index = {
        "source_dir": source_dir,
        "n_files": n_files,
        "sound_ids": {"01_canopy": 0},
        "train_curves": [f["curve"] for f in files[1:]],
        "val_curves": [files[0]["curve"]],
        "files": files,
    }
    with open(os.path.join(root, "index.json"), "w") as fh:
        json.dump(index, fh)
    notes = os.path.join(root, "notes.json")
    with open(notes, "w") as fh:
        json.dump({}, fh)
    return notes


def _tmp():
    return tempfile.mkdtemp(prefix="v6loader_")


# ---------------------------------------------------------------- (a) shift + trim
def test_shift_direction_and_trim():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    with whitelist(n_files_for_v6=3):
        ds = DS(root, "val", chunk_s=0.1, train=False, notes_json=notes, align_audio=True)
    assert len(ds.loaded) == 1
    fd = ds.full("01_canopy", "001")
    # click at raw sample CLICK_K + SHIFT must land at aligned sample CLICK_K
    assert int(np.argmax(np.abs(fd["audio"]))) == CLICK_K
    assert abs(fd["audio"][CLICK_K] - 20000 / 32768.0) < 1e-6
    # T_eff and no zero-padded tail: lengths consistent, audio = 480 * frames
    assert T_EFF == 20
    assert len(fd["sensor"]) == T_EFF
    assert len(fd["touch_raw"]) == T_EFF
    assert len(fd["note"]) == T_EFF
    assert len(fd["audio"]) == 480 * len(fd["sensor"]) == 480 * len(fd["touch_raw"])
    # sensor/touch are NOT shifted: first T_EFF raw frames unchanged
    raw_sensor = np.load(os.path.join(root, "01_canopy_proximity_curve_001.npz"))["sensor"]
    assert np.array_equal(fd["sensor"], raw_sensor[:T_EFF])
    # audio tail: last sample of aligned audio is raw sample (T_EFF*480 - 1 + SHIFT)
    raw_audio = np.load(os.path.join(root, "01_canopy_proximity_curve_001.npz"))["audio"]
    assert np.array_equal(
        fd["audio"] * 32768.0,
        raw_audio[SHIFT:SHIFT + T_EFF * 480].astype(np.float32))


def test_chunks_no_padding_and_epoch_len_consistent():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    chunk_s = 0.1  # 10 frames, 4800 samples
    with whitelist(n_files_for_v6=3):
        ds = DS(root, "train", chunk_s=chunk_s, train=False, notes_json=notes, align_audio=True)
    per_file = (T_EFF - 10) // 10 + 1   # = 2
    assert len(ds.loaded) == 2
    assert len(ds) == 2 * per_file == 4
    for idx in range(len(ds)):
        item = ds[idx]
        assert item["audio"].shape == (4800,)
        assert item["sensor"].shape == (10,)
        assert item["touch"].shape == (10, 3)
        assert item["offset"] + 10 <= T_EFF
        fd = ds.full("01_canopy", item["curve"])
        o = item["offset"]
        assert np.array_equal(item["audio"], fd["audio"][480 * o:480 * (o + 10)])
        assert np.array_equal(item["sensor"], fd["sensor"][o:o + 10])
    # random mode: every served chunk is full length and inside T_eff
    with whitelist(n_files_for_v6=3):
        dsr = DS(root, "train", chunk_s=chunk_s, train=True, notes_json=notes, seed=1,
                 align_audio=True)
    for idx in range(50):
        item = dsr[idx]
        assert item["audio"].shape == (4800,)
        assert item["offset"] + 10 <= T_EFF


def test_epoch_len_matches_deterministic_chunk_count():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    # chunk_s 0.2 -> 20 frames: exactly one chunk per file at T_EFF=20
    with whitelist(n_files_for_v6=3):
        ds = DS(root, "train", chunk_s=0.2, train=False, notes_json=notes, align_audio=True)
    assert len(ds) == 2
    assert ds[0]["audio"].shape == (9600,)
    assert ds[1]["audio"].shape == (9600,)
    # chunk_s 0.3 -> 30 frames > T_EFF: no chunks at all
    with whitelist(n_files_for_v6=3):
        ds0 = DS(root, "train", chunk_s=0.3, train=False, notes_json=notes, align_audio=True)
    assert len(ds0) == 0


# ---------------------------------------------------------------- (b) missing offset_s
def test_v6_missing_offset_loads_unshifted():
    """v6 without offset_s loads fine: no shift, no trim (T = index T)."""
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3, with_offset=False)
    for align in (False, True):  # without offset_s there is nothing to shift
        with whitelist(n_files_for_v6=3):
            ds = DS(root, "val", chunk_s=0.1, train=False, notes_json=notes,
                    align_audio=align)
        fd = ds.full("01_canopy", "001")
        assert len(fd["sensor"]) == RAW_T
        assert len(fd["audio"]) == RAW_T * 480
        assert int(np.argmax(np.abs(fd["audio"]))) == CLICK_K + SHIFT


# ---------------------------------------------------------------- default path (no shift)
def test_default_v6_no_shift_no_trim():
    """Default align_audio=False on v6: audio stored as-is, T unchanged."""
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    with whitelist(n_files_for_v6=3):
        ds = DS(root, "val", chunk_s=0.1, train=False, notes_json=notes)
    assert ds.align_audio is False
    fd = ds.full("01_canopy", "001")
    assert len(fd["sensor"]) == RAW_T
    assert len(fd["audio"]) == RAW_T * 480
    assert int(np.argmax(np.abs(fd["audio"]))) == CLICK_K + SHIFT  # not moved
    raw = np.load(os.path.join(root, "01_canopy_proximity_curve_001.npz"))
    assert np.array_equal(fd["audio"], raw["audio"].astype(np.float32) / 32768.0)


# ---------------------------------------------------------------- whitelist
def test_whitelist_rejects_unknown_and_wrong_n():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    # real whitelist expects 100 files for the v6 root -> 3 must be refused
    raised = False
    try:
        DS(root, "train", chunk_s=0.1, train=False, notes_json=notes)
    except ValueError as e:
        raised = "Provenance mismatch" in str(e)
    assert raised
    root2 = os.path.join(tmp, "ds2")
    notes2 = write_dataset(root2, "/somewhere/else", n_files=3)
    raised = False
    try:
        DS(root2, "train", chunk_s=0.1, train=False, notes_json=notes2)
    except ValueError as e:
        raised = "Provenance mismatch" in str(e)
    assert raised


# ---------------------------------------------------------------- (c) regression
def test_align_false_is_unshifted():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, V6_ROOT, n_files=3)
    with whitelist(n_files_for_v6=3):
        ds = DS(root, "val", chunk_s=0.1, train=False, notes_json=notes,
                align_audio=False)
    fd = ds.full("01_canopy", "001")
    raw = np.load(os.path.join(root, "01_canopy_proximity_curve_001.npz"))
    assert len(fd["audio"]) == RAW_T * 480
    assert np.array_equal(fd["audio"], raw["audio"].astype(np.float32) / 32768.0)
    assert np.array_equal(fd["sensor"], raw["sensor"])
    assert int(np.argmax(np.abs(fd["audio"]))) == CLICK_K + SHIFT


def test_old_provenance_no_offset_unchanged():
    tmp = _tmp()
    root = os.path.join(tmp, "ds")
    notes = write_dataset(root, OLD_ROOT, n_files=3, with_offset=False)
    outs = {}
    for align in (True, False):
        with whitelist(n_files_for_old=3):
            ds = DS(root, "train", chunk_s=0.1, train=False, notes_json=notes,
                    align_audio=align)
            chunks = [ds[i] for i in range(len(ds))]
            fd = ds.full("01_canopy", "002")
        outs[align] = (chunks, fd)
    (ca, fa), (cb, fb) = outs[True], outs[False]
    assert len(ca) == len(cb) == 10  # 2 train files x (50 - 10) // 10 + 1 = 5 chunks
    for x, y in zip(ca, cb):
        assert np.array_equal(x["audio"], y["audio"])
        assert np.array_equal(x["sensor"], y["sensor"])
        assert x["offset"] == y["offset"]
    assert np.array_equal(fa["audio"], fb["audio"])
    assert np.array_equal(fa["sensor"], fb["sensor"])
    # old data: T unchanged (full raw length)
    assert len(fa["sensor"]) == RAW_T


def test_old_real_dataset_align_true_equals_false():
    """Uses the real old-provenance dataset if present (read only)."""
    if not os.path.exists(os.path.join(OLD_DIR, "index.json")):
        print("  (skip: old dataset not present)")
        return
    with open(os.path.join(OLD_DIR, "index.json")) as fh:
        if json.load(fh)["source_dir"] != OLD_ROOT:
            print("  (skip: old dataset has a different source_dir)")
            return
    outs = {}
    for align in (True, False):
        ds = DS(OLD_DIR, "val", chunk_s=10.0, train=False, align_audio=align)
        outs[align] = ([ds[i] for i in range(min(3, len(ds)))],
                       ds.full(ds.loaded[0]["sound"], ds.loaded[0]["curve"]))
    (ca, fa), (cb, fb) = outs[True], outs[False]
    for x, y in zip(ca, cb):
        assert np.array_equal(x["audio"], y["audio"])
        assert np.array_equal(x["sensor"], y["sensor"])
    assert np.array_equal(fa["audio"], fb["audio"])
    assert np.array_equal(fa["sensor"], fb["sensor"])


# ---------------------------------------------------------------- (d) real v6 smoke
def test_real_v6_smoke():
    if not os.path.exists(os.path.join(V6_DIR, "index.json")):
        print("  (skip: real v6 dataset not present)")
        return
    ds = DS(V6_DIR, "val", chunk_s=10.0, train=False, notes_json=V6_NOTES)
    assert ds.align_audio is False
    assert len(ds) > 0
    assert len(ds) == sum(DS._count_chunks(ds, f) for f in ds.loaded)
    for i in range(len(ds)):
        item = ds[i]
        assert item["audio"].shape == (ds.chunk_samples,)
        assert item["sensor"].shape == (ds.chunk_frames,)
        assert item["note"].shape == (ds.chunk_frames,)
    fd = ds.full("01_canopy", "004")
    n = len(fd["sensor"])
    assert len(fd["touch_raw"]) == n
    assert len(fd["note"]) == n
    assert fd["audio"].dtype == np.float32
    # v6 audio carries a sub-frame remainder (npz audio is not an exact multiple
    # of 480 for this file); the default path does not trim it.
    assert 0 <= len(fd["audio"]) - 480 * n < 480

    # default loader: audio equals the npz audio exactly (no shift)
    raw = np.load(os.path.join(V6_DIR, "01_canopy_proximity_curve_004.npz"))
    raw_audio = raw["audio"]
    assert len(fd["audio"]) == len(raw_audio)
    assert np.array_equal(fd["audio"], raw_audio.astype(np.float32) / 32768.0)
    assert np.array_equal(fd["touch_raw"], raw["touch_time"])

    # data contract: touch_layer onset is 0..10 ms after the touch_time onset
    tt = raw["touch_time"]
    tl = raw["touch_layer"].astype(np.float64)
    edges = np.where((tt[1:] > 0) & (tt[:-1] <= 0))[0] + 1
    assert len(edges) >= 3
    for f in edges[:3]:
        lo, hi = f * 480 - 4800, f * 480 + 4800   # +-100 ms window
        w = np.abs(tl[lo:hi])
        k = int(np.where(w >= 0.1 * w.max())[0][0])  # first sample >= 10% of peak
        onset_ms = (lo + k - f * 480) / 48.0
        assert 0.0 <= onset_ms <= 10.0, (f, onset_ms)


TESTS = [
    test_shift_direction_and_trim,
    test_chunks_no_padding_and_epoch_len_consistent,
    test_epoch_len_matches_deterministic_chunk_count,
    test_v6_missing_offset_loads_unshifted,
    test_default_v6_no_shift_no_trim,
    test_whitelist_rejects_unknown_and_wrong_n,
    test_align_false_is_unshifted,
    test_old_provenance_no_offset_unchanged,
    test_old_real_dataset_align_true_equals_false,
    test_real_v6_smoke,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    raise SystemExit(1 if failed else 0)
