"""
SensorDDSP dataset and f0 estimation utilities for v5.
Loads aligned proximity sensor + audio data from bakeoff_data_aligned_v5 (old
provenance) or bakeoff_data_aligned_v6 (v6 provenance).

Audio alignment:
- v6 npz files: audio is ALREADY on the sensor clock (sensor = ch4 of the same
  recording; the touch-layer onset sits +2 ms after the touch_time onset without
  any shift). offset_s is metadata only (recording time vs. planned curve). The
  default loader (align_audio=False) therefore applies no shift and no trim:
  T = index T, audio/sensor/touch_time are used exactly as stored.
- align_audio=True is LEGACY-ONLY, for data whose audio lags the sensor clock
  by offset_s. It is NOT for v6. When requested and the npz has offset_s:
  aligned_audio[n] = audio[n + round(offset_s * 48000)], the sample tail with no
  matching frame is trimmed (no zero padding), and sensor/touch_time/notes are
  trimmed to T_eff = floor((n_samples - shift) / 480).
"""
import json
import os
import random
import numpy as np
import json as _json

_NOTES = None
_NOTES_PATH = None

def _set_notes_path(path: str):
    """Set the notes JSON path for reloading."""
    global _NOTES_PATH
    _NOTES_PATH = path
    global _NOTES
    _NOTES = None  # Force reload

def _notes():
    global _NOTES, _NOTES_PATH
    if _NOTES is None:
        import os
        if _NOTES_PATH is None:
            _NOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "bakeoff_notes_v5.json")
        _NOTES = _json.load(open(_NOTES_PATH)) if os.path.exists(_NOTES_PATH) else {}
    return _NOTES

import librosa
import torch
from torch.utils.data import Dataset


class AlignedChunkDatasetV5(Dataset):
    """
    Dataset of aligned sensor + audio chunks (v5 dataset).
    
    Data format:
    - index.json: source_dir, n_files, sound_ids dict, train_curves, val_curves, files list
    - each .npz: audio (int16, n_samples,), sensor (float32, T,), touch_time (float32, T,)
      with T = n_samples // 480 (100 Hz frame rate)
    
    Chunking:
    - chunk_s seconds -> F = int(chunk_s * 100) frames
    - audio chunk: audio[480*offset : 480*(offset+F)]
    - sensor chunk: sensor[offset:offset+F]
    - touch chunk: touch_time[offset:offset+F]
    
    Notes:
    - Onset labels are loaded from bakeoff_notes_v5.json or custom notes_json path
    """
    
    OLD_PROVENANCE_ROOT = "/home/marcello/sentire/Recordings/Dataset_20261121_134808"
    V6_PROVENANCE_ROOT = "/home/marcello/sentire/Recordings/Dataset_v6_20261122"
    # Whitelist of index.json source_dir -> expected n_files
    PROVENANCE_ROOTS = {
        OLD_PROVENANCE_ROOT: 100,
        V6_PROVENANCE_ROOT: 80,  # 4 sounds × 20 curves (excluded 04_modularPercussions)
    }
    AUDIO_SR = 48000

    def __init__(self, root, split, chunk_s=10.0, sounds=None, curves=None,
                 epoch_len=None, seed=0, train=True, notes_json=None,
                 align_audio=False, use_npz_note_hz=False):
        """
        Args:
            root: path to bakeoff_data_aligned_v5 or bakeoff_data_aligned_v6
            split: "train" or "val"
            chunk_s: chunk duration in seconds (default 10.0)
            sounds: list of sound names to include, or None for all
            curves: list of curve IDs to include, or None for all in split
            epoch_len: number of items per epoch (default: n_files * 20 for train)
            seed: random seed for shuffling
            train: True for random chunks, False for deterministic non-overlapping
            notes_json: path to notes JSON file (default: bakeoff_notes_v5.json)
            align_audio: default False (v6 audio is already on the sensor clock;
                no shift, no trim). True is legacy-only: shift audio by offset_s
                and trim the tail when the npz has offset_s (see module docstring).
                Never use True for v6.
            use_npz_note_hz: default False. If True and npz contains 'note_hz',
                use it instead of notes_json labels.
        """
        # Set notes path if provided
        if notes_json is not None:
            _set_notes_path(notes_json)
        
        self.root = root
        self.split = split
        self.chunk_s = chunk_s
        self.train = train
        self.align_audio = align_audio
        self.use_npz_note_hz = bool(use_npz_note_hz)

        # Load index
        index_path = os.path.join(root, "index.json")
        with open(index_path) as f:
            self.index = json.load(f)

        # Provenance guard (whitelist of source dirs)
        source_dir = self.index["source_dir"]
        if source_dir not in self.PROVENANCE_ROOTS:
            raise ValueError(
                f"Provenance mismatch: expected one of {sorted(self.PROVENANCE_ROOTS)}, "
                f"got {source_dir}"
            )
        expected_n = self.PROVENANCE_ROOTS[source_dir]
        if self.index["n_files"] != expected_n:
            raise ValueError(
                f"Provenance mismatch: expected {expected_n} files, "
                f"got {self.index['n_files']}"
            )
        
        # Filter files by split
        self.files = [f for f in self.index["files"] if f["split"] == split]
        
        # Apply sound/curve filters
        if sounds is not None:
            sound_ids = self.index["sound_ids"]
            self.files = [f for f in self.files if f["sound"] in sounds]
        
        if curves is not None:
            self.files = [f for f in self.files if f["curve"] in curves]
        
        # Build sound_id mapping using the integer IDs from index.json
        self.sound_ids = self.index["sound_ids"]
        self.sound_id_map = self.sound_ids.copy()
        
        # Load all selected files into RAM
        self._load_files()
        
        # Setup epoch length
        self.chunk_frames = int(chunk_s * 100)  # 100 Hz frame rate
        self.chunk_samples = self.chunk_frames * 480
        
        if epoch_len is None:
            if train:
                self.epoch_len = len(self.files) * 20
            else:
                # Deterministic: cover each file with non-overlapping chunks
                # (uses the loaded, possibly trimmed T_eff, not index.json T)
                self.epoch_len = sum(self._count_chunks(f) for f in self.loaded)
        else:
            self.epoch_len = epoch_len
        
        self.rng = random.Random(seed)
        
        # Print info
        print(f"Dataset: {root}")
        print(f"Files: {len(self.files)}, split={split}")
        curve_ids = sorted(set(f["curve"] for f in self.files))
        print(f"Curves: {curve_ids}")
        sounds_used = sorted(set(f["sound"] for f in self.files))
        print(f"Sounds: {sounds_used}")
    
    def _load_files(self):
        """Load all selected .npz files into RAM."""
        self.loaded = []
        for f in self.files:
            path = f["path"]
            data = np.load(path)
            has_offset = "offset_s" in data.files
            audio_raw = data["audio"]
            sensor = data["sensor"].astype(np.float32)
            touch_time = data["touch_time"].astype(np.float32)
            note_hz = data["note_hz"].astype(np.float32) if "note_hz" in data.files else None
            T = int(f["T"])
            if self.align_audio and has_offset:
                # Audio lags the sensor clock by offset_s: drop the first
                # `shift` samples, then trim to whole sensor frames (no padding).
                n_samples_raw = int(audio_raw.shape[0])
                shift = int(round(float(data["offset_s"]) * self.AUDIO_SR))
                T_eff = (n_samples_raw - shift) // 480
                if T_eff > len(sensor) or T_eff > len(touch_time):
                    raise ValueError(
                        f"Sensor shorter than aligned audio: {path} "
                        f"(T_eff={T_eff}, sensor={len(sensor)}, touch={len(touch_time)})"
                    )
                audio_raw = audio_raw[shift:shift + T_eff * 480]
                sensor = sensor[:T_eff]
                touch_time = touch_time[:T_eff]
                if note_hz is not None:
                    note_hz = note_hz[:T_eff]
                T = T_eff
            if note_hz is not None:
                note_hz = note_hz[:T]
            # audio: int16 -> float32 normalized
            audio = audio_raw.astype(np.float32) / 32768.0
            sound_id = self.sound_id_map[f["sound"]]
            self.loaded.append({
                "audio": audio,
                "sensor": sensor,
                "touch_time": touch_time,
                "note_hz": note_hz,
                "sound_id": sound_id,
                "sound": f["sound"],
                "curve": f["curve"],
                "n_samples": int(f["n_samples"]),
                "T": T,
            })

    def _count_chunks(self, file_entry):
        """Count how many non-overlapping chunks fit in a file.
        file_entry: a self.loaded dict (uses its effective T)."""
        T = file_entry["T"]
        if T < self.chunk_frames:
            return 0
        return (T - self.chunk_frames) // self.chunk_frames + 1
    
    def __len__(self):
        return self.epoch_len
    
    def __getitem__(self, idx):
        """
        Get a chunk.
        train=True: random file + random offset
        train=False: deterministic, non-overlapping chunks
        """
        if self.train:
            # Random file
            file_idx = self.rng.randrange(len(self.loaded))
            file_data = self.loaded[file_idx]
            # Random offset
            max_offset = file_data["T"] - self.chunk_frames
            if max_offset <= 0:
                # File too short, use what we have (pad later if needed)
                offset = 0
            else:
                offset = self.rng.randint(0, max_offset)
        else:
            # Deterministic: iterate over files and chunks
            cum = 0
            file_idx = 0
            # enumerate, not list.index: index() compares dicts holding numpy arrays
            for fi, f in enumerate(self.loaded):
                n_chunks = self._count_chunks(f)
                if cum + n_chunks > idx:
                    file_idx = fi
                    chunk_idx = idx - cum
                    max_offset = f["T"] - self.chunk_frames
                    offset = chunk_idx * self.chunk_frames
                    if offset > max_offset:
                        offset = max_offset
                    break
                cum += n_chunks
        
        file_data = self.loaded[file_idx]
        
        # Extract chunks
        offset = int(offset)
        end_frame = offset + self.chunk_frames
        end_sample = end_frame * 480
        
        # Handle files that might be slightly shorter than expected
        audio = file_data["audio"]
        sensor = file_data["sensor"]
        touch_time = file_data["touch_time"]
        
        audio_chunk = audio[480*offset : 480*end_frame]
        sensor_chunk = sensor[offset:end_frame]
        touch_chunk = touch_time[offset:end_frame]
        
        # Pad if needed (shouldn't happen normally)
        if len(audio_chunk) < self.chunk_samples:
            pad_samples = self.chunk_samples - len(audio_chunk)
            audio_chunk = np.pad(audio_chunk, (0, pad_samples), mode="constant")
            pad_frames = pad_samples // 480
            sensor_chunk = np.pad(sensor_chunk, (0, pad_frames), mode="constant")
            touch_chunk = np.pad(touch_chunk, (0, pad_frames), mode="constant")
        
        # Touch features: log1p(t), exp(-t/0.5), clip(t,0,40) where t = raw touch_time seconds
        t_sec = touch_chunk
        touch_feature_1 = np.log1p(t_sec)  # log1p(t_sec)
        touch_feature_2 = (t_sec > 0) * np.exp(-t_sec / 0.5)  # gated: 0 when not in touch (bug D fix)
        touch_feature_3 = np.clip(t_sec, 0.0, 40.0)  # clip(t_sec, 0, 40)
        # Stack to 3 features
        touch = np.stack([touch_feature_1, touch_feature_2, touch_feature_3], axis=-1)
        
        # Note labels: from npz note_hz if requested and available, else from notes JSON
        if self.use_npz_note_hz and file_data["note_hz"] is not None:
            note_full = file_data["note_hz"]
            if len(note_full) < end_frame:
                note_full = np.pad(note_full, (0, end_frame - len(note_full)), mode="constant")
            note_chunk = np.asarray(note_full[offset:end_frame], dtype=np.float32)
            if len(note_chunk) < self.chunk_frames:
                note_chunk = np.pad(note_chunk, (0, self.chunk_frames - len(note_chunk)), mode="constant")
        else:
            note_key = f"{file_data['sound']}_proximity_curve_{file_data['curve']}"
            note_full = _notes().get(note_key, [0.0] * len(touch_time))
            # Ensure note_full is long enough for the chunk
            if len(note_full) < end_frame:
                # Pad with zeros if note labels are shorter than file
                note_full = note_full + [0.0] * (end_frame - len(note_full))
            note_chunk = np.asarray(note_full[offset:end_frame], dtype=np.float32)
            # Pad note_chunk if needed (should match chunk_frames)
            if len(note_chunk) < self.chunk_frames:
                note_chunk = np.pad(note_chunk, (0, self.chunk_frames - len(note_chunk)), mode="constant")
        
        return {
            "audio": audio_chunk.astype(np.float32),
            "sensor": sensor_chunk,
            "touch": touch,
            "touch_raw": touch_chunk,
            "note": note_chunk,
            "sound_id": file_data["sound_id"],
            "sound": file_data["sound"],
            "curve": file_data["curve"],
            "offset": offset,
        }

    def full(self, sound, curve):
        """
        Get the full file for sound+curve (for rendering).
        Returns same dict structure but with full-length arrays.
        """
        for f in self.loaded:
            if f["sound"] == sound and f["curve"] == curve:
                audio = f["audio"]
                sensor = f["sensor"]
                touch_time = f["touch_time"]
                # Touch features
                t_sec = touch_time
                touch_feature_1 = np.log1p(t_sec)
                touch_feature_2 = (t_sec > 0) * np.exp(-t_sec / 0.5)
                touch_feature_3 = np.clip(t_sec, 0.0, 40.0)
                touch = np.stack([touch_feature_1, touch_feature_2, touch_feature_3], axis=-1)
                
                # Note labels - ensure same length as touch_time
                if self.use_npz_note_hz and f["note_hz"] is not None:
                    note_full = f["note_hz"]
                    if len(note_full) < len(touch_time):
                        note_full = np.pad(note_full, (0, len(touch_time) - len(note_full)), mode="constant")
                    elif len(note_full) > len(touch_time):
                        note_full = note_full[:len(touch_time)]
                    note = np.asarray(note_full, dtype=np.float32)
                else:
                    note_key = f"{f['sound']}_proximity_curve_{f['curve']}"
                    note_full = _notes().get(note_key, [0.0]*len(touch_time))
                    if len(note_full) < len(touch_time):
                        note_full = note_full + [0.0] * (len(touch_time) - len(note_full))
                    elif len(note_full) > len(touch_time):
                        note_full = note_full[:len(touch_time)]
                    note = np.asarray(note_full, dtype=np.float32)
                
                return {
                    "audio": audio,
                    "sensor": sensor,
                    "touch": touch,
                    "touch_raw": touch_time,
                    "note": note,
                    "sound_id": f["sound_id"],
                    "sound": f["sound"],
                    "curve": f["curve"],
                    "offset": 0,
                }
        raise ValueError(f"File not found: sound={sound}, curve={curve}")


# Public alias (train_sensor_ddsp_v5.py imports it under this name).
AlignedChunkDataset = AlignedChunkDatasetV5
