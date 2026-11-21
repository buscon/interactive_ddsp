"""
SensorDDSP dataset and f0 estimation utilities.
Loads aligned proximity sensor + audio data from bakeoff_data_aligned.
"""
import json
import os
import random
import numpy as np
import json as _json
_NOTES = None
def _notes():
    global _NOTES
    if _NOTES is None:
        import os
        p = os.path.join(os.path.dirname(__file__), "..", "bakeoff_notes.json")
        _NOTES = _json.load(open(p)) if os.path.exists(p) else {}
    return _NOTES
import librosa
import torch
from torch.utils.data import Dataset


class AlignedChunkDataset(Dataset):
    """
    Dataset of aligned sensor + audio chunks.
    
    Data format:
    - index.json: source_dir, n_files, sound_ids dict, train_curves, val_curves, files list
    - each .npz: audio (int16, n_samples,), sensor (float32, T,), touch_time (float32, T,)
      with T = n_samples // 480 (100 Hz frame rate)
    
    Chunking:
    - chunk_s seconds -> F = int(chunk_s * 100) frames
    - audio chunk: audio[480*offset : 480*(offset+F)]
    - sensor chunk: sensor[offset:offset+F]
    - touch chunk: touch_time[offset:offset+F]
    """
    
    PROVENANCE_ROOT = "/home/marcello/sentire/Recordings/Dataset_20261120_110047"
    PROVENANCE_N_FILES = 100
    
    def __init__(self, root, split, chunk_s=10.0, sounds=None, curves=None,
                 epoch_len=None, seed=0, train=True):
        """
        Args:
            root: path to bakeoff_data_aligned
            split: "train" or "val"
            chunk_s: chunk duration in seconds (default 10.0)
            sounds: list of sound names to include, or None for all
            curves: list of curve IDs to include, or None for all in split
            epoch_len: number of items per epoch (default: n_files * 20 for train)
            seed: random seed for shuffling
            train: True for random chunks, False for deterministic non-overlapping
        """
        self.root = root
        self.split = split
        self.chunk_s = chunk_s
        self.train = train
        
        # Load index
        index_path = os.path.join(root, "index.json")
        with open(index_path) as f:
            self.index = json.load(f)
        
        # Provenance guard
        if self.index["source_dir"] != self.PROVENANCE_ROOT:
            raise ValueError(
                f"Provenance mismatch: expected {self.PROVENANCE_ROOT}, "
                f"got {self.index['source_dir']}"
            )
        if self.index["n_files"] != self.PROVENANCE_N_FILES:
            raise ValueError(
                f"Provenance mismatch: expected {self.PROVENANCE_N_FILES} files, "
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
        # self.index["sound_ids"] is a dict {"sound_name": sound_id_int, ...}
        self.sound_ids = self.index["sound_ids"]
        self.sound_id_map = self.sound_ids.copy()  # Copy dict with correct integer IDs
        
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
                self.epoch_len = sum(self._count_chunks(f) for f in self.files)
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
            # audio: int16 -> float32 normalized
            audio = data["audio"].astype(np.float32) / 32768.0
            sensor = data["sensor"].astype(np.float32)
            touch_time = data["touch_time"].astype(np.float32)
            sound_id = self.sound_id_map[f["sound"]]
            self.loaded.append({
                "audio": audio,
                "sensor": sensor,
                "touch_time": touch_time,
                "sound_id": sound_id,
                "sound": f["sound"],
                "curve": f["curve"],
                "n_samples": int(f["n_samples"]),
                "T": int(f["T"]),
            })
    
    def _count_chunks(self, file_entry):
        """Count how many non-overlapping chunks fit in a file."""
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
            for f in self.loaded:
                n_chunks = self._count_chunks(f)
                if cum + n_chunks > idx:
                    file_idx = self.loaded.index(f)
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
        
        # Touch features: log1p(t/30) and exp(-t/30/0.5), each clipped to [0, ~10]
        # t_sec = touch_time in seconds (touch_time is already in seconds from 0..~30)
        t_sec = touch_chunk
        touch_feature_1 = np.log1p(t_sec)  # log1p(t_sec)
        touch_feature_2 = np.exp(-t_sec / 0.5)  # exp(-t_sec/0.5)
        # Stack and clip to [0, ~10]
        touch = np.stack([touch_feature_1, touch_feature_2], axis=-1)
        touch = np.clip(touch, 0.0, 10.0)
        
        return {
            "audio": audio_chunk.astype(np.float32),
            "sensor": sensor_chunk,
            "touch": touch,
            "touch_raw": touch_chunk,
            "note": np.asarray(_notes().get(f"{file_data['sound']}_proximity_curve_{file_data['curve']}", [0.0]*len(touch_time)), dtype=np.float32)[offset:end_frame],
                "note": _notes().get(f"{file_data['sound']}_proximity_curve_{file_data['curve']}", [0.0]*len(touch_chunk))[offset:end_frame] if _notes() else np.zeros(len(touch_chunk), dtype=np.float32),
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
                # Touch features: log1p(t/30) and exp(-t/30/0.5)
                t_sec = touch_time
                touch_feature_1 = np.log1p(t_sec)
                touch_feature_2 = np.exp(-t_sec / 0.5)
                touch = np.stack([touch_feature_1, touch_feature_2], axis=-1)
                touch = np.clip(touch, 0.0, 10.0)
                return {
                    "audio": audio,
                    "sensor": sensor,
                    "touch": touch,
                    "touch_raw": touch_time,
                    "note": np.asarray(_notes().get(f"{f['sound']}_proximity_curve_{f['curve']}", [0.0]*len(touch_time)), dtype=np.float32),
                    "sound_id": f["sound_id"],
                    "sound": f["sound"],
                    "curve": f["curve"],
                    "offset": 0,
                }
        raise ValueError(f"File not found: sound={sound}, curve={curve}")


def estimate_f0_base(root, n_files=3, excerpt_s=20, fmin=50, fmax=1000):
    """
    Estimate base f0 per sound using pYIN on excerpts.
    
    Args:
        root: path to bakeoff_data_aligned
        n_files: number of TRAIN files per sound to sample
        excerpt_s: duration of excerpt per file in seconds
        fmin, fmax: f0 range for pyin
    
    Returns:
        dict {sound: {"f0_hz": float, "voiced_frac": float, "voiced": bool}}
        Also cached to bakeoff_data_aligned/f0_base.json
    """
    with open(os.path.join(root, "index.json")) as f:
        index = json.load(f)
    
    # Get train files per sound
    train_files = [f for f in index["files"] if f["split"] == "train"]
    
    # Group by sound
    files_by_sound = {}
    for f in train_files:
        sound = f["sound"]
        if sound not in files_by_sound:
            files_by_sound[sound] = []
        files_by_sound[sound].append(f)
    
    # Estimate for each sound
    result = {}
    for sound, files in files_by_sound.items():
        # Sample n_files
        samples = files[:min(n_files, len(files))]
        
        f0_list = []
        total_voiced = 0
        total_frames = 0
        
        for sf in samples:
            # Load and extract excerpt
            data = np.load(sf["path"])
            audio = data["audio"].astype(np.float32) / 32768.0
            sr = 48000
            
            # Downsample for speed
            target_sr = 16000
            audio_ds = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # Extract excerpt
            n_samples_excerpt = int(excerpt_s * target_sr)
            start = 0
            excerpt = audio_ds[start:start+n_samples_excerpt]
            
            # pYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                excerpt, fmin=fmin, fmax=fmax, sr=target_sr
            )
            
            # Collect voiced f0 values
            voiced_mask = ~np.isnan(f0)
            f0_voiced = f0[voiced_mask]
            f0_list.extend(f0_voiced)
            total_voiced += voiced_mask.sum()
            total_frames += len(f0)
        
        # Compute stats
        if len(f0_list) > 0 and total_voiced / total_frames >= 0.10:
            base_f0 = float(np.median(f0_list))
            voiced_frac = float(total_voiced / total_frames)
            voiced = True
        else:
            base_f0 = 220.0
            voiced_frac = 0.0 if total_frames > 0 else 0.0
            voiced = False
        
        result[sound] = {
            "f0_hz": base_f0,
            "voiced_frac": voiced_frac,
            "voiced": voiced,
        }
    
    # Cache
    cache_path = os.path.join(root, "f0_base.json")
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    
    # Print table
    print("\nf0 base estimation:")
    print(f"{'Sound':<25} {'f0_hz':>8} {'voiced':>8} {'voiced_frac':>12}")
    print("-" * 55)
    for sound, info in sorted(result.items()):
        print(f"{sound:<25} {info['f0_hz']:>8.1f} {str(info['voiced']):>8} {info['voiced_frac']:>12.3f}")
    print()
    
    return result


if __name__ == "__main__":
    # Quick test
    root = "/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned"
    dataset = AlignedChunkDataset(root, "train", chunk_s=1.0, sounds=["01_canopy"])
    print(f"Dataset length: {len(dataset)}")
    item = dataset[0]
    print(f"Item keys: {list(item.keys())}")
    print(f"Audio shape: {item['audio'].shape}")
    print(f"Sensor shape: {item['sensor'].shape}")
    print(f"Touch shape: {item['touch'].shape}")
    print(f"Sound ID: {item['sound_id']}")
