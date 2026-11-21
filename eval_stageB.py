import argparse
import os
import json
import yaml
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter

# Add the project to path for imports
import sys
sys.path.insert(0, "/home/marcello/Documents/interactive_ddsp")

from engine_ddsp.sensor_data import AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import multiscale_fft, safe_log


def compute_touch_from_sensor(sensor, threshold=0.98):
    """
    Compute touch_time from sensor values.
    Rule: run length in seconds of consecutive frames with sensor >= threshold (0 when below).
    This matches the dataset's touch_time computation.
    """
    # Find consecutive runs where sensor >= threshold
    touch_time = np.zeros_like(sensor)
    run_start = 0
    
    for i in range(len(sensor)):
        if sensor[i] >= threshold:
            # Continue the run
            touch_time[run_start:i+1] = (i - run_start + 1) / 100.0  # Convert frames to seconds
        else:
            # Reset run
            run_start = i + 1
    
    return touch_time


def build_synthetic_reverb(sampling_rate=48000, duration_s=1.8, pre_delay_ms=20, seed=42):
    """
    Build a synthetic reverb impulse response.
    - Exponentially decaying noise
    - RT60 about duration_s
    - Pre-delay of pre_delay_ms
    - Low-passed tail
    """
    np.random.seed(seed)
    
    # Pre-delay samples
    pre_delay_samples = int(pre_delay_ms * sampling_rate / 1000)
    
    # Decay constant for RT60
    # RT60 = duration_s, so decay to -60dB at that time
    # decay = exp(-60 * ln(10) / (20 * RT60 * sampling_rate))
    decay_factor = np.exp(-60 * np.log(10) / (20 * duration_s * sampling_rate))
    
    # Number of samples
    n_samples = int(duration_s * sampling_rate)
    
    # Generate exponentially decaying noise
    t = np.arange(n_samples) / sampling_rate
    noise = np.random.randn(n_samples)
    decay = np.exp(-t / (duration_s / 5))  # Fast initial decay
    ir = noise * decay
    
    # Apply pre-delay (shift)
    ir_full = np.zeros(pre_delay_samples + n_samples)
    ir_full[pre_delay_samples:] = ir
    
    # Low-pass filter the tail (simple moving average)
    lp_order = 101
    lp_filter = signal.windows.hann(lp_order)
    lp_filter = lp_filter / lp_filter.sum()
    ir_full = signal.convolve(ir_full, lp_filter, mode='same')
    
    # Normalize to peak = 1
    peak = np.abs(ir_full).max()
    if peak > 0:
        ir_full = ir_full / peak
    
    return ir_full


def compute_multiscale_loss(pred, target, scales, overlap, device='cpu'):
    """Compute multiscale STFT loss on chunks or full file."""
    # Pad or truncate to same length
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    
    # Convert to tensors
    pred_t = torch.from_numpy(pred).unsqueeze(0).to(device)
    target_t = torch.from_numpy(target).unsqueeze(0).to(device)
    
    # Compute loss
    pred_stft = multiscale_fft(pred_t, scales, overlap)
    target_stft = multiscale_fft(target_t, scales, overlap)
    
    loss = 0.0
    for p_s, t_s in zip(pred_stft, target_stft):
        loss = loss + (p_s - t_s).abs().mean() + (safe_log(p_s) - safe_log(t_s)).abs().mean()
    
    return loss.item()


def compute_rms_envelope(audio, frame_size=480, hop=480):
    """Compute log-RMS envelope."""
    n_frames = (len(audio) - frame_size) // hop + 1
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+frame_size]
        rms[i] = np.sqrt(np.mean(frame**2))
    
    # Log RMS (add small epsilon to avoid log(0))
    log_rms = np.log(rms + 1e-8)
    return log_rms


def compute_spectral_centroid_envelope(audio, frame_size=2048, hop=480, sr=48000):
    """Compute spectral centroid envelope."""
    n_frames = (len(audio) - frame_size) // hop + 1
    centroids = np.zeros(n_frames)
    
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+frame_size]
        frame = frame * np.hanning(frame_size)  # Use fixed frame_size for window
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(frame_size, 1/sr)
        centroids[i] = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-8)
    
    return centroids


def compute_hf_ratio(audio, sr=48000, cutoff=6000.0):
    """Compute log-energy above cutoff frequency over total log-energy."""
    frame_size = 2048
    hop = 480
    n_frames = (len(audio) - frame_size) // hop + 1
    hf_ratios = []
    for i in range(n_frames):
        frame = audio[i*hop:i*hop+frame_size]
        frame = frame * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(frame_size, 1/sr)
        # Total energy
        total_energy = (spectrum**2).sum() + 1e-12
        # High-frequency energy (above cutoff)
        hf_mask = freqs > cutoff
        hf_energy = (spectrum[hf_mask]**2).sum() + 1e-12
        # Log-energy ratio
        hf_ratios.append(np.log(hf_energy / total_energy + 1e-8))
    return np.mean(hf_ratios)


def compute_mod_depth(audio, touch_time_raw, sr=48000):
    """Compute std of log-RMS envelope over frames where touch_time_raw > 2 s."""
    frame_size = 480
    hop = 480
    n_blocks = len(audio) // frame_size
    touch_time = touch_time_raw[:n_blocks] if len(touch_time_raw) > n_blocks else touch_time_raw
    mask = (touch_time > 2.0)
    if mask.sum() < 4:
        return 0.0
    rms = np.zeros(n_blocks)
    for i in range(n_blocks):
        frame = audio[i*hop:(i+1)*hop]
        rms[i] = np.sqrt(np.mean(frame**2) + 1e-12)
    log_rms = np.log(rms + 1e-8)
    return np.std(log_rms[mask])


def _estimate_f0_autocorr(x, sr=48000, fmin=50.0, fmax=500.0):
    """Autocorrelation f0 estimate on a short window; None if unvoiced/quiet."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if np.sqrt((x**2).mean()) < 1e-3:
        return None
    ac = np.correlate(x, x, mode='full')[len(x)-1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo, hi = int(sr / fmax), int(sr / fmin)
    lo = max(lo, 1); hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < 0.3:
        return None
    return sr / lag

def compute_touch_onset_f0_error(pred, target, touch_time, sr=48000, window_s=0.5):
    """Mean |f0 error| in cents over the first window_s after each touch onset."""
    min_len = min(len(pred), len(target))
    pred, target = pred[:min_len], target[:min_len]
    tt = touch_time[:len(pred) // 100 + 1] if len(touch_time) > len(pred) // 100 else touch_time
    tt = np.resize(touch_time, (min_len + 99) // 100)[: (min_len // 100) + 1]
    onsets = [i for i in range(1, len(tt)) if tt[i] > 0 and tt[i-1] == 0]
    if tt[0] > 0:
        onsets.insert(0, 0)
    W = int(window_s * sr)
    errs = []
    for o in onsets:
        a0 = o * 480  # touch frame at 100 Hz -> samples (block 480)
        a1 = min(a0 + W, min_len)
        if a1 - a0 < W // 2:
            continue
        f_p = _estimate_f0_autocorr(pred[a0:a1], sr)
        f_t = _estimate_f0_autocorr(target[a0:a1], sr)
        if f_p and f_t:
            errs.append(abs(1200 * np.log2(f_p / f_t)))
    return float(np.mean(errs)) if errs else None

def compute_sustained_touch_corr(pred, target, touch_time, sr=48000):
    """Pearson corr of log-RMS envelopes over samples where touch_time > 2 s."""
    from scipy.stats import pearsonr as _pr
    min_len = min(len(pred), len(target))
    pred, target = pred[:min_len], target[:min_len]
    n_blocks = min_len // 480
    mask = (touch_time[:n_blocks] > 2.0)
    if mask.sum() < 4:
        return None
    idx = np.repeat(mask, 480)[:min_len]
    def env(x):
        e = np.array([np.sqrt(np.mean(x[i*480:(i+1)*480]**2) + 1e-12) for i in range(n_blocks)])
        return np.log(e + 1e-4)
    ep, et = env(pred), env(target)
    m = mask
    if m.sum() < 4:
        return None
    return float(_pr(ep[m], et[m])[0])

def compute_metrics_with_same_len(pred, target, scales, overlap, device='cpu'):
    """Compute metrics ensuring pred and target have same length."""
    # Pad or truncate to same length
    min_len = min(len(pred), len(target))
    pred = pred[:min_len]
    target = target[:min_len]
    
    # (a) Multiscale STFT loss
    multiscale_loss = compute_multiscale_loss(pred, target, scales, overlap, device)
    
    # (b) Pearson correlation of log-RMS envelope
    pred_rms = compute_rms_envelope(pred)
    target_rms = compute_rms_envelope(target)
    
    # (c) Pearson correlation of spectral centroid envelope
    pred_centroid = compute_spectral_centroid_envelope(pred)
    target_centroid = compute_spectral_centroid_envelope(target)
    
    return multiscale_loss, pred_rms, target_rms, pred_centroid, target_centroid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run directory")
    p.add_argument("--ckpt", default="state_final.pth", help="Checkpoint file")
    p.add_argument("--out", required=True, help="Output renders directory")
    p.add_argument("--sounds", nargs="+", default=None, help="Sound names to render (default: all)")
    p.add_argument("--curves", nargs="+", default=["017", "018", "019", "020"], help="Curve IDs to render")
    p.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split (train or val)")
    args = p.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rendering on {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    config_path = os.path.join(args.run, "config.yaml")
    with open(config_path) as f:
        run_config = yaml.safe_load(f)
    
    # Build f0_base list
    with open("/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned/index.json") as f:
        index = json.load(f)
    
    with open("/home/marcello/Documents/interactive_ddsp/bakeoff_data_aligned/f0_base.json") as f:
        f0_base_json = json.load(f)
    
    # Order by sound_ids integer values
    sound_names = sorted(f0_base_json.keys(), key=lambda k: index["sound_ids"][k])
    f0_base = [f0_base_json[s]["f0_hz"] for s in sound_names]
    
    # Create model
    model_config = run_config["model"].copy()
    override = model_config.pop("f0_base_hz_override", None)
    model_config["f0_base_hz"] = override if override else f0_base
    model = SensorDDSP(**model_config).to(device)
    model.eval()
    
    # Load checkpoint
    ckpt_path = os.path.join(args.run, args.ckpt)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Build synthetic reverb IR
    print("\nBuilding synthetic reverb IR...")
    ir = build_synthetic_reverb(sampling_rate=48000, duration_s=1.8, pre_delay_ms=20, seed=42)
    print(f"  IR length: {len(ir)} samples ({len(ir)/48000:.2f}s)")
    
    # Determine sounds to render
    if args.sounds:
        sounds = args.sounds
        curves = args.curves
    else:
        sounds = list(f0_base_json.keys())
        curves = args.curves
    
    # Process each sound/curve combination
    all_metrics = []
    metrics_by_sound = {s: [] for s in sounds}
    
    scales = run_config["train"]["scales"]
    overlap = run_config["train"]["overlap"]
    chunk_s = run_config["data"]["chunk_s"]
    
    print(f"\nRendering {len(sounds) * len(curves)} combinations...")
    
    # Dataset for loading full files
    root = run_config["data"]["root"]
    full_dataset = AlignedChunkDataset(
        root, split=args.split, chunk_s=chunk_s,
        epoch_len=None, seed=0, train=False  # train=False for deterministic
    )
    
    for sound in sounds:
        for curve in curves:
            print(f"\n  {sound}_{curve}...")
            
            try:
                item = full_dataset.full(sound, curve)
            except ValueError as e:
                print(f"    File not found: {e}")
                continue
            
            audio = item["audio"]
            sensor = item["sensor"]
            touch = item["touch"]
            sound_id = item["sound_id"]
            
            # Convert to tensor
            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
            sensor_t = torch.from_numpy(sensor).unsqueeze(0).unsqueeze(-1).to(device)
            touch_t = torch.from_numpy(touch).unsqueeze(0).to(device)
            if touch_t.dim() == 2:
                touch_t = touch_t.unsqueeze(-1)
            sound_id_t = torch.tensor([sound_id], dtype=torch.long, device=device)
            
            # Run model in eval mode
            try:
                with torch.no_grad():
                    pred = model(sensor_t, touch_t, sound_id_t).squeeze(0).squeeze(-1).cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("    GPU OOM, falling back to chunked rendering...")
                    # Chunked rendering with 30s windows and 2s crossfade
                    chunk_size = int(30 * 48000)
                    overlap_size = int(2 * 48000)
                    hop = chunk_size - overlap_size
                    
                    n_chunks = (len(audio) - overlap_size) // hop + 1
                    pred = np.zeros_like(audio)
                    
                    for i in range(n_chunks):
                        start = i * hop
                        end = min(start + chunk_size, len(audio))
                        if end - start < 1000:  # Minimum chunk size
                            break
                        
                        # Extract chunk
                        sensor_chunk = torch.from_numpy(sensor[start//480:end//480]).unsqueeze(0).unsqueeze(-1).to(device)
                        touch_chunk = torch.from_numpy(touch[start//480:end//480]).unsqueeze(0).to(device)
                        if touch_chunk.dim() == 2:
                            touch_chunk = touch_chunk.unsqueeze(-1)
                        audio_target_chunk = audio[start:end]
                        
                        with torch.no_grad():
                            pred_chunk = model(sensor_chunk, touch_chunk, sound_id_t).squeeze(0).squeeze(-1).cpu().numpy()
                        
                        # Crossfade
                        fade_samples = overlap_size // 480
                        fade_in = np.linspace(0, 1, fade_samples)
                        fade_out = np.linspace(1, 0, fade_samples)
                        
                        pred_chunk = pred_chunk[:len(audio_target_chunk)]
                        
                        if i == 0:
                            # First chunk: no fade out
                            pred[start:end] = pred_chunk
                        elif i == n_chunks - 1:
                            # Last chunk: no fade in
                            pred[start:end] = pred_chunk
                        else:
                            # Middle chunks: crossfade
                            fade_start = i * hop
                            fade_end = fade_start + overlap_size
                            pred[fade_start:fade_end] = (
                                pred[fade_start:fade_end] * fade_out[:overlap_size] + 
                                pred_chunk[:overlap_size] * fade_in[:overlap_size]
                            )
                            # Outside fade region
                            pred[fade_end:end] = pred_chunk[overlap_size:]
                else:
                    raise
            
            # Write dry files with COMMON gain for loudness comparability
            pred_dry_path = os.path.join(args.out, f"{sound}_{curve}_pred.wav")
            target_dry_path = os.path.join(args.out, f"{sound}_{curve}_target.wav")
            
            # Compute peak for both
            pred_peak = np.abs(pred).max()
            target_peak = np.abs(audio).max()
            # Common gain = 1 / max(peak_pred, peak_target)
            common_gain = 1.0 / max(pred_peak, target_peak)
            
            # Apply common gain to both
            pred_norm = pred * common_gain
            target_norm = audio * common_gain
            
            sf.write(pred_dry_path, pred_norm, 48000, subtype='PCM_16')
            sf.write(target_dry_path, target_norm, 48000, subtype='PCM_16')
            print(f"    Dry files: {os.path.basename(pred_dry_path)}, {os.path.basename(target_dry_path)}")
            print(f"    Common gain: {common_gain:.6f} (pred_peak={pred_peak:.4f}, target_peak={target_peak:.4f})")
            
            # Apply synthetic reverb (25% wet, 75% dry)
            pred_reverb = 0.75 * pred_norm + 0.25 * signal.convolve(pred_norm, ir, mode='same')
            target_reverb = 0.75 * target_norm + 0.25 * signal.convolve(target_norm, ir, mode='same')
            
            # Re-peak normalize (they might have changed peak after reverb)
            pred_reverb = pred_reverb / (np.abs(pred_reverb).max() + 1e-8)
            target_reverb = target_reverb / (np.abs(target_reverb).max() + 1e-8)
            
            pred_reverb_path = os.path.join(args.out, f"{sound}_{curve}_pred_reverb.wav")
            target_reverb_path = os.path.join(args.out, f"{sound}_{curve}_target_reverb.wav")
            
            sf.write(pred_reverb_path, pred_reverb, 48000, subtype='PCM_16')
            sf.write(target_reverb_path, target_reverb, 48000, subtype='PCM_16')
            print(f"    Reverb files: {os.path.basename(pred_reverb_path)}, {os.path.basename(target_reverb_path)}")
            
            # Compute metrics on dry pred vs target (on unnormalised audio)
            # Ensure same length first
            metrics = compute_metrics_with_same_len(pred, audio, scales, overlap, device)
            multiscale_loss, pred_rms, target_rms, pred_centroid, target_centroid = metrics
            
            # Compute correlations
            rms_corr, _ = pearsonr(pred_rms, target_rms)
            centroid_corr, _ = pearsonr(pred_centroid, target_centroid)
            
            # (d) RMS ratio (on unnormalised audio)
            pred_rms_val = np.sqrt(np.mean(pred**2))
            target_rms_val = np.sqrt(np.mean(audio**2))
            rms_ratio = pred_rms_val / (target_rms_val + 1e-8)
            
            touch_time_raw = item["touch_raw"] if "touch_raw" in item else np.expm1(touch[..., 0])  # raw seconds
            onset_f0_err = compute_touch_onset_f0_error(pred, audio, touch_time_raw)
            sustained_corr = compute_sustained_touch_corr(pred, audio, touch_time_raw)
            # New metrics: HF ratio (log-energy above 6kHz over total) and mod depth (std of log-RMS where touch>2s)
            hf_ratio_pred = compute_hf_ratio(pred)
            hf_ratio_tgt = compute_hf_ratio(audio)
            mod_depth_pred = compute_mod_depth(pred, touch_time_raw)
            mod_depth_tgt = compute_mod_depth(audio, touch_time_raw)
            metrics = {
                "sound": sound,
                "curve": curve,
                "multiscale_loss": multiscale_loss,
                "rms_envelope_corr": rms_corr if not np.isnan(rms_corr) else 0,
                "centroid_corr": centroid_corr if not np.isnan(centroid_corr) else 0,
                "rms_ratio": rms_ratio,
                "touch_onset_f0_err_cents": onset_f0_err if onset_f0_err is not None else -1.0,
                "sustained_touch_corr": sustained_corr if sustained_corr is not None else -1.0,
                "hf_ratio_pred": hf_ratio_pred,
                "hf_ratio_tgt": hf_ratio_tgt,
                "mod_depth_pred": mod_depth_pred,
                "mod_depth_tgt": mod_depth_tgt,
            }
            all_metrics.append(metrics)
            metrics_by_sound[sound].append(metrics)
            
            print(f"    Metrics: MS loss={multiscale_loss:.4f}, RMS corr={rms_corr:.4f}, Centroid corr={centroid_corr:.4f}, RMS ratio={rms_ratio:.4f}")
            
            # (e) Sensor ablation
            print(f"    Sensor ablation...")
            
            # (i) sensor=0, touch=0
            sensor_zero = torch.zeros_like(sensor_t)
            touch_zero = torch.zeros_like(touch_t)
            with torch.no_grad():
                pred_abl_zero = model(sensor_zero, touch_zero, sound_id_t).squeeze(0).squeeze(-1).cpu().numpy()
            
            multiscale_abl_zero = compute_multiscale_loss(pred_abl_zero, audio, scales, overlap, device)
            pred_rms_abl_zero = compute_rms_envelope(pred_abl_zero)
            rms_corr_abl_zero, _ = pearsonr(pred_rms_abl_zero, pred_rms)
            
            # (ii) sensor circularly rolled, touch recomputed with dataset rule
            # Dataset rule: run length in seconds of consecutive frames with sensor >= 0.98
            sensor_rolled = np.roll(sensor, len(sensor) // 2)
            touch_time_rolled = compute_touch_from_sensor(sensor_rolled, threshold=0.98)
            touch_rolled = np.clip(np.stack([np.log1p(touch_time_rolled), np.exp(-touch_time_rolled / 0.5)], axis=-1), 0.0, 10.0)  # 2-feature touch, dataset rule
            
            # Verify against stored touch_time
            stored_touch_time = item["touch_raw"]  # This is touch_time
            diff = np.abs(touch_time_rolled - stored_touch_time).max()
            if diff > 0.001:
                print(f"    WARNING: Touch_time recomputation diff={diff:.6f} from stored")
            else:
                print(f"    Touch_time recomputation verified (max diff={diff:.6f})")
            
            # Also verify touch
            stored_touch = item["touch"]
            touch_diff = np.abs(touch_rolled - stored_touch).max()
            if touch_diff > 0.001:
                print(f"    WARNING: Touch recomputation diff={touch_diff:.6f} from stored")
            else:
                print(f"    Touch recomputation verified (max diff={touch_diff:.6f})")
            
            sensor_rolled_t = torch.from_numpy(sensor_rolled).unsqueeze(0).unsqueeze(-1).to(device)
            touch_rolled_t = torch.from_numpy(touch_rolled).unsqueeze(0).to(device)
            if touch_rolled_t.dim() == 2:
                touch_rolled_t = touch_rolled_t.unsqueeze(-1)
            
            with torch.no_grad():
                pred_abl_rolled = model(sensor_rolled_t, touch_rolled_t, sound_id_t).squeeze(0).squeeze(-1).cpu().numpy()
            
            multiscale_abl_rolled = compute_multiscale_loss(pred_abl_rolled, audio, scales, overlap, device)
            pred_rms_abl_rolled = compute_rms_envelope(pred_abl_rolled)
            rms_corr_abl_rolled, _ = pearsonr(pred_rms_abl_rolled, pred_rms)
            
            print(f"      Ablation zero: MS loss={multiscale_abl_zero:.4f}, RMS corr vs normal={rms_corr_abl_zero:.4f}")
            print(f"      Ablation rolled: MS loss={multiscale_abl_rolled:.4f}, RMS corr vs normal={rms_corr_abl_rolled:.4f}")
            
            # Save ablation results with SAME common_gain as pred/target dry (no per-file peak normalization)
            abl_zero_path = os.path.join(args.out, f"{sound}_{curve}_pred_abl_zero.wav")
            abl_rolled_path = os.path.join(args.out, f"{sound}_{curve}_pred_abl_rolled.wav")
            
            sf.write(abl_zero_path, pred_abl_zero * common_gain, 48000, subtype='PCM_16')
            sf.write(abl_rolled_path, pred_abl_rolled * common_gain, 48000, subtype='PCM_16')
            
            # Update metrics with ablation
            metrics["multiscale_abl_zero"] = multiscale_abl_zero
            metrics["rms_corr_abl_zero"] = rms_corr_abl_zero if not np.isnan(rms_corr_abl_zero) else 0
            metrics["multiscale_abl_rolled"] = multiscale_abl_rolled
            metrics["rms_corr_abl_rolled"] = rms_corr_abl_rolled if not np.isnan(rms_corr_abl_rolled) else 0
    
    # Write metrics.json
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump([{k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in m.items()} for m in all_metrics], f, indent=2)
    
    # Compute means
    overall_mean = {k: np.mean([m[k] for m in all_metrics]) for k in ["multiscale_loss", "rms_envelope_corr", "centroid_corr", "rms_ratio"]}
    overall_mean["multiscale_abl_zero"] = np.mean([m.get("multiscale_abl_zero", 0) for m in all_metrics])
    overall_mean["rms_corr_abl_zero"] = np.mean([m.get("rms_corr_abl_zero", 0) for m in all_metrics])
    overall_mean["multiscale_abl_rolled"] = np.mean([m.get("multiscale_abl_rolled", 0) for m in all_metrics])
    overall_mean["rms_corr_abl_rolled"] = np.mean([m.get("rms_corr_abl_rolled", 0) for m in all_metrics])
    overall_mean["hf_ratio_pred"] = np.mean([m.get("hf_ratio_pred", 0) for m in all_metrics])
    overall_mean["hf_ratio_tgt"] = np.mean([m.get("hf_ratio_tgt", 0) for m in all_metrics])
    overall_mean["mod_depth_pred"] = np.mean([m.get("mod_depth_pred", 0) for m in all_metrics])
    overall_mean["mod_depth_tgt"] = np.mean([m.get("mod_depth_tgt", 0) for m in all_metrics])
    
    sound_means = {}
    for sound in sounds:
        sound_metrics = [m for m in all_metrics if m["sound"] == sound]
        if sound_metrics:
            sound_means[sound] = {k: np.mean([m[k] for m in sound_metrics]) for k in ["multiscale_loss", "rms_envelope_corr", "centroid_corr", "rms_ratio"]}
    
    # Write STAGE_B_REPORT.md
    report_path = os.path.join(args.out, "STAGE_B_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Stage B Evaluation Report\n\n")
        f.write("## Run Configuration\n\n")
        f.write(f"- Run directory: {args.run}\n")
        f.write(f"- Checkpoint: {args.ckpt}\n")
        f.write(f"- Output directory: {args.out}\n")
        f.write(f"- Sounds: {sounds}\n")
        f.write(f"- Curves: {curves}\n")
        f.write(f"- Dataset split: {args.split}\n\n")
        
        f.write("## Validation Loss vs Silence\n\n")
        f.write("| Sound | Multiscale Loss | Silence Loss | Loss Ratio |\n")
        f.write("|-------|------------------|---------------|------------|\n")
        for sound in sounds:
            if sound in sound_means:
                m = sound_means[sound]
                silence_loss = 23.216  # Silence loss from training log
                loss_ratio = m["multiscale_loss"] / silence_loss
                f.write(f"| {sound} | {m['multiscale_loss']:.6f} | {silence_loss:.6f} | {loss_ratio:.6f} |\n")
        
        f.write("\n## Metrics\n\n")
        f.write("### Overall Mean\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Multiscale STFT Loss | {overall_mean['multiscale_loss']:.6f} |\n")
        f.write(f"| RMS Envelope Correlation | {overall_mean['rms_envelope_corr']:.6f} |\n")
        f.write(f"| Spectral Centroid Correlation | {overall_mean['centroid_corr']:.6f} |\n")
        f.write(f"| RMS Ratio (pred/target) | {overall_mean['rms_ratio']:.6f} |\n")
        f.write(f"| Multiscale Loss (ablation zero) | {overall_mean['multiscale_abl_zero']:.6f} |\n")
        f.write(f"| RMS Corr vs Normal (ablation zero) | {overall_mean['rms_corr_abl_zero']:.6f} |\n")
        f.write(f"| Multiscale Loss (ablation rolled) | {overall_mean['multiscale_abl_rolled']:.6f} |\n")
        f.write(f"| RMS Corr vs Normal (ablation rolled) | {overall_mean['rms_corr_abl_rolled']:.6f} |\n")
        
        f.write("\n### Per-Sound Means\n\n")
        f.write("| Sound | MS Loss | RMS Env Corr | Centroid Corr | RMS Ratio |\n")
        f.write("|-------|---------|--------------|---------------|-----------|\n")
        for sound in sounds:
            if sound in sound_means:
                m = sound_means[sound]
                f.write(f"| {sound} | {m['multiscale_loss']:.6f} | {m['rms_envelope_corr']:.6f} | ")
                f.write(f"{m['centroid_corr']:.6f} | {m['rms_ratio']:.6f} |\n")
        
        f.write("\n## Ablation Results\n\n")
        f.write("| Sound | Ablation Zero MS Loss | Ablation Rolled MS Loss | RMS Corr Zero | RMS Corr Rolled |\n")
        f.write("|-------|------------------------|--------------------------|---------------|------------------|\n")
        for sound in sounds:
            if sound in sound_means:
                m = sound_means[sound]
                f.write(f"| {sound} | {m.get('multiscale_abl_zero', 0):.6f} | {m.get('multiscale_abl_rolled', 0):.6f} | ")
                f.write(f"{m.get('rms_corr_abl_zero', 0):.6f} | {m.get('rms_corr_abl_rolled', 0):.6f} |\n")
        
        f.write("\n## Pass/Fail Criteria\n\n")
        f.write("- Audible: peak > 1e-3, RMS 0.4-2.0\n")
        f.write("- Loss ratio < 0.6\n")
        f.write("- Ablation shuffled < 0.5\n")
        f.write("- Band-shape > 0.65\n\n")
        
        f.write("## Notes\n\n")
        f.write("- 04_modularPercussions expected weaker fit\n")
    
    print(f"\nResults written to: {args.out}")
    print(f"  - metrics.json: Per-file metrics")
    print(f"  - STAGE_B_REPORT.md: Summary report")
    print(f"\nDry files: pred vs target (common gain applied)")
    print(f"Reverb files: synthetic IR (1.8s, 25% wet)")
    print(f"Ablation files: sensor=0, touch=0 and sensor rolled")

if __name__ == "__main__":
    main()