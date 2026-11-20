#!/usr/bin/env python
"""
SensorDDSP rendering script.
Renders audio from the trained model and computes metrics.
"""
import argparse
import os
import sys
import json
import yaml
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter

# Add the project to path for imports
sys.path.insert(0, "/home/marcello/Documents/interactive_ddsp")

from engine_ddsp.sensor_data import AlignedChunkDataset
from engine_ddsp.sensor_ddsp import SensorDDSP
from engine_ddsp.synth import multiscale_fft, safe_log


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
    p.add_argument("--ckpt", default="state_best.pth", help="Checkpoint file")
    p.add_argument("--out", required=True, help="Output renders directory")
    p.add_argument("--sounds", nargs="+", default=None, help="Sound names to render (default: all)")
    p.add_argument("--curves", nargs="+", default=["017", "018", "019", "020"], help="Curve IDs to render")
    p.add_argument("--overfit_file", action="store_true", help="Render overfit file (03_harmonicSwarm_001)")
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
    with open("bakeoff_data_aligned/index.json") as f:
        index = json.load(f)
    
    with open("bakeoff_data_aligned/f0_base.json") as f:
        f0_base_json = json.load(f)
    
    # Order by sound_ids integer values
    sound_names = sorted(f0_base_json.keys(), key=lambda k: index["sound_ids"][k])
    f0_base = [f0_base_json[s]["f0_hz"] for s in sound_names]
    
    # Create model
    model_config = run_config["model"].copy()
    model_config["f0_base_hz"] = f0_base
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
    if args.overfit_file:
        sounds = ["03_harmonicSwarm"]
        curves = ["001"]
    elif args.sounds:
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
        root, split="val", chunk_s=chunk_s,
        epoch_len=None, seed=0, train=False
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
            touch_t = torch.from_numpy(touch).unsqueeze(0).unsqueeze(-1).to(device)
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
                        touch_chunk = torch.from_numpy(touch[start//480:end//480]).unsqueeze(0).unsqueeze(-1).to(device)
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
            
            # Write dry files
            pred_dry_path = os.path.join(args.out, f"{sound}_{curve}_pred.wav")
            target_dry_path = os.path.join(args.out, f"{sound}_{curve}_target.wav")
            
            # Peak normalize
            pred_norm = pred / (np.abs(pred).max() + 1e-8)
            target_norm = audio / (np.abs(audio).max() + 1e-8)
            
            sf.write(pred_dry_path, pred_norm, 48000, subtype='PCM_16')
            sf.write(target_dry_path, target_norm, 48000, subtype='PCM_16')
            print(f"    Dry files: {os.path.basename(pred_dry_path)}, {os.path.basename(target_dry_path)}")
            
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
            
            # Compute metrics on dry pred vs target
            # Ensure same length first
            metrics = compute_metrics_with_same_len(pred, audio, scales, overlap, device)
            multiscale_loss, pred_rms, target_rms, pred_centroid, target_centroid = metrics
            
            # Compute correlations
            rms_corr, _ = pearsonr(pred_rms, target_rms)
            centroid_corr, _ = pearsonr(pred_centroid, target_centroid)
            
            # (d) RMS ratio
            pred_rms_val = np.sqrt(np.mean(pred**2))
            target_rms_val = np.sqrt(np.mean(audio**2))
            rms_ratio = pred_rms_val / (target_rms_val + 1e-8)
            
            metrics = {
                "sound": sound,
                "curve": curve,
                "multiscale_loss": multiscale_loss,
                "rms_envelope_corr": rms_corr if not np.isnan(rms_corr) else 0,
                "centroid_corr": centroid_corr if not np.isnan(centroid_corr) else 0,
                "rms_ratio": rms_ratio,
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
            
            # (ii) sensor circularly rolled, touch recomputed
            sensor_rolled = np.roll(sensor, len(sensor) // 2)
            touch_rolled = np.clip(sensor_rolled / 30.0, 0.0, 2.0)
            
            sensor_rolled_t = torch.from_numpy(sensor_rolled).unsqueeze(0).unsqueeze(-1).to(device)
            touch_rolled_t = torch.from_numpy(touch_rolled).unsqueeze(0).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                pred_abl_rolled = model(sensor_rolled_t, touch_rolled_t, sound_id_t).squeeze(0).squeeze(-1).cpu().numpy()
            
            multiscale_abl_rolled = compute_multiscale_loss(pred_abl_rolled, audio, scales, overlap, device)
            pred_rms_abl_rolled = compute_rms_envelope(pred_abl_rolled)
            rms_corr_abl_rolled, _ = pearsonr(pred_rms_abl_rolled, pred_rms)
            
            print(f"      Ablation zero: MS loss={multiscale_abl_zero:.4f}, RMS corr vs normal={rms_corr_abl_zero:.4f}")
            print(f"      Ablation rolled: MS loss={multiscale_abl_rolled:.4f}, RMS corr vs normal={rms_corr_abl_rolled:.4f}")
            
            # Save ablation results
            abl_zero_path = os.path.join(args.out, f"{sound}_{curve}_pred_abl_zero.wav")
            abl_rolled_path = os.path.join(args.out, f"{sound}_{curve}_pred_abl_rolled.wav")
            
            sf.write(abl_zero_path, pred_abl_zero / (np.abs(pred_abl_zero).max() + 1e-8), 48000, subtype='PCM_16')
            sf.write(abl_rolled_path, pred_abl_rolled / (np.abs(pred_abl_rolled).max() + 1e-8), 48000, subtype='PCM_16')
            
            # Update metrics with ablation
            metrics["multiscale_abl_zero"] = multiscale_abl_zero
            metrics["rms_corr_abl_zero"] = rms_corr_abl_zero if not np.isnan(rms_corr_abl_zero) else 0
            metrics["multiscale_abl_rolled"] = multiscale_abl_rolled
            metrics["rms_corr_abl_rolled"] = rms_corr_abl_rolled if not np.isnan(rms_corr_abl_rolled) else 0
    
    # Write metrics.csv
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w") as f:
        f.write("sound,curve,multiscale_loss,rms_envelope_corr,centroid_corr,rms_ratio,"
                "multiscale_abl_zero,rms_corr_abl_zero,multiscale_abl_rolled,rms_corr_abl_rolled\n")
        for m in all_metrics:
            f.write(f"{m['sound']},{m['curve']},{m['multiscale_loss']:.6f},{m['rms_envelope_corr']:.6f},"
                    f"{m['centroid_corr']:.6f},{m['rms_ratio']:.6f},"
                    f"{m.get('multiscale_abl_zero', 0):.6f},{m.get('rms_corr_abl_zero', 0):.6f},"
                    f"{m.get('multiscale_abl_rolled', 0):.6f},{m.get('rms_corr_abl_rolled', 0):.6f}\n")
    
    # Compute means
    overall_mean = {k: np.mean([m[k] for m in all_metrics]) for k in ["multiscale_loss", "rms_envelope_corr", "centroid_corr", "rms_ratio"]}
    overall_mean["multiscale_abl_zero"] = np.mean([m.get("multiscale_abl_zero", 0) for m in all_metrics])
    overall_mean["rms_corr_abl_zero"] = np.mean([m.get("rms_corr_abl_zero", 0) for m in all_metrics])
    overall_mean["multiscale_abl_rolled"] = np.mean([m.get("multiscale_abl_rolled", 0) for m in all_metrics])
    overall_mean["rms_corr_abl_rolled"] = np.mean([m.get("rms_corr_abl_rolled", 0) for m in all_metrics])
    
    sound_means = {}
    for sound in sounds:
        sound_metrics = [m for m in all_metrics if m["sound"] == sound]
        if sound_metrics:
            sound_means[sound] = {k: np.mean([m[k] for m in sound_metrics]) for k in ["multiscale_loss", "rms_envelope_corr", "centroid_corr", "rms_ratio"]}
    
    # Write metrics_summary.md
    md_path = os.path.join(args.out, "metrics_summary.md")
    with open(md_path, "w") as f:
        f.write("# Sensor DDSP Rendering Metrics\n\n")
        f.write("## Dry files: pred vs target\n\n")
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
                f.write(f"| {sound} | {m['multiscale_loss']:.6f} | {m['rms_envelope_corr']:.6f} | "
                        f"{m['centroid_corr']:.6f} | {m['rms_ratio']:.6f} |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- Dry files: `<sound>_<curve>_pred.wav`, `<sound>_<curve>_target.wav`\n")
        f.write("- Reverb files (synthetic 1.8s IR, 25% wet): `<sound>_<curve>_pred_reverb.wav`, `<sound>_<curve>_target_reverb.wav`\n")
        f.write("- Ablation files: `<sound>_<curve>_pred_abl_zero.wav` (sensor=0, touch=0), "
                "`<sound>_<curve>_pred_abl_rolled.wav` (sensor rolled)\n")
        f.write("- Multiscale STFT loss computed on full file (or 10s windows if needed).\n")
        f.write("- RMS envelope: 480-sample frames, log-RMS.\n")
        f.write("- Spectral centroid: 2048-sample window, 480-sample hop.\n")
    
    print(f"\nResults written to: {args.out}")
    print(f"  - metrics.csv: Per-file metrics")
    print(f"  - metrics_summary.md: Summary tables")
    print(f"\nDry files: pred vs target")
    print(f"Reverb files: synthetic IR (1.8s, 25% wet)")


if __name__ == "__main__":
    main()
